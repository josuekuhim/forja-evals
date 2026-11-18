"""Cliente assíncrono para qualquer endpoint OpenAI-compatível (forja-serve, vLLM, APIs externas)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from forja_evals.cache import Cache

RETRYABLE_STATUS = {408, 409, 429}


class Endpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    model: str
    api_key_env: str | None = None  # nome da variável de ambiente; a chave nunca vai pra config
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0
    # Campos extras específicos do servidor, mesclados no corpo da requisição. Ex.: no llama.cpp,
    # {"cache_prompt": false} evita que o reuso de KV cache entre requisições mude a saída.
    extra_body: dict[str, Any] = {}

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"variável de ambiente {self.api_key_env} não definida")
        return key


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cost_usd: float
    cached: bool = False


class ChatClient:
    def __init__(
        self,
        endpoint: Endpoint,
        *,
        concurrency: int = 8,
        rpm: int | None = None,
        cache: Cache | None = None,
        max_retries: int = 5,
        backoff_base_s: float = 1.0,
        timeout_s: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.ep = endpoint
        self.cache = cache
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._sem = asyncio.Semaphore(concurrency)
        self._min_interval = 60.0 / rpm if rpm else 0.0
        self._next_slot = 0.0
        self._lock = asyncio.Lock()
        headers = {}
        if key := endpoint.api_key():
            headers["Authorization"] = f"Bearer {key}"
        self.http = httpx.AsyncClient(
            base_url=endpoint.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=timeout_s,
            transport=transport,
        )

    async def __aenter__(self) -> ChatClient:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.http.aclose()

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.ep.price_in_per_mtok + completion_tokens * self.ep.price_out_per_mtok
        ) / 1e6

    async def _throttle(self) -> None:
        if not self._min_interval:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._min_interval
        if wait > 0:
            await asyncio.sleep(wait)

    def _cache_key(self, body: dict[str, Any]) -> str:
        raw = json.dumps({"base_url": self.ep.base_url, **body}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.ep.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        body.update(self.ep.extra_body)
        key = self._cache_key(body)
        if self.cache and (hit := self.cache.get(key)):
            return ChatResult(**{**hit, "cached": True})

        async with self._sem:
            last_err: Exception | None = None
            for attempt in range(self.max_retries + 1):
                await self._throttle()
                delay: float | None = None
                t0 = time.perf_counter()
                try:
                    r = await self.http.post("chat/completions", json=body)
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    last_err = e
                else:
                    if r.status_code == 200:
                        data = r.json()
                        usage = data.get("usage") or {}
                        pt, ct = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
                        res = ChatResult(
                            text=data["choices"][0]["message"].get("content") or "",
                            prompt_tokens=pt,
                            completion_tokens=ct,
                            latency_s=time.perf_counter() - t0,
                            cost_usd=self.cost(pt, ct),
                        )
                        if self.cache:
                            self.cache.set(key, {k: v for k, v in asdict(res).items() if k != "cached"})
                        return res
                    if r.status_code not in RETRYABLE_STATUS and r.status_code < 500:
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                    last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                    if ra := r.headers.get("retry-after"):
                        try:
                            delay = float(ra)
                        except ValueError:
                            delay = None
                if attempt == self.max_retries:
                    break
                if delay is None:
                    delay = self.backoff_base_s * 2**attempt * (0.5 + random.random())
                await asyncio.sleep(delay)
            raise RuntimeError(f"falhou após {self.max_retries + 1} tentativas: {last_err}")
