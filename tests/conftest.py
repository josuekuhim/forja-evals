"""Servidor OpenAI-compatível falso (httpx.MockTransport) com um modelo "bom", um "ruim" e um juiz."""

from __future__ import annotations

import json
import re

import httpx
import pytest
from forja_contracts import EvalCase

from forja_evals.client import Endpoint
from forja_evals.datasets import load_dataset
from forja_evals.runner import RunConfig

SEED = "datasets/forja-seed-v0"

BOAS_CORRECOES = {
    "media": "def media(xs):\n    return sum(xs) / len(xs) if xs else 0.0",
    "eh_palindromo": "def eh_palindromo(s):\n    t = s.replace(' ', '').lower()\n    return t == t[::-1]",
    "fatorial": "def fatorial(n):\n    if n <= 1:\n        return 1\n    return n * fatorial(n - 1)",
    "conta_palavras": "def conta_palavras(texto):\n    return len(texto.split())",
}


def resposta_boa(case: EvalCase) -> str:
    if case.categoria.value == "recusa":
        return "Não sei: não tenho contexto suficiente para responder isso. BOM"
    if case.categoria.value == "correcao":
        for nome, code in BOAS_CORRECOES.items():
            if f"def {nome}(" in case.input:
                return f"Corrigido:\n\n```python\n{code}\n```\nBOM"
    if case.categoria.value == "deteccao_bug":
        return f"O bug: {case.referencia} BOM"
    return f"{case.referencia} BOM"


def resposta_ruim(case: EvalCase) -> str:
    if case.categoria.value == "correcao":
        return "```python\ndef x(:\n```"
    return "Com certeza! É só usar o parâmetro mágico. RUIM"


class FakeOpenAI:
    def __init__(self, cases: list[EvalCase], fail_first: int = 0) -> None:
        self.by_input = {c.input: c for c in cases}
        self.calls: list[dict] = []
        self.fail_first = fail_first

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/v1/chat/completions")
        body = json.loads(request.content)
        self.calls.append(body)
        if self.fail_first > 0:
            self.fail_first -= 1
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "rate limited"})
        model = body["model"]
        user = body["messages"][-1]["content"]
        if model == "juiz":
            resposta = user.split("## Resposta do assistente\n", 1)[1]
            nota = 5 if "BOM" in resposta else 1
            text = json.dumps({"justificativa": "ok", "nota": nota})
        else:
            case = self.by_input[user]
            text = resposta_boa(case) if model in ("bom", "loop") else resposta_ruim(case)
        finish = "stop"
        if model == "loop" and case.categoria.value in ("explicacao", "api"):
            # Mesma nota do "bom" (o juiz vê BOM), mas a resposta degenera em loop e é cortada.
            text += "\n" + "Isso é importante porque a função retorna o valor. " * 40
            finish = "length"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}],
                "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": len(text) // 4},
            },
        )


def make_cfg(model: str, tmp_path, judge: bool = True, dataset: str = SEED) -> RunConfig:
    ep = lambda m: Endpoint(  # noqa: E731
        base_url="http://fake/v1", model=m, price_in_per_mtok=1.0, price_out_per_mtok=2.0
    )
    return RunConfig(
        nome=model,
        dataset=dataset,
        modelo=ep(model),
        juiz=ep("juiz") if judge else None,
        cache=str(tmp_path / "cache.sqlite"),
        concorrencia=4,
    )


@pytest.fixture
def seed_cases() -> list[EvalCase]:
    return load_dataset(SEED).cases


@pytest.fixture
def fake(seed_cases) -> FakeOpenAI:
    return FakeOpenAI(seed_cases)


def strip_code(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)
