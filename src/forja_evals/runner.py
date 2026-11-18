"""Roda um dataset contra um endpoint, aplica métricas e juiz, grava resultados em runs/<run_id>/."""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from forja_contracts import EvalCase, EvalRunRef
from pydantic import BaseModel, ConfigDict, Field

from forja_evals import judge as judge_mod
from forja_evals.cache import Cache
from forja_evals.client import ChatClient, Endpoint
from forja_evals.datasets import Dataset, load_dataset
from forja_evals.metrics import deterministic
from forja_evals.prompts import SYSTEM_PROMPT
from forja_evals.stats import percentiles


class Geracao(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int | None = 0


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nome: str
    dataset: str
    modelo: Endpoint
    geracao: Geracao = Field(default_factory=Geracao)
    juiz: Endpoint | None = None
    system_prompt: str = SYSTEM_PROMPT
    concorrencia: int = 8
    rpm: int | None = None
    cache: str | None = ".cache/forja-evals.sqlite"
    timeout_testes_s: float = 10.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


async def _eval_case(
    case: EvalCase, cfg: RunConfig, model: ChatClient, judge: ChatClient | None
) -> dict[str, Any]:
    row: dict[str, Any] = {"id": case.id, "categoria": case.categoria.value}
    messages = [{"role": "system", "content": cfg.system_prompt}, {"role": "user", "content": case.input}]
    try:
        res = await model.chat(messages, **cfg.geracao.model_dump())
    except Exception as e:  # erro de infraestrutura: não conta como erro do modelo
        return {**row, "erro": str(e), "score": None}

    row.update(
        output=res.text,
        latency_s=res.latency_s,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        cost_usd=res.cost_usd,
        cached=res.cached,
    )
    row["metricas"] = await asyncio.to_thread(deterministic, case, res.text, timeout_s=cfg.timeout_testes_s)

    nota = None
    if judge is not None:
        try:
            jr = await judge.chat(judge_mod.build_messages(case, res.text), temperature=0.0, max_tokens=300)
            nota, just = judge_mod.parse(jr.text)
            row["juiz"] = {"nota": nota, "justificativa": just, "cost_usd": jr.cost_usd}
        except Exception as e:
            row["juiz"] = {"nota": None, "erro": str(e)}

    det = row["metricas"]["score_det"]
    row["score"] = det if det is not None else judge_mod.nota_to_score(nota)
    row["fonte_score"] = "deterministico" if det is not None else ("juiz" if nota is not None else None)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    scored = [r for r in rows if r.get("score") is not None]
    por_cat: dict[str, dict[str, Any]] = {}
    for cat in sorted({r["categoria"] for r in rows}):
        s = [r["score"] for r in scored if r["categoria"] == cat]
        por_cat[cat] = {"score": mean(s), "n": len(s)}
    ok = [r for r in rows if "erro" not in r]
    custo_modelo = sum(r.get("cost_usd", 0.0) for r in ok)
    custo_juiz = sum((r.get("juiz") or {}).get("cost_usd", 0.0) for r in ok)
    return {
        "geral": {
            "score": mean([r["score"] for r in scored]),
            "n": len(scored),
            "n_total": len(rows),
            "n_erros": len(rows) - len(ok),
            "n_sem_score": len(ok) - len(scored),
        },
        "por_categoria": por_cat,
        "latencia_s": percentiles([r["latency_s"] for r in ok if not r.get("cached")]),
        "tokens": {
            "prompt": sum(r.get("prompt_tokens", 0) for r in ok),
            "completion": sum(r.get("completion_tokens", 0) for r in ok),
        },
        "custo_usd": {
            "modelo": custo_modelo,
            "juiz": custo_juiz,
            "modelo_por_1k_req": 1000 * custo_modelo / len(ok) if ok else None,
        },
        "recusas_indevidas": sum(
            1 for r in ok if r["categoria"] != "recusa" and r.get("metricas", {}).get("recusou")
        ),
    }


async def run_eval(
    cfg: RunConfig,
    *,
    out_root: str | Path = "runs",
    transport: httpx.AsyncBaseTransport | None = None,
    limit: int | None = None,
    dataset: Dataset | None = None,
) -> Path:
    ds = dataset or load_dataset(cfg.dataset)
    cases = ds.cases[:limit] if limit else ds.cases
    cache = Cache(cfg.cache) if cfg.cache else None
    t0 = time.perf_counter()
    created = datetime.now(UTC)
    run_id = f"{created:%Y%m%d-%H%M%S}-{_slug(cfg.nome)}"

    kw = dict(concurrency=cfg.concorrencia, rpm=cfg.rpm, cache=cache, transport=transport)
    try:
        async with ChatClient(cfg.modelo, **kw) as model:
            judge = ChatClient(cfg.juiz, **kw) if cfg.juiz else None
            try:
                rows = await asyncio.gather(*(_eval_case(c, cfg, model, judge) for c in cases))
            finally:
                if judge:
                    await judge.__aexit__()
    finally:
        if cache:
            cache.close()

    out = Path(out_root) / run_id
    out.mkdir(parents=True, exist_ok=True)
    ref = EvalRunRef(run_id=run_id, dataset_hash=ds.hash, modelo=cfg.modelo.model, criado_em=created)
    summary = {
        "ref": ref.model_dump(mode="json"),
        "nome": cfg.nome,
        "dataset": {
            "nome": ds.nome,
            "hash": ds.hash,
            "uso": ds.uso,
            "n_casos": len(cases),
            "parcial": bool(limit),
        },
        "duracao_s": time.perf_counter() - t0,
        **summarize(rows),
    }
    (out / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    with (out / "results.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_run(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    p = Path(path)
    summary = json.loads((p / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line) for line in (p / "results.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    return summary, rows
