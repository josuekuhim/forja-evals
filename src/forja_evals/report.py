"""Tabela-resumo de várias execuções no mesmo dataset (a tabela do hub do Forja)."""

from __future__ import annotations

from pathlib import Path

from forja_evals.runner import load_run


def _fmt(x: float | None, spec: str) -> str:
    return "—" if x is None else format(x, spec)


def runs_table(run_dirs: list[str | Path]) -> str:
    runs = [load_run(d)[0] for d in run_dirs]
    hashes = {r["dataset"]["hash"] for r in runs}
    cats = sorted({c for r in runs for c in r["por_categoria"]})
    head = ["Configuração", "Score", *cats, "Latência p50", "Custo / 1k req", "Run"]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for r in runs:
        cells = [
            r["nome"],
            _fmt(r["geral"]["score"], ".3f"),
            *[_fmt(r["por_categoria"].get(c, {}).get("score"), ".2f") for c in cats],
            _fmt(r["latencia_s"]["p50"], ".2f") + ("s" if r["latencia_s"]["p50"] is not None else ""),
            "$" + _fmt(r["custo_usd"]["modelo_por_1k_req"], ".2f"),
            f"`{r['ref']['run_id']}`",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if len(hashes) > 1:
        lines.append("\n> ⚠️ As execuções usam datasets diferentes; os números não são comparáveis.")
    else:
        lines.append(f"\n<sub>Dataset `{runs[0]['dataset']['nome']}` @ `{next(iter(hashes))[:12]}`</sub>")
    return "\n".join(lines)
