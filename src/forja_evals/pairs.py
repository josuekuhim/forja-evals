"""Pares de preferência para o DPO, a partir das notas do juiz em várias execuções.

Proteção contra circularidade: só aceita datasets marcados com `uso: treino`. O conjunto de
avaliação final nunca vira par de preferência.
"""

from __future__ import annotations

from pathlib import Path

from forja_contracts import EvalCase, OrigemPar, PreferencePair

from forja_evals.datasets import load_dataset
from forja_evals.runner import load_run


def build_pairs(
    dataset_dir: str | Path, run_dirs: list[str | Path], *, min_gap: int = 2
) -> list[PreferencePair]:
    ds = load_dataset(dataset_dir)
    if ds.uso != "treino":
        raise ValueError(
            f"dataset '{ds.nome}' tem uso={ds.uso!r}. Pares de preferência só saem de datasets de treino "
            "(evita o juiz avaliar dados que ele mesmo gerou)."
        )
    by_id: dict[str, EvalCase] = {c.id: c for c in ds.cases}
    candidatos: dict[str, list[tuple[int, str]]] = {}
    for d in run_dirs:
        summary, rows = load_run(d)
        if summary["dataset"]["hash"] != ds.hash:
            raise ValueError(f"execução {d} usou outro dataset")
        for r in rows:
            nota = (r.get("juiz") or {}).get("nota")
            if nota is not None and r.get("output"):
                candidatos.setdefault(r["id"], []).append((nota, r["output"]))

    pairs = []
    for cid, outs in sorted(candidatos.items()):
        outs.sort(key=lambda t: t[0])
        (nota_min, pior), (nota_max, melhor) = outs[0], outs[-1]
        if nota_max - nota_min >= min_gap and pior != melhor:
            pairs.append(
                PreferencePair(
                    prompt=by_id[cid].input, escolhida=melhor, rejeitada=pior, origem=OrigemPar.JUIZ
                )
            )
    return pairs
