"""Calibração do juiz: rotulagem humana às cegas e concordância (kappa) juiz vs. humano."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from forja_evals.runner import load_run
from forja_evals.stats import cohen_kappa

LABELS_FILE = "labels.jsonl"


def pending_for_labeling(run_dir: str | Path, n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Amostra estratificada por categoria dos casos com nota do juiz que ainda não têm rótulo humano."""
    _, rows = load_run(run_dir)
    done = {lab["id"] for lab in read_labels(run_dir)}
    pool = [r for r in rows if (r.get("juiz") or {}).get("nota") is not None and r["id"] not in done]
    rng = random.Random(seed)
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in pool:
        by_cat.setdefault(r["categoria"], []).append(r)
    for v in by_cat.values():
        rng.shuffle(v)
    out: list[dict[str, Any]] = []
    while len(out) < n and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(out) < n:
                out.append(by_cat[cat].pop())
    return out


def read_labels(run_dir: str | Path) -> list[dict[str, Any]]:
    p = Path(run_dir) / LABELS_FILE
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]


def append_label(run_dir: str | Path, row: dict[str, Any], humano: int) -> None:
    rec = {"id": row["id"], "categoria": row["categoria"], "humano": humano, "juiz": row["juiz"]["nota"]}
    with (Path(run_dir) / LABELS_FILE).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def label_interactive(
    run_dir: str | Path, cases_by_id: dict[str, str], n: int, ask: Callable[[dict[str, Any], str], int | None]
) -> int:
    """`ask(row, pergunta)` devolve 1..5 ou None para parar. A nota do juiz NÃO é mostrada (rotulagem às cegas)."""
    count = 0
    for row in pending_for_labeling(run_dir, n):
        nota = ask(row, cases_by_id.get(row["id"], ""))
        if nota is None:
            break
        append_label(run_dir, row, nota)
        count += 1
    return count


def agreement(run_dir: str | Path) -> dict[str, Any]:
    labels = read_labels(run_dir)
    if not labels:
        raise ValueError("nenhum rótulo humano ainda; rode `forja-evals label` primeiro")
    h = [lab["humano"] for lab in labels]
    j = [lab["juiz"] for lab in labels]
    res: dict[str, Any] = {
        "n": len(labels),
        "concordancia_exata": sum(a == b for a, b in zip(h, j, strict=True)) / len(h),
        "concordancia_mais_menos_1": sum(abs(a - b) <= 1 for a, b in zip(h, j, strict=True)) / len(h),
        "kappa": cohen_kappa(h, j),
        "kappa_quadratico": cohen_kappa(h, j, weights="quadratic"),
        "vies_medio_juiz": sum(b - a for a, b in zip(h, j, strict=True)) / len(h),
        "por_categoria": {},
    }
    for cat in sorted({lab["categoria"] for lab in labels}):
        sub = [lab for lab in labels if lab["categoria"] == cat]
        hh, jj = [s["humano"] for s in sub], [s["juiz"] for s in sub]
        res["por_categoria"][cat] = {
            "n": len(sub),
            "kappa_quadratico": cohen_kappa(hh, jj, weights="quadratic"),
        }
    return res
