"""Datasets versionados e imutáveis: a versão é o hash do conteúdo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from forja_contracts import EvalCase, dataset_hash, read_jsonl

Uso = Literal["avaliacao", "treino"]


@dataclass(frozen=True)
class Dataset:
    nome: str
    uso: Uso
    cases: list[EvalCase]
    hash: str
    path: Path

    @property
    def short_hash(self) -> str:
        return self.hash[:12]


def load_dataset(path: str | Path) -> Dataset:
    """Lê `<dir>/cases.jsonl` e `<dir>/dataset.yaml` (nome, uso, descrição, licença)."""
    p = Path(path)
    meta = yaml.safe_load((p / "dataset.yaml").read_text(encoding="utf-8")) or {}
    cases = read_jsonl(p / "cases.jsonl", EvalCase)
    ids = [c.id for c in cases]
    dups = {i for i in ids if ids.count(i) > 1}
    if dups:
        raise ValueError(f"IDs duplicados em {p}: {sorted(dups)}")
    uso = meta.get("uso", "avaliacao")
    if uso not in ("avaliacao", "treino"):
        raise ValueError(f"uso inválido em {p}/dataset.yaml: {uso!r}")
    return Dataset(nome=meta.get("nome", p.name), uso=uso, cases=cases, hash=dataset_hash(cases), path=p)
