from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    n: int


def paired_bootstrap(
    base: Sequence[float], cand: Sequence[float], *, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> BootstrapResult:
    """IC da diferença média (cand - base) reamostrando casos pareados.

    Pareado porque os dois sistemas respondem os mesmos casos: a variância entre casos (uns são
    difíceis pra todo mundo) se cancela e o teste fica bem mais sensível que comparar médias soltas.
    """
    b, c = np.asarray(base, dtype=float), np.asarray(cand, dtype=float)
    if b.shape != c.shape or b.size == 0:
        raise ValueError("base e candidato precisam ter o mesmo número (>0) de casos pareados")
    d = c - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(float(d.mean()), float(lo), float(hi), int(d.size))


def cohen_kappa(a: Sequence[int], b: Sequence[int], *, weights: str | None = None) -> float:
    """Kappa de Cohen (sem peso, linear ou quadrático). Quadrático é o padrão para notas ordinais."""
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if a_arr.shape != b_arr.shape or a_arr.size == 0:
        raise ValueError("listas de rótulos precisam ter o mesmo tamanho (>0)")
    labels = np.union1d(a_arr, b_arr)
    k = labels.size
    pos = {v: i for i, v in enumerate(labels)}
    obs = np.zeros((k, k))
    for x, y in zip(a_arr, b_arr, strict=True):
        obs[pos[x], pos[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / max(1, k - 1)
    elif weights == "quadratic":
        w = ((i - j) / max(1, k - 1)) ** 2
    else:
        raise ValueError(f"weights inválido: {weights}")
    denom = (w * exp).sum()
    return 1.0 if denom == 0 else float(1 - (w * obs).sum() / denom)


def percentiles(values: Sequence[float], qs: Sequence[float] = (50, 95, 99)) -> dict[str, float | None]:
    if not values:
        return {f"p{int(q)}": None for q in qs}
    arr = np.asarray(values, dtype=float)
    return {f"p{int(q)}": float(np.percentile(arr, q)) for q in qs}
