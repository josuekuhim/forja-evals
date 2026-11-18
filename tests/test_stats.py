import numpy as np
import pytest

from forja_evals.stats import cohen_kappa, paired_bootstrap, percentiles


def test_bootstrap_detecta_queda_clara():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.6, 1.0, 200)
    r = paired_bootstrap(base, base - 0.1)
    assert r.ci_high < 0
    assert r.mean_diff == pytest.approx(-0.1)


def test_bootstrap_nao_acusa_ruido():
    rng = np.random.default_rng(2)
    base = rng.integers(0, 2, 200).astype(float)
    cand = base.copy()
    flip = rng.choice(200, 10, replace=False)  # 5 sobem, 5 descem
    cand[flip[:5]] = 1 - cand[flip[:5]]
    cand[flip[5:]] = 1 - cand[flip[5:]]
    r = paired_bootstrap(base, cand)
    assert r.ci_low <= 0 <= r.ci_high


def test_kappa_valores_conhecidos():
    assert cohen_kappa([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    # Exemplo clássico 2x2: concordância 0.7, esperada 0.5 -> kappa 0.4
    a = [1] * 25 + [1] * 25 + [0] * 0 + [0] * 50
    b = [1] * 25 + [0] * 25 + [0] * 0 + [0] * 50
    po = 0.75
    pe = 0.5 * 0.25 + 0.5 * 0.75
    assert cohen_kappa(a, b) == pytest.approx((po - pe) / (1 - pe))


def test_kappa_quadratico_pune_menos_erros_vizinhos():
    h = [1, 2, 3, 4, 5] * 10
    perto = [min(5, x + 1) for x in h]
    longe = [6 - x for x in h]
    assert cohen_kappa(h, perto, weights="quadratic") > cohen_kappa(h, perto)
    assert cohen_kappa(h, longe, weights="quadratic") < 0


def test_percentiles():
    p = percentiles(list(range(1, 101)))
    assert p["p50"] == pytest.approx(50.5)
    assert percentiles([])["p95"] is None
