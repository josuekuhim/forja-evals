"""Gate de regressão: compara candidato vs. baseline com bootstrap pareado."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from forja_evals.runner import load_run
from forja_evals.stats import BootstrapResult, paired_bootstrap


@dataclass
class Comparacao:
    base_nome: str
    cand_nome: str
    geral: BootstrapResult
    por_categoria: dict[str, BootstrapResult]
    regressao: bool
    alertas: list[str] = field(default_factory=list)
    min_effect: float = 0.02
    alpha: float = 0.05


def _verdict(r: BootstrapResult, min_effect: float) -> str:
    if r.ci_high < 0 and r.mean_diff <= -min_effect:
        return "regressão"
    if r.ci_low > 0 and r.mean_diff >= min_effect:
        return "melhora"
    return "sem diferença significativa"


def compare(
    base_dir: str | Path,
    cand_dir: str | Path,
    *,
    min_effect: float = 0.02,
    alpha: float = 0.05,
    max_erro_rate: float = 0.05,
) -> Comparacao:
    bs, brows = load_run(base_dir)
    cs, crows = load_run(cand_dir)
    if bs["dataset"]["hash"] != cs["dataset"]["hash"]:
        raise ValueError(
            "datasets diferentes (hash não bate): comparação inválida. Mesma régua ou nada."
            f"\n  base: {bs['dataset']['hash'][:12]}  candidato: {cs['dataset']['hash'][:12]}"
        )
    b = {r["id"]: r for r in brows if r.get("score") is not None}
    c = {r["id"]: r for r in crows if r.get("score") is not None}
    ids = sorted(b.keys() & c.keys())
    if not ids:
        raise ValueError("nenhum caso pontuado em comum entre as duas execuções")

    geral = paired_bootstrap([b[i]["score"] for i in ids], [c[i]["score"] for i in ids], alpha=alpha)
    por_cat: dict[str, BootstrapResult] = {}
    for cat in sorted({b[i]["categoria"] for i in ids}):
        cids = [i for i in ids if b[i]["categoria"] == cat]
        por_cat[cat] = paired_bootstrap(
            [b[i]["score"] for i in cids], [c[i]["score"] for i in cids], alpha=alpha
        )

    alertas = []
    erro_rate = cs["geral"]["n_erros"] / max(1, cs["geral"]["n_total"])
    if erro_rate > max_erro_rate:
        alertas.append(f"{erro_rate:.0%} dos casos do candidato falharam por erro de infraestrutura")
    for cat, r in por_cat.items():
        if _verdict(r, min_effect) == "regressão":
            alertas.append(f"regressão significativa na categoria `{cat}` ({r.mean_diff:+.3f})")
    faltando = len(b.keys() ^ c.keys())
    if faltando:
        alertas.append(f"{faltando} casos sem par (pontuados em só uma das execuções) ficaram de fora")

    regressao = _verdict(geral, min_effect) == "regressão" or erro_rate > max_erro_rate
    return Comparacao(bs["nome"], cs["nome"], geral, por_cat, regressao, alertas, min_effect, alpha)


def to_markdown(cmp: Comparacao) -> str:
    conf = int((1 - cmp.alpha) * 100)
    status = "❌ **Regressão detectada**" if cmp.regressao else "✅ **Sem regressão**"
    lines = [
        f"## forja-evals: `{cmp.cand_nome}` vs. `{cmp.base_nome}`",
        "",
        status,
        "",
        f"| Escopo | n | Δ score | IC {conf}% | Veredito |",
        "| --- | --- | --- | --- | --- |",
    ]

    def row(nome: str, r: BootstrapResult) -> str:
        return (
            f"| {nome} | {r.n} | {r.mean_diff:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | "
            f"{_verdict(r, cmp.min_effect)} |"
        )

    lines.append(row("**geral**", cmp.geral))
    lines += [row(cat, r) for cat, r in cmp.por_categoria.items()]
    if cmp.alertas:
        lines += ["", "**Alertas**", *[f"- {a}" for a in cmp.alertas]]
    lines += [
        "",
        f"<sub>Bootstrap pareado, 10k reamostragens. Regressão = IC inteiro abaixo de 0 e "
        f"queda ≥ {cmp.min_effect}. O gate decide pelo escopo geral; categorias com n pequeno têm IC largo.</sub>",
    ]
    return "\n".join(lines)
