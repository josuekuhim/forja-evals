"""Gate de regressão: compara candidato vs. baseline com bootstrap pareado."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from forja_evals.metrics import degenerate
from forja_evals.runner import health, load_run
from forja_evals.stats import BootstrapResult, paired_bootstrap


@dataclass
class Comparacao:
    base_nome: str
    cand_nome: str
    geral: BootstrapResult
    por_categoria: dict[str, BootstrapResult]
    regressao: bool
    alertas: list[str] = field(default_factory=list)
    # Saúde: Δ na fração de respostas SEM loop (pareado, todos os casos), e as taxas de cada lado.
    saude: BootstrapResult | None = None
    saude_base: dict | None = None
    saude_cand: dict | None = None
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

    # A nota só existe onde há métrica; loops aparecem em qualquer caso. Um modelo que já erra tudo
    # (efeito chão) não tem nota para perder, mas ainda pode degenerar: o gate olha os dois.
    bo = {r["id"]: r for r in brows if "output" in r}
    co = {r["id"]: r for r in crows if "output" in r}
    oids = sorted(bo.keys() & co.keys())
    saude = paired_bootstrap(
        [0.0 if degenerate(bo[i]["output"]) else 1.0 for i in oids],
        [0.0 if degenerate(co[i]["output"]) else 1.0 for i in oids],
        alpha=alpha,
    )
    hb = health(brows, _max_tokens(base_dir))
    hc = health(crows, _max_tokens(cand_dir))
    degen = _verdict(saude, min_effect) == "regressão"
    if degen:
        alertas.insert(
            0,
            f"respostas em loop: {hb['taxa_degeneradas']:.0%} → {hc['taxa_degeneradas']:.0%} "
            f"(repetição de 8-gramas ≥ limiar), mesmo onde a nota não mede",
        )
    if hc["taxa_truncadas"] > hb["taxa_truncadas"]:
        alertas.append(
            f"respostas cortadas por max_tokens: {hb['taxa_truncadas']:.0%} → {hc['taxa_truncadas']:.0%}"
        )

    regressao = _verdict(geral, min_effect) == "regressão" or erro_rate > max_erro_rate or degen
    return Comparacao(
        bs["nome"], cs["nome"], geral, por_cat, regressao, alertas, saude, hb, hc, min_effect, alpha
    )


def _max_tokens(run_dir: str | Path) -> int | None:
    p = Path(run_dir) / "config.yaml"
    if not p.exists():
        return None
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return (cfg.get("geracao") or {}).get("max_tokens")


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
    if cmp.saude is not None:
        lines.append(row("sem loop (todos os casos)", cmp.saude))
    if cmp.saude_base and cmp.saude_cand:
        b, c = cmp.saude_base, cmp.saude_cand
        lines += [
            "",
            "| Saúde (todos os casos) | baseline | candidato |",
            "| --- | --- | --- |",
            f"| em loop | {b['taxa_degeneradas']:.0%} | {c['taxa_degeneradas']:.0%} |",
            f"| cortadas por max_tokens | {b['taxa_truncadas']:.0%} | {c['taxa_truncadas']:.0%} |",
        ]
    if cmp.alertas:
        lines += ["", "**Alertas**", *[f"- {a}" for a in cmp.alertas]]
    lines += [
        "",
        f"<sub>Bootstrap pareado, 10k reamostragens. Regressão = IC inteiro abaixo de 0 e "
        f"queda ≥ {cmp.min_effect}. O gate decide pelo escopo geral e pela linha 'sem loop'; "
        f"categorias com n pequeno têm IC largo.</sub>",
    ]
    return "\n".join(lines)
