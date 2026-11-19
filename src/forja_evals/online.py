"""Evals online: avalia uma amostra do tráfego real do forja-review a partir dos traces OpenTelemetry.

Offline, o dataset é fixo e tem referência. Online não há referência: o juiz avalia cada comentário
publicado ("útil" ou não) e a review do arquivo como um todo (1 a 5). Dois sinais saem daí:

- **qualidade em produção:** nota média e taxa de comentários úteis segundo o juiz, por modelo;
- **calibração do juiz com humanos de verdade:** onde há 👍/👎 no comentário, compara o veredito do
  juiz com o dos usuários (concordância e kappa). É a mesma ideia do `kappa` offline, só que com
  rótulos que custaram zero, porque vêm do uso.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from forja_contracts.telemetry import Feedback, LLMInteraction, load_interactions
from pydantic import BaseModel, ConfigDict, Field

from forja_evals.cache import Cache
from forja_evals.client import ChatClient, Endpoint
from forja_evals.runner import Geracao
from forja_evals.stats import cohen_kappa, percentiles

JUDGE_REVIEW_SYSTEM = """Você avalia um revisor automático de pull requests. Recebe o diff de UM arquivo (com o número da linha nova à esquerda) e os comentários que o revisor publicou.

Para cada comentário, decida se é ÚTIL: aponta um problema real naquela linha (ou muito perto), o raciocínio está correto e a correção sugerida faz sentido. Não é útil se descreve o código em vez de apontar problema, se o problema não existe, se está errado ou se é só estilo.

Depois dê uma nota de 1 a 5 para a review do arquivo:
5 = achou os problemas importantes e todos os comentários são úteis.
4 = útil, com uma omissão ou um comentário fraco.
3 = metade útil, metade ruído; ou deixou passar um problema importante.
2 = majoritariamente ruído ou errado.
1 = errado ou enganoso; ou nenhum comentário quando havia bug claro.
Se não há problema real no diff e o revisor não comentou nada, a nota é 5.

Responda SOMENTE com JSON:
{"comentarios": [{"linha": <int>, "util": <true|false>, "motivo": "<curto>"}], "problemas_nao_apontados": ["<curto>"], "justificativa": "<uma ou duas frases>", "nota": <1 a 5>}"""


class OnlineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nome: str
    traces: str  # arquivo .jsonl ou diretório com spans do forja-review
    juiz: Endpoint
    juiz_geracao: Geracao = Field(default_factory=lambda: Geracao(max_tokens=2048))
    amostra: int = 50
    seed: int = 0
    concorrencia: int = 1
    timeout_s: float = 900.0
    cache: str | None = ".cache/forja-evals.sqlite"

    @classmethod
    def from_yaml(cls, path: str | Path) -> OnlineConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def sample(calls: list[LLMInteraction], n: int, seed: int) -> list[LLMInteraction]:
    """Amostra estável: mesma seed e mesmos traces → mesmos casos, em qualquer ordem de leitura."""
    pool = sorted((c for c in calls if c.mensagens and not c.erro), key=lambda c: c.span_id)
    return pool if len(pool) <= n else random.Random(seed).sample(pool, n)


def build_messages(call: LLMInteraction) -> list[dict[str, str]]:
    diff = next((m["content"] for m in call.mensagens if m.get("role") == "user"), "")
    coms = "\n".join(f"- linha {c.linha} ({c.severidade}): {c.comentario}" for c in call.comentarios)
    user = f"## Diff\n{diff}\n\n## Comentários publicados\n{coms or '(nenhum)'}"
    return [{"role": "system", "content": JUDGE_REVIEW_SYSTEM}, {"role": "user", "content": user}]


_JSON = re.compile(r"\{.*\}", re.S)


def parse_judgment(text: str) -> dict[str, Any] | None:
    m = _JSON.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    nota = obj.get("nota")
    if not isinstance(nota, int | float) or not 1 <= int(nota) <= 5:
        return None
    vereditos = {}
    for c in obj.get("comentarios") or []:
        if isinstance(c, dict) and isinstance(c.get("util"), bool):
            try:
                vereditos[int(c["linha"])] = c["util"]
            except (KeyError, TypeError, ValueError):
                continue
    return {
        "nota": int(nota),
        "justificativa": str(obj.get("justificativa", "")),
        "util_por_linha": vereditos,
        "nao_apontados": [str(x) for x in obj.get("problemas_nao_apontados") or []],
    }


def agreement(judged: list[dict[str, Any]], feedback: list[Feedback]) -> dict[str, Any]:
    """Juiz × humanos, por comentário. Humano: útil se 👍 > 👎; empate fica de fora."""
    fb = {(f.review_trace_id, f.arquivo, f.linha): f for f in feedback if f.up != f.down}
    pares = []
    for j in judged:
        for linha, util in j["util_por_linha"].items():
            f = fb.get((j["trace_id"], j["arquivo"], linha))
            if f:
                pares.append((int(util), int(f.up > f.down)))
    if not pares:
        return {"n": 0}
    juiz, humano = zip(*pares, strict=True)
    out: dict[str, Any] = {
        "n": len(pares),
        "concordancia": sum(a == b for a, b in pares) / len(pares),
        "taxa_util_juiz": sum(juiz) / len(juiz),
        "taxa_util_humano": sum(humano) / len(humano),
    }
    if len(set(juiz)) > 1 or len(set(humano)) > 1:
        out["kappa"] = cohen_kappa(list(juiz), list(humano))
    return out


async def run_online(
    cfg: OnlineConfig, *, out_root: str | Path = "runs", transport: httpx.AsyncBaseTransport | None = None
) -> Path:
    calls, feedback = load_interactions(cfg.traces)
    amostra = sample(calls, cfg.amostra, cfg.seed)
    cache = Cache(cfg.cache) if cfg.cache else None
    created = datetime.now(UTC)

    async def judge_one(client: ChatClient, c: LLMInteraction) -> dict[str, Any]:
        row: dict[str, Any] = {
            "trace_id": c.trace_id,
            "span_id": c.span_id,
            "repo": c.repo,
            "pr": c.pr,
            "arquivo": c.arquivo,
            "modelo": c.modelo,
            "n_comentarios": len(c.comentarios),
            "problemas": c.problemas,
        }
        try:
            r = await client.chat(build_messages(c), **cfg.juiz_geracao.model_dump())
        except Exception as e:
            return {**row, "erro": str(e)}
        j = parse_judgment(r.text)
        if j is None:
            return {**row, "erro": "juiz não devolveu JSON válido", "saida_juiz": r.text[-500:]}
        # Comentário que o juiz não citou conta como não julgado (não como inútil).
        return {**row, **j, "util_por_linha": j["util_por_linha"]}

    try:
        async with ChatClient(
            cfg.juiz, concurrency=cfg.concorrencia, cache=cache, transport=transport, timeout_s=cfg.timeout_s
        ) as client:
            rows = await asyncio.gather(*(judge_one(client, c) for c in amostra))
    finally:
        if cache:
            cache.close()

    ok = [r for r in rows if "erro" not in r]
    julgados = [u for r in ok for u in r["util_por_linha"].values()]
    por_modelo: dict[str, dict[str, Any]] = {}
    for m in sorted({r["modelo"] for r in ok}):
        rs = [r for r in ok if r["modelo"] == m]
        us = [u for r in rs for u in r["util_por_linha"].values()]
        por_modelo[m] = {
            "n": len(rs),
            "nota_media": sum(r["nota"] for r in rs) / len(rs),
            "taxa_comentarios_uteis": (sum(us) / len(us)) if us else None,
        }
    todos_problemas = [p for c in calls for p in c.problemas]
    summary = {
        "nome": cfg.nome,
        "criado_em": created.isoformat(),
        "traces": {
            "chamadas": len(calls),
            "com_conteudo": sum(bool(c.mensagens) for c in calls),
            "com_erro": sum(bool(c.erro) for c in calls),
            "reviews": len({c.trace_id for c in calls}),
            "comentarios_publicados": sum(len(c.comentarios) for c in calls),
            "feedback_humano": len(feedback),
            "problemas_de_formato": {p: todos_problemas.count(p) for p in sorted(set(todos_problemas))},
            "duracao_s": percentiles([c.duracao_s for c in calls if c.duracao_s is not None]),
        },
        "amostra": {"n": len(amostra), "seed": cfg.seed, "julgados": len(ok), "erros": len(rows) - len(ok)},
        "juiz": {
            "modelo": cfg.juiz.model,
            "nota_media": (sum(r["nota"] for r in ok) / len(ok)) if ok else None,
            "taxa_comentarios_uteis": (sum(julgados) / len(julgados)) if julgados else None,
            "por_modelo": por_modelo,
        },
        "juiz_vs_humanos": agreement(ok, feedback),
    }
    out = (
        Path(out_root) / f"{created:%Y%m%d-%H%M%S}-online-{re.sub(r'[^a-z0-9]+', '-', cfg.nome.lower())[:40]}"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    with (out / "julgamentos.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {**r, "util_por_linha": {str(k): v for k, v in r.get("util_por_linha", {}).items()}},
                    ensure_ascii=False,
                )
                + "\n"
            )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "online.md").write_text(to_markdown(summary), encoding="utf-8")
    return out


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def to_markdown(s: dict[str, Any]) -> str:
    t, a, j, h = s["traces"], s["amostra"], s["juiz"], s["juiz_vs_humanos"]
    lines = [
        f"## forja-evals online: `{s['nome']}`",
        "",
        f"{t['reviews']} reviews, {t['chamadas']} chamadas ao modelo, {t['comentarios_publicados']} comentários "
        f"publicados, {t['feedback_humano']} com feedback humano. Amostra: {a['julgados']}/{a['n']} julgadas "
        f"(seed {a['seed']}) pelo juiz `{j['modelo']}`.",
        "",
        "| Modelo | n | Nota média (1-5) | Comentários úteis (juiz) |",
        "| --- | --- | --- | --- |",
    ]
    for m, r in j["por_modelo"].items():
        lines.append(f"| `{m}` | {r['n']} | {r['nota_media']:.2f} | {_pct(r['taxa_comentarios_uteis'])} |")
    if h.get("n"):
        k = f", kappa {h['kappa']:.2f}" if "kappa" in h else ""
        lines += [
            "",
            f"**Juiz × humanos** ({h['n']} comentários com 👍/👎): concordância {_pct(h['concordancia'])}{k}. "
            f"Úteis segundo o juiz: {_pct(h['taxa_util_juiz'])}; segundo os humanos: {_pct(h['taxa_util_humano'])}.",
        ]
    else:
        lines += ["", "Sem feedback humano na amostra: a nota do juiz ainda não está calibrada em produção."]
    if t["problemas_de_formato"]:
        probs = ", ".join(f"{k} ({v})" for k, v in t["problemas_de_formato"].items())
        lines += ["", f"Problemas na saída do modelo (todas as chamadas): {probs}."]
    return "\n".join(lines)
