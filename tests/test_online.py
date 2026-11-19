import asyncio
import json

import httpx
from forja_contracts import telemetry as T

from forja_evals.client import Endpoint
from forja_evals.online import OnlineConfig, agreement, parse_judgment, run_online, sample


def _llm_span(i: int, comentarios: list[dict], modelo="qwen", com_conteudo=True) -> dict:
    attrs = {
        T.ATTR_REPO: "eu/repo",
        T.ATTR_PR: 1,
        T.ATTR_FILE: f"f{i}.py",
        T.ATTR_MODEL: modelo,
        T.ATTR_COMMENTS: json.dumps(comentarios),
        T.ATTR_PROBLEMS: json.dumps(["linha_fora_do_diff"] if i == 0 else []),
    }
    if com_conteudo:
        attrs[T.ATTR_INPUT_MESSAGES] = json.dumps(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": f"Arquivo: f{i}.py\n    3 + x = 1/0"},
            ]
        )
    return {
        "name": T.SPAN_LLM,
        "context": {"trace_id": f"0x{i:032x}", "span_id": f"0x{i:016x}"},
        "attributes": attrs,
        "start_time": "2026-11-19T10:00:00Z",
        "end_time": "2026-11-19T10:00:04Z",
    }


def _fb(i: int, linha: int, up: int, down: int) -> dict:
    return {
        "name": T.SPAN_FEEDBACK,
        "context": {"trace_id": "0xff", "span_id": "0x01"},
        "attributes": {
            T.ATTR_REVIEW_TRACE: f"{i:032x}",
            T.ATTR_COMMENT_ID: 100 * i + linha,
            T.ATTR_FILE: f"f{i}.py",
            T.ATTR_LINE: linha,
            T.ATTR_FEEDBACK_UP: up,
            T.ATTR_FEEDBACK_DOWN: down,
        },
    }


def _com(linha: int, texto: str) -> dict:
    return {"arquivo": "x", "linha": linha, "severidade": "bug", "comentario": texto}


def test_parse_judgment():
    j = parse_judgment(
        'pensando... {"comentarios": [{"linha": 3, "util": true}, {"linha": "x", "util": false}], "nota": 4}'
    )
    assert j["nota"] == 4 and j["util_por_linha"] == {3: True}
    assert parse_judgment('{"nota": 9}') is None
    assert parse_judgment("sem json") is None


def test_amostra_estavel_e_ignora_sem_conteudo():
    from forja_contracts.telemetry import LLMInteraction

    calls = [
        LLMInteraction(
            trace_id="t",
            span_id=f"{i:02d}",
            repo="r",
            arquivo="a",
            modelo="m",
            mensagens=[{"role": "user", "content": "d"}] if i % 2 else [],
        )
        for i in range(20)
    ]
    a = sample(calls, 5, seed=1)
    assert len(a) == 5 and all(c.mensagens for c in a)
    assert [c.span_id for c in a] == [c.span_id for c in sample(list(reversed(calls)), 5, seed=1)]


def test_agreement():
    from forja_contracts.telemetry import Feedback

    judged = [{"trace_id": "t", "arquivo": "a.py", "util_por_linha": {1: True, 2: False, 3: True}}]
    fb = [
        Feedback(review_trace_id="t", comment_id=1, arquivo="a.py", linha=1, up=2, down=0),
        Feedback(review_trace_id="t", comment_id=2, arquivo="a.py", linha=2, up=0, down=1),
        Feedback(review_trace_id="t", comment_id=3, arquivo="a.py", linha=3, up=0, down=1),
        Feedback(review_trace_id="t", comment_id=4, arquivo="a.py", linha=9, up=1, down=1),  # empate: fora
    ]
    a = agreement(judged, fb)
    assert a["n"] == 3 and abs(a["concordancia"] - 2 / 3) < 1e-9
    assert agreement(judged, []) == {"n": 0}


def test_run_online(tmp_path):
    spans = [
        _llm_span(0, [_com(3, "Divisão por zero."), _com(4, "Adicionou x.")]),
        _llm_span(1, [_com(3, "Divisão por zero.")]),
        _llm_span(2, [], com_conteudo=False),  # repo privado: fora da amostra
        _fb(0, 3, up=1, down=0),
        _fb(0, 4, up=0, down=2),
        _fb(1, 3, up=0, down=1),  # humano discorda do juiz aqui
    ]
    (tmp_path / "spans.jsonl").write_text("\n".join(json.dumps(s) for s in spans), encoding="utf-8")

    def juiz(req: httpx.Request) -> httpx.Response:
        user = json.loads(req.content)["messages"][-1]["content"]
        vered = [{"linha": 3, "util": True}] + ([{"linha": 4, "util": False}] if "linha 4" in user else [])
        text = json.dumps({"comentarios": vered, "justificativa": "ok", "nota": 4})
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}], "usage": {}})

    cfg = OnlineConfig(
        nome="teste", traces=str(tmp_path), juiz=Endpoint(base_url="http://fake/v1", model="juiz"), cache=None
    )
    out = asyncio.run(run_online(cfg, out_root=tmp_path / "runs", transport=httpx.MockTransport(juiz)))
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["traces"]["chamadas"] == 3 and s["traces"]["com_conteudo"] == 2
    assert s["traces"]["problemas_de_formato"] == {"linha_fora_do_diff": 1}
    assert s["amostra"]["julgados"] == 2
    assert s["juiz"]["nota_media"] == 4 and abs(s["juiz"]["taxa_comentarios_uteis"] - 2 / 3) < 1e-9
    h = s["juiz_vs_humanos"]
    assert h["n"] == 3 and abs(h["concordancia"] - 2 / 3) < 1e-9
    md = (out / "online.md").read_text(encoding="utf-8")
    assert "Juiz × humanos" in md and "`qwen`" in md
