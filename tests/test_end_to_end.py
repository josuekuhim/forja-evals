import asyncio
import json
import shutil

import httpx
import pytest
from typer.testing import CliRunner

from forja_evals import calibration, gate, pairs, report
from forja_evals.cli import app
from forja_evals.runner import load_run, run_eval
from tests.conftest import SEED, FakeOpenAI, make_cfg


def _run(model, tmp_path, fake, **kw):
    return asyncio.run(
        run_eval(
            make_cfg(model, tmp_path, **kw), out_root=tmp_path / "runs", transport=httpx.MockTransport(fake)
        )
    )


def test_modelo_bom_tira_nota_maxima(tmp_path, fake):
    s, rows = load_run(_run("bom", tmp_path, fake))
    assert s["geral"]["n"] == 16 and s["geral"]["n_erros"] == 0
    assert s["geral"]["score"] == pytest.approx(1.0), [r for r in rows if r["score"] != 1.0]
    assert s["recusas_indevidas"] == 0
    assert s["custo_usd"]["modelo"] > 0
    fontes = {r["fonte_score"] for r in rows}
    assert fontes == {"deterministico", "juiz"}


def test_modelo_ruim_e_gate_bloqueia(tmp_path, fake):
    base = _run("bom", tmp_path, fake)
    cand = _run("ruim", tmp_path, fake)
    s, _ = load_run(cand)
    assert s["geral"]["score"] < 0.2
    cmp = gate.compare(base, cand)
    assert cmp.regressao
    assert "Regressão detectada" in gate.to_markdown(cmp)
    assert not gate.compare(base, base).regressao


def test_cache_evita_chamadas_repetidas(tmp_path, fake):
    _run("bom", tmp_path, fake)
    n = len(fake.calls)
    s, rows = load_run(_run("bom", tmp_path, fake))
    assert len(fake.calls) == n
    assert all(r["cached"] for r in rows)
    assert s["latencia_s"]["p50"] is None  # cache não entra na latência


def test_retry_em_429(tmp_path, seed_cases):
    fake = FakeOpenAI(seed_cases, fail_first=3)
    s, _ = load_run(_run("bom", tmp_path, fake, judge=False))
    assert s["geral"]["n_erros"] == 0


def test_erro_de_infra_nao_vira_nota_zero(tmp_path):
    def down(request):
        return httpx.Response(400, json={"error": "modelo não existe"})

    cfg = make_cfg("bom", tmp_path, judge=False)
    out = asyncio.run(run_eval(cfg, out_root=tmp_path / "runs", transport=httpx.MockTransport(down)))
    s, _ = load_run(out)
    assert s["geral"]["n_erros"] == 16
    assert s["geral"]["score"] is None


def test_gate_recusa_datasets_diferentes(tmp_path, fake):
    base = _run("bom", tmp_path, fake)
    s = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    other = tmp_path / "outro"
    shutil.copytree(base, other)
    s["dataset"]["hash"] = "0" * 64
    (other / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    with pytest.raises(ValueError, match="Mesma régua"):
        gate.compare(base, other)


def test_pares_recusam_dataset_de_avaliacao(tmp_path, fake):
    run = _run("bom", tmp_path, fake)
    with pytest.raises(ValueError, match="uso='avaliacao'"):
        pairs.build_pairs(SEED, [run])


def test_pares_em_dataset_de_treino(tmp_path, fake):
    ds = tmp_path / "treino"
    shutil.copytree(SEED, ds)
    (ds / "dataset.yaml").write_text("nome: treino-teste\nuso: treino\n", encoding="utf-8")
    runs = [
        _run("bom", tmp_path, fake, dataset=str(ds)),
        _run("ruim", tmp_path, fake, dataset=str(ds)),
    ]
    ps = pairs.build_pairs(ds, runs)
    assert len(ps) == 16
    assert all("BOM" in p.escolhida for p in ps if "```" not in p.escolhida)


def test_calibracao(tmp_path, fake):
    run = _run("bom", tmp_path, fake)
    # humano concorda com o juiz em tudo, exceto discordar por 1 ponto num caso
    vistos = []

    def ask(row, pergunta):
        vistos.append(row)
        assert "nota" not in pergunta
        return row["juiz"]["nota"] - (1 if len(vistos) == 1 else 0)

    n = calibration.label_interactive(run, {}, 5, ask)
    assert n == 5
    assert len({r["categoria"] for r in vistos}) == 5  # amostra estratificada
    assert calibration.label_interactive(run, {}, 5, ask) == 5  # retoma sem repetir
    ids = [lab["id"] for lab in calibration.read_labels(run)]
    assert len(ids) == len(set(ids)) == 10
    ag = calibration.agreement(run)
    assert ag["n"] == 10
    assert ag["concordancia_mais_menos_1"] == 1.0


def test_cli_compare_e_report(tmp_path, fake):
    base, cand = _run("bom", tmp_path, fake), _run("ruim", tmp_path, fake)
    runner = CliRunner()
    r = runner.invoke(app, ["compare", str(base), str(cand)])
    assert r.exit_code == 1 and "Regressão" in r.output
    r = runner.invoke(app, ["compare", str(base), str(base)])
    assert r.exit_code == 0
    table = report.runs_table([base, cand])
    assert "| bom |" in table and "| ruim |" in table
