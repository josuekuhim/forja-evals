from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
import yaml
from forja_contracts import write_jsonl

from forja_evals import calibration, gate, pairs, report
from forja_evals.datasets import load_dataset
from forja_evals.runner import RunConfig, rescore, run_eval

app = typer.Typer(help="forja-evals: a régua do Forja.", no_args_is_help=True)

# No Windows, stdout redirecionado (pipe, CI) vem em cp1252 e o markdown do gate (✅/❌) quebra.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")


@app.command()
def run(
    config: Annotated[Path, typer.Argument(help="YAML da execução")],
    out: Annotated[Path, typer.Option(help="Diretório das execuções")] = Path("runs"),
    limit: Annotated[int | None, typer.Option(help="Só os N primeiros casos (smoke)")] = None,
) -> None:
    """Roda um dataset contra um endpoint OpenAI-compatível."""
    cfg = RunConfig.from_yaml(config)
    run_dir = asyncio.run(run_eval(cfg, out_root=out, limit=limit))
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    typer.echo(report.runs_table([run_dir]))
    typer.echo(f"\nResultados em {run_dir}")
    if s["geral"]["n_erros"]:
        typer.secho(f"{s['geral']['n_erros']} casos com erro de infraestrutura", fg="yellow")
    _github_output("run_dir", str(run_dir))


@app.command()
def compare(
    baseline: Path,
    candidato: Path,
    min_effect: Annotated[float, typer.Option(help="Queda mínima que importa")] = 0.02,
    alpha: float = 0.05,
    markdown_out: Annotated[Path | None, typer.Option(help="Salva o relatório em Markdown")] = None,
    fail_on_regression: Annotated[bool, typer.Option(help="Sai com código 1 se houver regressão")] = True,
) -> None:
    """Gate de regressão: bootstrap pareado candidato vs. baseline."""
    cmp = gate.compare(baseline, candidato, min_effect=min_effect, alpha=alpha)
    md = gate.to_markdown(cmp)
    typer.echo(md)
    if markdown_out:
        markdown_out.write_text(md, encoding="utf-8")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    _github_output("regressao", str(cmp.regressao).lower())
    if cmp.regressao and fail_on_regression:
        raise typer.Exit(code=1)


@app.command("rescore")
def rescore_cmd(runs: list[Path]) -> None:
    """Recalcula métricas e resumo de runs salvos quando a régua muda (não chama modelo nem juiz)."""
    for r in runs:
        s = rescore(r)
        g, h = s["geral"], s["saude"]
        typer.echo(
            f"{r}: score {g['score']:.3f} (n={g['n']})  loops {h['taxa_degeneradas']:.0%}  "
            f"cortadas {h['taxa_truncadas']:.0%}"
        )


@app.command("report")
def report_cmd(runs: list[Path]) -> None:
    """Tabela-resumo de várias execuções."""
    typer.echo(report.runs_table(runs))


@app.command()
def label(
    run_dir: Path,
    n: Annotated[int, typer.Option(help="Quantos casos rotular nesta sessão")] = 30,
) -> None:
    """Rotulagem humana às cegas (a nota do juiz fica escondida). Retomável."""
    cfg = RunConfig.model_validate(yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")))
    inputs = {c.id: c.input for c in load_dataset(cfg.dataset).cases}

    def ask(row: dict, pergunta: str) -> int | None:
        typer.secho(f"\n{'=' * 80}\n[{row['id']}] ({row['categoria']})", fg="cyan", bold=True)
        typer.echo(f"\n--- Pergunta ---\n{pergunta}\n\n--- Resposta do modelo ---\n{row['output']}\n")
        while True:
            v = typer.prompt("Nota 1-5 (q para sair)")
            if v.strip().lower() == "q":
                return None
            if v.strip() in {"1", "2", "3", "4", "5"}:
                return int(v)

    done = calibration.label_interactive(run_dir, inputs, n, ask)
    typer.echo(f"{done} rótulos salvos em {run_dir / calibration.LABELS_FILE}")


@app.command()
def kappa(run_dir: Path) -> None:
    """Concordância juiz vs. humano."""
    typer.echo(json.dumps(calibration.agreement(run_dir), indent=2, ensure_ascii=False))


@app.command("pairs")
def pairs_cmd(
    dataset: Path,
    runs: list[Path],
    out: Annotated[Path, typer.Option(help="JSONL de saída")] = Path("pairs.jsonl"),
    min_gap: int = 2,
) -> None:
    """Gera pares de preferência (DPO) a partir de execuções num dataset de TREINO."""
    ps = pairs.build_pairs(dataset, list(runs), min_gap=min_gap)
    write_jsonl(out, ps)
    typer.echo(f"{len(ps)} pares escritos em {out}")


@app.command("hash")
def hash_cmd(dataset: Path) -> None:
    """Hash (versão) de um dataset."""
    ds = load_dataset(dataset)
    typer.echo(f"{ds.nome}  uso={ds.uso}  casos={len(ds.cases)}  hash={ds.hash}")


def _github_output(key: str, value: str) -> None:
    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")
