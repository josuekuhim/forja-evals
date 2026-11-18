# forja-evals

**A régua do [Forja](../forja).** Tudo que é treinado ou otimizado no sistema passa por aqui, com o mesmo dataset e o mesmo gate.

> Resultado: _pendente_ — kappa juiz vs. humano e o primeiro PR bloqueado pelo gate entram aqui.

## O que faz

| | |
| --- | --- |
| **Datasets versionados** | JSONL imutável; a versão é o SHA-256 do conteúdo. Comparar execuções com hashes diferentes é erro, não aviso. |
| **Runner** | Assíncrono, para qualquer endpoint OpenAI-compatível (forja-serve, vLLM, APIs). Concorrência, rate limit, retry com backoff e `Retry-After`, cache em SQLite. |
| **Métricas determinísticas** | Código sugerido compila e **passa nos testes** (subprocesso isolado com timeout), recusa detectada, palavras-chave do bug. |
| **LLM-as-judge** | Rubrica por categoria, saída JSON, parsing tolerante. Só decide onde não há métrica determinística. |
| **Calibração** | Rotulagem humana às cegas, estratificada e retomável; kappa de Cohen (simples e quadrático), viés médio do juiz. |
| **Gate de regressão** | Bootstrap **pareado** (10k reamostragens), IC 95%, efeito mínimo. GitHub Action reutilizável. |
| **Pares de preferência** | Gera pares para o DPO, e **recusa** datasets de avaliação (anti-circularidade). |
| **Operacional** | Latência p50/p95/p99 (sem cache), tokens, custo total e por 1k requisições, erros de infraestrutura separados de erros do modelo. |

## Decisões e trade-offs

- **Determinístico primeiro, juiz depois.** Se dá para verificar (testes passam? recusou?), essa é a nota. O juiz é caro, enviesado e precisa de calibração; ele só entra onde não tem outro jeito.
- **Bootstrap pareado, não médias soltas.** Os dois sistemas respondem os mesmos casos; parear cancela a variância entre casos e o gate detecta quedas bem menores com o mesmo n.
- **O gate decide pelo escopo geral.** Com ~15 casos por categoria, o IC por categoria é largo demais para bloquear build; regressões por categoria viram alerta.
- **Erro de infraestrutura não é nota zero.** Timeout ou 5xx sai da média e é reportado. Se passar de 5% dos casos, o gate falha (avaliação inválida, não modelo ruim).
- **Arquivos em vez de Postgres na v1.** Cada execução é um diretório (`config.yaml`, `results.jsonl`, `summary.json`) que dá para commitar como baseline. Postgres + dashboard entram na v2, junto com evals online.

## Uso

```bash
uv sync
uv run forja-evals hash datasets/forja-seed-v0
uv run forja-evals run configs/qwen-base-local.yaml          # precisa de um endpoint em localhost:8000
uv run forja-evals run configs/api-referencia.yaml           # precisa de OPENROUTER_API_KEY
uv run forja-evals report runs/<a> runs/<b>                  # tabela do hub
uv run forja-evals compare runs/<baseline> runs/<candidato>  # gate (exit 1 se regrediu)
uv run forja-evals label runs/<id> --n 30                    # rotulagem humana às cegas
uv run forja-evals kappa runs/<id>
uv run forja-evals pairs datasets/<treino> runs/<a> runs/<b> --out pairs.jsonl
```

Para servir o modelo base localmente em CPU, sirva um GGUF quantizado com o `llama-server` (llama.cpp), que expõe `/v1/chat/completions` na porta configurada.

### Gate num PR (em outro repositório do Forja)

```yaml
- uses: <usuario>/forja-evals@main
  with:
    config: evals/candidato.yaml
    baseline: evals/baseline     # execução commitada
```

## Categorias

`explicacao` (juiz) · `deteccao_bug` (palavras-chave + juiz) · `correcao` (testes executados) · `api` (juiz vs. referência) · `recusa` (detecção de recusa).

## Limitações

- `forja-seed-v0` tem 16 casos: valida o pipeline, não mede qualidade. A meta da v1 é ~50 por categoria, com rótulo humano.
- A execução de testes **não é um sandbox**. No CI, rode em container descartável sem segredos nem rede.
- Detecção de recusa por regex em português; tem falso negativo para recusas criativas (o juiz cobre).
- Sem TTFT ainda: o runner não usa streaming. Entra junto com o forja-serve.
