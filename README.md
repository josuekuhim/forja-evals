# forja-evals

**A régua do [Forja](../forja).** Tudo que é treinado ou otimizado no sistema passa por aqui, com o mesmo dataset e o mesmo gate.

> Resultado: _pendente_ — kappa juiz vs. humano e o primeiro PR bloqueado pelo gate entram aqui.

## Primeiro baseline real

Qwen2.5-Coder-1.5B-Instruct (GGUF Q4_K_M) em CPU (Ryzen 5 5625U, 4 threads) via `llama-server`, com o prompt de sistema padrão e sem juiz ([`baselines/qwen2.5-coder-1.5b-q4km-sem-juiz`](baselines/qwen2.5-coder-1.5b-q4km-sem-juiz)):

| Categoria | Score | O que aconteceu |
| --- | --- | --- |
| correcao | 3/4 | No palíndromo, afirmou que o código já estava correto |
| deteccao_bug | 0/4 | Diagnóstico errado nos quatro; "correções" iguais ao original ou com bug novo |
| recusa | 0/3 | Inventou o retorno de `processa_pedido` e um tutorial do inexistente `turbo_mode` do `json.dumps` |
| explicacao, api | — | Sem juiz (ver abaixo) |

Score geral 0,273 em 11 casos com nota; latência p50 de 9,9s. É exatamente o que o pós-treino precisa atacar: alucinação confiante e diagnóstico de bug.

### Com juiz local: Gemma 4 26B-A4B

Mesmo modelo, agora com o juiz [`ggml-org/gemma-4-26B-A4B-it-Q4_0`](configs/qwen-base-local-juiz-gemma.yaml) também em CPU (MoE, ~4B parâmetros ativos, ~8 tok/s). Resultado em [`baselines/qwen2.5-coder-1.5b-q4km-juiz-gemma4-26b`](baselines/qwen2.5-coder-1.5b-q4km-juiz-gemma4-26b):

| Categoria | Score | Fonte |
| --- | --- | --- |
| explicacao | 0,75 | juiz: 5, 5 e 2 (no 0001 o exemplo diz que 17 ≥ 18) |
| api | 0,75 | juiz: 5 e 3 (no 0001 afirma que tipos diferentes são comparados como string; em Python 3 é `TypeError`) |
| correcao | 3/4 | testes (iguais ao baseline sem juiz) |
| deteccao_bug | 0/4 | determinístico; o juiz deu nota 1 nos quatro, concordando |
| recusa | 0/3 | regex de recusa |

Score geral **0,422** em 16 casos; 33 min para o run inteiro (o juiz raciocina ~1.300 tokens por caso).

O juiz avalia todos os casos, mas **a métrica determinística tem precedência** quando existe, e isso já se pagou:

- `recusa-0003` ("essa linha tem bug?" com uma linha sem contexto): o modelo afirmou que **tem** bug porque `calcula` "não está definida". O juiz deu **5**; a regex de recusa deu 0, que é o certo. Juiz complacente é o erro clássico, e é por isso que ele é fallback.
- `correcao-0001/0002`: testes passam, mas o juiz deu 4 e 3 porque o modelo disse que o código original "já estava correto". O teste não vê isso; o juiz vê. Dá argumento para uma nota mista (testes + juiz) em `correcao` numa v2.

Lição de infraestrutura: o timeout HTTP padrão (120 s) é menor que um julgamento longo a 8 tok/s. O cliente cancelava, tentava de novo e o mesmo caso recomeçava em loop. Agora há `timeout_s` e `juiz_timeout_s` na config.

**Ainda não vale como régua**: falta rotular as respostas (`forja-evals label`) e medir o kappa juiz × humano (`forja-evals kappa`).

## Determinismo (e por que a config local usa 1 slot)

Na primeira execução com `--parallel 2`, **só 1 de 16 respostas** se repetiu entre duas rodadas com temperatura 0, e uma nota mudou. Isolando:

- Mesma requisição, 1 slot, sequencial: saída sempre idêntica.
- Com o *prompt cache* do llama.cpp ligado, a saída muda conforme o prefixo (o system prompt, igual em todos os casos) é reaproveitado ou recalculado: a conta em ponto flutuante sai em outra ordem.
- Com 2 slots, qual prefixo é reaproveitado depende da ordem de chegada, então cada execução dá um resultado.

Correção: `extra_body: {cache_prompt: false}` no endpoint e `concorrencia: 1`. Resultado: **16/16 respostas idênticas** em duas execuções completas sem cache. Ruído que não vem do modelo vira regressão falsa no gate; isso volta como tema no forja-serve (batching vs. reprodutibilidade).

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
