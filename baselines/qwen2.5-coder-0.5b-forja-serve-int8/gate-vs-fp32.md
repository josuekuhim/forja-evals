## forja-evals: `qwen2.5-coder-0.5b-forja-serve-int8` vs. `qwen2.5-coder-0.5b-forja-serve-fp32`

✅ **Sem regressão**

| Escopo | n | Δ score | IC 95% | Veredito |
| --- | --- | --- | --- | --- |
| **geral** | 11 | +0.000 | [+0.000, +0.000] | sem diferença significativa |
| correcao | 4 | +0.000 | [+0.000, +0.000] | sem diferença significativa |
| deteccao_bug | 4 | +0.000 | [+0.000, +0.000] | sem diferença significativa |
| recusa | 3 | +0.000 | [+0.000, +0.000] | sem diferença significativa |
| sem loop (todos os casos) | 16 | +0.000 | [-0.188, +0.188] | sem diferença significativa |

| Saúde (todos os casos) | baseline | candidato |
| --- | --- | --- |
| em loop | 6% | 6% |
| cortadas por max_tokens | 12% | 12% |

<sub>Bootstrap pareado, 10k reamostragens. Regressão = IC inteiro abaixo de 0 e queda ≥ 0.02. O gate decide pelo escopo geral e pela linha 'sem loop'; categorias com n pequeno têm IC largo.</sub>