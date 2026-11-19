## forja-evals: `qwen2.5-coder-0.5b-forja-serve-int4` vs. `qwen2.5-coder-0.5b-forja-serve-fp32`

✅ **Sem regressão**

| Escopo | n | Δ score | IC 95% | Veredito |
| --- | --- | --- | --- | --- |
| **geral** | 11 | +0.091 | [+0.000, +0.273] | sem diferença significativa |
| correcao | 4 | +0.000 | [+0.000, +0.000] | sem diferença significativa |
| deteccao_bug | 4 | +0.000 | [+0.000, +0.000] | sem diferença significativa |
| recusa | 3 | +0.333 | [+0.000, +1.000] | sem diferença significativa |

<sub>Bootstrap pareado, 10k reamostragens. Regressão = IC inteiro abaixo de 0 e queda ≥ 0.02. O gate decide pelo escopo geral; categorias com n pequeno têm IC largo.</sub>