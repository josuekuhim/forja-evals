# forja-evals

**The yardstick of [Forja](../forja).** Everything trained or optimized in the system goes through here, with the same dataset and the same gate.

> **Result:** it already did its job twice. As a gate, it **rejected** int4 quantization in forja-serve, after gaining a degeneration metric the score alone could not see. As a yardstick, it holds `review-eval-v0` (199 real bugs from 2025–2026 bug-fix commits plus 199 already-fixed files), where post-training on real data went from 9 to 27 bugs found with no forgetting on `forja-seed-v0`. Still pending: judge × human kappa and the first PR blocked by the gate in CI.

## First real baseline

Qwen2.5-Coder-1.5B-Instruct (GGUF Q4_K_M) on CPU (Ryzen 5 5625U, 4 threads) via `llama-server`, with the default system prompt and no judge ([`baselines/qwen2.5-coder-1.5b-q4km-sem-juiz`](baselines/qwen2.5-coder-1.5b-q4km-sem-juiz)):

| Category | Score | What happened |
| --- | --- | --- |
| correcao (fix) | 3/4 | On the palindrome, it claimed the code was already correct |
| deteccao_bug (bug detection) | 0/4 | Wrong diagnosis on all four; "fixes" identical to the original or with a new bug |
| recusa (refusal) | 0/3 | Invented the return value of `processa_pedido` and a tutorial for `json.dumps`' nonexistent `turbo_mode` |
| explicacao, api | — | No judge (see below) |

Overall score 0.273 on 11 scored cases; p50 latency of 9.9 s. That is exactly what post-training has to attack: confident hallucination and bug diagnosis.

### With a local judge: Gemma 4 26B-A4B

The same model, now with the judge [`ggml-org/gemma-4-26B-A4B-it-Q4_0`](configs/qwen-base-local-juiz-gemma.yaml), also on CPU (MoE, ~4B active parameters, ~8 tok/s). Result in [`baselines/qwen2.5-coder-1.5b-q4km-juiz-gemma4-26b`](baselines/qwen2.5-coder-1.5b-q4km-juiz-gemma4-26b):

| Category | Score | Source |
| --- | --- | --- |
| explicacao (explanation) | 0.75 | judge: 5, 5 and 2 (in 0001 the example says 17 ≥ 18) |
| api | 0.75 | judge: 5 and 3 (in 0001 it claims different types are compared as strings; in Python 3 it is a `TypeError`) |
| correcao (fix) | 3/4 | tests (same as the no-judge baseline) |
| deteccao_bug (bug detection) | 0/4 | deterministic; the judge gave 1 on all four, agreeing |
| recusa (refusal) | 0/3 | refusal regex |

Overall score **0.422** on 16 cases; 33 min for the whole run (the judge reasons ~1,300 tokens per case).

The judge evaluates every case, but **the deterministic metric takes precedence** when there is one, and that has already paid off:

- `recusa-0003` ("does this line have a bug?" with a line and no context): the model claimed it **has** a bug because `calcula` "is not defined". The judge gave **5**; the refusal regex gave 0, which is right. A lenient judge is the classic error, and that is why it is the fallback.
- `correcao-0001/0002`: the tests pass, but the judge gave 4 and 3 because the model said the original code "was already correct". The test does not see that; the judge does. It argues for a mixed score (tests + judge) for `correcao` in a v2.

Infrastructure lesson: the default HTTP timeout (120 s) is shorter than a long judgment at 8 tok/s. The client cancelled, retried, and the same case restarted in a loop. There are now `timeout_s` and `juiz_timeout_s` in the config.

**Not yet valid as a yardstick:** the answers still need labeling (`forja-evals label`) and the judge × human kappa measured (`forja-evals kappa`).

### First use as a gate: quantization in forja-serve

[forja-serve](../forja-serve) served Qwen2.5-Coder-0.5B in fp32, int8 and int4, and `compare` ran each variant against fp32. In the first version of the gate, **both passed**, but int4 looped in 8 of 16 answers. The gate missed it for three reasons:

- **floor effect:** the 0.5B's score was already ~0 in fp32;
- **a refusal-regex false positive:** "`calcula`, which does not exist" was the model asserting a bug, not refusing;
- **no degeneration metric.**

What changed in the yardstick:

| Change | Where |
| --- | --- |
| `repeticao` (fraction of repeated 8-grams) and `degenerou` (≥ 0.30) on every case. The threshold was calibrated on real answers: healthy ones stay ≤ 0.16, loops fall between 0.36 and 0.93. | `metrics.py` |
| `finish_reason` saved per case; `saude` (health) in the summary (loop rate and rate cut by `max_tokens`) | `client.py`, `runner.py` |
| Gate: paired bootstrap of the **loop-free** fraction over all cases (not only scored ones). A regression there blocks, even with a tied score. | `gate.py` |
| The refusal regex accepts "does not exist" only when it refers to a parameter/option | `metrics.py` |
| `forja-evals rescore <runs>`: recomputes metrics and summaries of saved runs when the yardstick changes, without calling the model | `cli.py` |

After `rescore`, the result: int8 is approved (1/16 in a loop, same as fp32) and int4 is **rejected**, with 50% of answers in a loop and Δ −0.44 on the loop-free fraction, CI [−0.69; −0.19]. The 1.5B baselines' scores did not change (0.273 and 0.422).

### Real bugs: `review-eval-v0`

[`datasets/review-eval-v0`](datasets/review-eval-v0) is built by [forja-dados](../forja-dados) from bug-fix commits in public Python repositories with permissive licenses: 199 files before the fix (the answer key is the line the commit changed) and the same 199 files after it (no bug). The commits are from 2025–2026; forja-posttrain's real training data stops at 2024-09-19. It is marked `uso: avaliacao`, so `forja-posttrain` and `forja-evals pairs` refuse to train on it. The labeler that decides whether a commit is a bug fix was validated by hand: ~80% precision (95% CI 65–90%, n = 40). Its results are in the [forja-posttrain README](../forja-posttrain).

### Online eval: the first forja-review trace

[forja-review](../forja-review) records one OpenTelemetry span per model call (diff, answer, published comments). `forja-evals online` samples those spans, and the judge rates **each comment** (useful or not) and the file's review (1 to 5). Once there is 👍/👎 from users, the same command measures judge × human agreement.

First run ([`baselines/online-forja-review-local`](baselines/online-forja-review-local)): 29 calls from the planted-bug tests, 12 sampled, local Gemma 4 26B judge.

| Reviewer model | Reviews judged | Mean score (1-5) | Useful comments (judge) |
| --- | --- | --- | --- |
| Qwen2.5-Coder 1.5B | 7 | 1.43 | 1 of 11 |
| Qwen2.5-Coder 0.5B (forja-serve) | 3 | 1.00 | 0 of 3 |

- **The judge agrees with the answer key.** The only comment it marked as useful is the 1.5B's only real hit (the `//` that loses the last page). Its justifications cite the missed bugs: mutable argument, SQL injection, inverted TTL.
- **A defect found and fixed:** the 2 sampled Gemma calls were a truncated answer and one that **aborted on timeout**. A failed call publishes nothing and should not be judged. The contract now reads the span's status (`LLMInteraction.erro`), and the sample excludes those calls.
- **No humans yet:** without 👍/👎, the judge's score is not calibrated in production. That only changes with the App installed on real repositories.

## Determinism (and why the local config uses 1 slot)

In the first run with `--parallel 2`, **only 1 of 16 answers** repeated between two runs at temperature 0, and one score changed. Isolating it:

- Same request, 1 slot, sequential: always identical output.
- With llama.cpp's *prompt cache* on, the output changes depending on whether the prefix (the system prompt, the same for every case) is reused or recomputed: the floating-point math runs in a different order.
- With 2 slots, which prefix is reused depends on arrival order, so each run gives a different result.

Fix: `extra_body: {cache_prompt: false}` on the endpoint and `concorrencia: 1` (concurrency). Result: **16/16 identical answers** across two full runs without the cache. Noise that does not come from the model becomes a false regression in the gate; this comes back as a topic in forja-serve (batching vs. reproducibility).

## What it does

| | |
| --- | --- |
| **Versioned datasets** | Immutable JSONL; the version is the content's SHA-256. Comparing runs with different hashes is an error, not a warning. |
| **Runner** | Async, for any OpenAI-compatible endpoint (forja-serve, vLLM, APIs). Concurrency, rate limiting, retry with backoff and `Retry-After`, SQLite cache. |
| **Deterministic metrics** | Suggested code compiles and **passes the tests** (isolated subprocess with a timeout), refusal detected, bug keywords. |
| **LLM-as-judge** | Per-category rubric, JSON output, tolerant parsing. It only decides where there is no deterministic metric. |
| **Calibration** | Blind, stratified, resumable human labeling; Cohen's kappa (simple and quadratic), mean judge bias. |
| **Regression gate** | **Paired** bootstrap (10k resamples), 95% CI, minimum effect. Reusable GitHub Action. |
| **Preference pairs** | Generates pairs for DPO, and **refuses** evaluation datasets (anti-circularity). |
| **Operational** | p50/p95/p99 latency (without cache), tokens, total cost and cost per 1k requests, infrastructure errors separated from model errors. |

## Decisions and trade-offs

- **Deterministic first, judge second.** If it can be verified (do the tests pass? did it refuse?), that is the score. The judge is expensive, biased and needs calibration; it only comes in where there is no other way.
- **Paired bootstrap, not loose means.** Both systems answer the same cases; pairing cancels the between-case variance, and the gate detects much smaller drops with the same n.
- **The gate decides on the overall scope.** With ~15 cases per category, the per-category CI is too wide to block a build; per-category regressions become warnings.
- **An infrastructure error is not a zero score.** A timeout or 5xx leaves the mean and is reported. If it exceeds 5% of cases, the gate fails (invalid evaluation, not a bad model).
- **Files instead of Postgres in v1.** Each run is a directory (`config.yaml`, `results.jsonl`, `summary.json`) that can be committed as a baseline. Postgres + a dashboard come in v2.

## Usage

```bash
uv sync
uv run forja-evals hash datasets/forja-seed-v0
uv run forja-evals run configs/qwen-base-local.yaml          # needs an endpoint at localhost:8000
uv run forja-evals run configs/api-referencia.yaml           # needs OPENROUTER_API_KEY
uv run forja-evals report runs/<a> runs/<b>                  # the hub table
uv run forja-evals compare runs/<baseline> runs/<candidate>  # gate (exit 1 on regression)
uv run forja-evals online configs/online-forja-review-local.yaml  # online evals (forja-review traces)
uv run forja-evals rescore baselines/*                        # the yardstick changed: recompute without calling the model
uv run forja-evals label runs/<id> --n 30                    # blind human labeling
uv run forja-evals kappa runs/<id>
uv run forja-evals pairs datasets/<train> runs/<a> runs/<b> --out pairs.jsonl
```

To serve the base model locally on CPU, use forja-serve, or serve a quantized GGUF with `llama-server` (llama.cpp), which exposes `/v1/chat/completions` on the configured port.

### Gate on a PR (in another Forja repository)

```yaml
- uses: <user>/forja-evals@main
  with:
    config: evals/candidate.yaml
    baseline: evals/baseline     # committed run
```

## Categories

`explicacao` (explanation, judge) · `deteccao_bug` (bug detection, keywords + judge) · `correcao` (fix, executed tests) · `api` (judge vs. reference) · `recusa` (refusal detection).

## Limitations

- `forja-seed-v0` has 16 cases: it validates the pipeline and catches large damage, it does not measure fine quality differences. The v1 target is ~50 per category, with human labels.
- Test execution **is not a sandbox**. In CI, run it in a disposable container with no secrets and no network.
- Refusal detection is a Portuguese regex; it has false negatives for creative refusals (the judge covers them).
- No TTFT yet: the runner does not use streaming.
- Datasets, prompts and field names are in Portuguese, the target language of the models being measured.
