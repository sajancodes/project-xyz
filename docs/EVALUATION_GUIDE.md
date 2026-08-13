# EVALUATION GUIDE

How the model is tested, why, and what each number means.

## 1. Why evaluation is the point

Training loss measures prediction of training data — NOT understanding. A model can memorize "John has a red car" and still fail on "What color is John's car?" in a new phrasing. Evaluation on **held-out** examples is the only way to detect real (vs memorized) behavior.

**Core rule**: evaluate on examples that never entered training. That is the whole scientific method of this project.

## 2. Data splits

| Split | Role | Example |
|---|---|---|
| TRAINING | seen by the optimizer | SEMV2E binding data (names Aaron, Bella, ...) |
| VALIDATION | monitor only (no tuning) | FineWeb held-out window (42,312 tokens) |
| HELD-OUT TEST | benchmarks/diagnostics | `data/semv2e_benchmark.json` (names Maya, Ravi, Sita, ...) |
| AD-HOC REAL TESTS | free-form qualitative | the 15-prompt chat test |

**Leakage guarantee**: SEMV2E training names are disjoint from benchmark names; the benchmark file itself is never read by the trainer.

## 3. Evaluation programs

| Program | Protocol | Decoding | Output |
|---|---|---|---|
| `src/evaluation/heldout_benchmark.py` | Cat A (10 cat, 51 prompts) + Cat B (7 gen) + plain QA | temp 0.3, top-k 40 | `experiments/bench_*.json` |
| `src/evaluation/canonical_eval.py` | same core, **greedy**, + sun/arithmetic/plain diagnostics | temp 0 (deterministic) | `experiments/eval_*.json` / `semv2e_bench_*_greedy.json` |
| `src/evaluation/semv2e_benchmark.py` | 600 prompts, 10 categories (A–J), chat-wrapped AND plain | greedy | `experiments/semv2e_*binding*.json` |
| `src/evaluation/arithmetic_diagnostic.py` | arithmetic (add 50, sub 40, word 30) | temp 0.3 | `experiments/arith_*.json` |
| `src/evaluation/sun_diagnostic.py` | sunrise/sunset (26) | temp 0.3 | `experiments/sun_*.json` |
| `src/evaluation/plain_qa_diagnostic.py` | plain QA (unwrapped) | temp 0.3 | `experiments/plainqa_*.json` |
| `src/evaluation/targeted_diagnostics.py` | why_context, negation, relation_reversal, transitive, arithmetic | temp 0.3 | — |
| `src/evaluation/evaluation_suite.py` | classic 8-category suite (grammar, factual QA, ...) | — | `experiments/eval_*.json` |

## 4. Cat A categories (held-out, strict scoring)

"Strict" = expected keyword appears in the first 8 generated tokens. "Loose" = anywhere. "Echo" = the model just repeats the prompt (counted as FAIL).

| Category | What it tests | Strong baseline? |
|---|---|---|
| sun_facts | sunrise→east, sunset→west | partial (33% strict; west weak) |
| relations | ownership / relationship | YES (100%) |
| state | location state | YES (100%) |
| why_context | why-questions over context | YES (80%) |
| negation | "not" handling | partial (75%) |
| entity_tracking | track entity across sentences | partial (80%) |
| transitive | chain reasoning (Alice gave Bob a pen) | YES (100%) |
| arithmetic | simple computation | NO (0%) |
| instruction | follow instructions | partial (86%) |
| context_retention | recall earlier fact | YES (100%) |

## 5. Cat B categories (generation quality)

- **Topic relevance** — is the output on-topic?
- **Repetition ratio** — how much of the output repeats.
- **Dialog drift** — does it spontaneously emit `User:`/`Assistant:` markers (bad).

## 6. Composite score (official protocol)

```
composite = 0.55*catA_strict + 0.15*catA_loose + 0.15*catB_topic
          + 0.10*(1 - catB_dialog_drift) + 0.05*(1 - catB_repetition)
```

**Do NOT read the composite as "percent understanding".** It is a weighted index of keyword-match scores. Free-form capability is a separate measurement.

## 7. Deterministic (canonical) evaluation — why we added it

Two interactive runs of the same checkpoint gave 13/15 then 5/15. Benchmarks run at temperature 0.3 are stochastic too. For decisions that matter (promote / don't promote), we use `canonical_eval.py` with **temp 0 (greedy, argmax)** — deterministic, reproducible, and it also aggregates the sun/arithmetic/plain diagnostics. Baseline canonical composite: **0.7463**.

## 8. The SEMV2E binding benchmark (600 prompts, categories A–J)

| Category | Description | Baseline (chat) |
|---|---|---|
| A single_fact | one fact, one question | 53% |
| B multi_fact | two facts, pick the right one | 37% |
| C distractor | noise facts present | 25% |
| D relation_reversal | reverse the relation direction | 75% |
| E paraphrase | different wording, same meaning | 58% |
| F pronoun | follow pronoun reference | 30% |
| G multi_hop | two-step chain | 52% |
| H sentence_order | re-ordered facts | 37% |
| I entity_generalization | entity with unseen spelling | 33% |
| J distractor_binding | hard binding under competition | 32% |

- **Chat-wrapped baseline: 43.2% (259/600)** · **Plain (no wrapper): 25.8% (155/600)**.
- The wrapper matters hugely (e.g. D_reversal 75% wrapped vs 0% plain).
- The binding gap (C, F, I, J) is exactly what SEMV2E training targets.

Leakage rules for this benchmark: names disjoint from training; templates differ; fixed deterministic seed; the JSON file (`data/semv2e_benchmark.json`) is evaluation-only.

## 9. Reading the real generation test

The 15-prompt ad-hoc test measures open behavior, not keyword matches. 5/15 does not "contradict" the 0.78 composite — they measure different things. Both are reported everywhere, deliberately.

## 10. Reproducibility checklist

Every evaluation record should include: checkpoint path + step + type, protocol, decoding (temp/top-k/top-p/seeded or greedy), max_new_tokens, chat wrapper on/off, and the raw output file. `canonical_eval.py` and `semv2e_benchmark.py` already do this — always keep their JSON outputs.

---

*Related: [docs/CHECKPOINT_GUIDE.md](CHECKPOINT_GUIDE.md), [docs/SEMANTIC_RESEARCH.md](SEMANTIC_RESEARCH.md).*