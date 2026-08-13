# SEMANTIC RESEARCH

The research thread of Project XYZ: why we care about semantics, what we observed, and what we are testing now.

## 1. The central question

Can a **14.66M-parameter** decoder-only transformer — trained on a single 6 GB laptop GPU — demonstrate **task-level semantic understanding** beyond fluent next-token prediction?

The model is tiny compared to modern LLMs. The question is not "can it be smart like GPT" but **"can a small model reliably bind simple facts and answer unseen questions"** — and, if so, *how well and under what conditions*.

## 2. The semantic binding hypothesis

The SEMV2E experiment targets **semantic binding**: the model must reliably carry the triple

```
entity  --relationship-->  object/property
```

and use it to answer a question, even when there are competing facts, pronouns, reversed relations, reordered sentences, or paraphrase.

Example of the *hard* version the baseline fails:

```
Sajan owns a red car. Ravi owns a blue bicycle.
What color is Ravi's bicycle?        → blue   (baseline often says red / Sajan)
```

The hypothesis: **targeted, leakage-safe, paraphrased procedural training can teach the small model to construct, maintain, and retrieve simple semantic relationships** — turning `language → entities → relationships → properties → context → answer`.

## 3. What the measurements have shown (chronological)

| Stage | Measurement | Reading |
|---|---|---|
| Pretraining 106k | factual QA 0% | next-token ≠ QA |
| Semantic fine-tune 5k | factual QA 100%, grammar 33.3% | catastrophic forgetting |
| Mixed 2k (90/10) | factual QA 100%, grammar 55.6% | mixture prevents forgetting |
| Mixed 10k (constant LR) | factual QA 28.6% | constant LR oscillates |
| SEMV2 3k | composite 0.697 | loss-masking + procedural + LR schedule work |
| SEMV2B 3k | composite 0.732 | enriched data + continuation |
| SEMV2C 3k | **composite 0.780** | best model; weak sun/arithmetic remain |
| Real test 15 prompts | **5/15 (33%)** | benchmark ≠ free-form behavior |
| SEMV2E binding (baseline) | chat 43.2% / plain 25.8% | binding gap confirmed |

## 4. Observed phenomena (what the model actually does)

- **Strong** (held-out, canonical greedy baseline): relations, state, transitive, context_retention → 100%. Instruction → 86%. Single-fact retrieval in chat → ~53% on the binding benchmark.
- **Partial**: negation 75%, why_context 80%, entity_tracking 80%.
- **Weak**: sun_facts 33% (rise_east ~62% but set_west ~12%), multi-fact binding 25–37%, pronoun 30%, entity generalization 33%, distractor binding 32%.
- **Absent**: arithmetic 0% across every protocol and scale tested (addition, subtraction, multi-step).
- **Format-dependent**: plain (no wrapper) drops to 25.8% vs 43.2% chat on the binding benchmark.

## 5. Why benchmark and real-test disagree

The 0.78 composite and the 5/15 real test are both TRUE. The benchmark measures narrow keyword-match tasks with short expected answers; free-form prompts stress open-ended behavior, multi-fact reasoning, and instruction content. We report both everywhere, and we added **deterministic (greedy) canonical evaluation** so that promotion decisions do not depend on sampling luck.

## 6. The SEMV2E experimental design

### Goal
Improve semantic binding robustly, verified on the 600-question binding benchmark, without destroying strong areas (relations/state/transitive/context).

### Training data (leakage-safe)
- 55% binding, 15% instruction (Alpaca), 15% existing semantic/reasoning, 10% FineWeb (preservation), 5% arithmetic.
- Training names (Aaron, Bella, ...) DISJOINT from benchmark names (Maya, Ravi, ...). Overlap guard raises a hard error.
- Procedural generator: `src/utils/semv2e_data.py` (seed 20260813). Fixed strings never repeated — facts + surface variations.

### Training
- From `checkpoints/best/checkpoint_best.pt`; 5000 steps; AdamW peak 1e-4, warmup 150 + cosine to 10%; batch 16; assistant-loss masking; FineWeb held-out val monitor.
- **COMPLETED**: final train loss 0.2976; val 4.3915; ~963 s GPU. Milestones at 500/1000/2000/3000/5000.

### Evaluation (the current next step)
- Run the binding benchmark + canonical eval on **each** milestone (500, 1000, 2000, 3000, 5000).
- Compare to baseline (chat 43.2% / plain 25.8% on binding; composite 0.7463 canonical / 0.780 held-out).
- Per-category regression analysis vs baseline.
- Promotion only if genuine, reproducible improvement with no unacceptable regression.

## 7. Promotion criteria (mission-defined)

A SEMV2E checkpoint becomes the new best ONLY if:
1. Benchmark improves genuinely (deterministic, reproducible).
2. Meaningful improvement in semantic binding.
3. No unacceptable regression in strong areas.
4. Held-out validity maintained (leakage rules).
5. Result does not depend on decoding randomness.

## 8. What we deliberately do NOT claim

- No human-like understanding; we measure demonstrable task-level behavior.
- **Arithmetic**: we do NOT declare a capacity limit from 0% alone without further controlled experiments — it stays a secondary experiment.
- No result that was not measured. Anything unknown is recorded as UNKNOWN / NOT MEASURED.

---

*Related: [docs/DATA_GUIDE.md](DATA_GUIDE.md), [docs/EVALUATION_GUIDE.md](EVALUATION_GUIDE.md).*