# Experiment Report Template

Copy this template for a new training or evaluation run. Keep the raw output files and configuration next to the report whenever possible.

## Summary

- **Experiment ID/name:**
- **Date (UTC):**
- **Track:** Track A / Track B
- **Status:** Planned / Running / Completed / Failed / Evaluation pending
- **Git commit SHA:**
- **Owner:**

## Setup

- **Goal / hypothesis:**
- **Base model or parent checkpoint:**
- **Configuration file:**
- **Dataset sources and versions:**
- **Train / validation / held-out split:**
- **Hardware and GPU memory:**
- **Python / PyTorch / CUDA / library versions:**
- **Command or notebook cells:**
- **Seed:**

## Training

- **Steps planned / completed:**
- **Effective batch size and sequence length:**
- **Optimizer and learning-rate schedule:**
- **Final training loss:**
- **Validation metric:**
- **Checkpoint paths:**
- **Resume events or interruptions:**
- **Logs:**

## Evaluation

- **Protocol and code version:**
- **Checkpoint path and embedded step:**
- **Decoding settings:**
- **Wrapper format:** Chat-wrapped / Plain / Other
- **Aggregate score:**
- **Per-category scores:**
- **Raw output file:**
- **Baseline compared against:**
- **Regression checks:**

## Findings

### What improved?

List measured changes and link to the exact output file.

### What regressed or failed?

Include negative results and categories that remain weak.

### Limitations

State uncertainty, small sample sizes, protocol differences, and any unverified assumptions.

## Decision

- [ ] Keep as an experiment only
- [ ] Repeat or investigate
- [ ] Candidate for promotion
- [ ] Promoted after reproducible comparison

**Reason for decision:**

## Evidence status

Use one label for each major claim:
- **VERIFIED** — supported by saved logs, raw outputs, or checkpoint metadata.
- **NOT MEASURED** — the relevant run/evaluation has not been performed.
- **UNKNOWN** — evidence is missing or ambiguous.

Do not treat training loss, a successful process exit, or a single ad-hoc generation as sufficient proof of generalization.

Related: [Reproducibility Checklist](REPRODUCIBILITY_CHECKLIST.md), [Evaluation Guide](EVALUATION_GUIDE.md), [Checkpoint Guide](CHECKPOINT_GUIDE.md).
