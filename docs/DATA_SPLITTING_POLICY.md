# Dataset Splitting Policy

The purpose of a split is to estimate behavior on examples that did not influence training or model selection.

## General policy

- Separate training data from validation data used for decisions and held-out test data used for final reporting.
- Choose the split rule before inspecting final test scores.
- Keep a stable split definition across comparable experiments.
- When examples share a source document, entity, generated template, or underlying fact, consider grouping them before splitting.
- Store split IDs or a reproducible assignment rule; do not rely on undocumented manual file copies.

## Synthetic data

Randomly generated strings can still leak structure when examples share a template or underlying fact. Where feasible, hold out entities, relation combinations, templates, or generator seeds to test generalization beyond memorized patterns.

## Benchmark protection

Do not use held-out benchmark examples to generate training records, tune prompts, select hyperparameters, or iteratively repair individual failures before reporting the same benchmark as untouched.

If benchmark information influenced a decision, disclose that fact and use a separate untouched evaluation set for a stronger claim.

## Reporting

Report split sizes, assignment method, grouping strategy, known overlap risks, and the exact dataset version. A split is only as independent as its construction.