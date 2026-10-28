# Evaluation Reporting Policy

Evaluation reports should make it possible for another person to reproduce the comparison and understand its limitations.

## Always include

- Exact checkpoint path, identifier, and step where available.
- Evaluation code/version and dataset version.
- Prompt format and wrapper, decoding parameters, and maximum generation length.
- Scoring rules, normalization, aggregation, and treatment of invalid outputs.
- Aggregate and per-category scores, sample counts, and baseline comparison.
- Raw outputs or a clear reference to where they are stored.

## Be precise about claims

Distinguish measured benchmark results from subjective spot checks. State whether decoding is deterministic or sampled. Include failed categories and regressions, not only the best result.

Training loss is not a substitute for held-out task evaluation. A single aggregate score can hide severe weaknesses, so include category-level results whenever the benchmark supports them.

## Changes to the protocol

If the prompt, scoring rule, dataset, or decoding settings change, identify the change and avoid presenting the new score as directly comparable without qualification.

Use **NOT MEASURED** when a test has not run and **UNKNOWN** when evidence is missing. Never invent scores or infer success from a completed training process.