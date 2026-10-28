# Generation Settings for Evaluation

Generation settings influence model outputs and can change measured benchmark results. Record them with every evaluation.

## Settings to record

- Greedy decoding or sampling.
- Temperature, top-k, and top-p when applicable.
- Maximum new tokens and any stop sequences.
- Repetition penalties or other logits processors.
- Chat template, system prompt, and user prompt formatting.
- Seed and library/runtime versions for sampled generation.

Only record settings that the evaluation actually uses; do not imply that an unused option affected the run.

## Deterministic comparisons

Greedy decoding often reduces sampling variation, but it does not guarantee identical results across different model implementations, library versions, prompt templates, or hardware kernels. Save raw outputs and environment metadata.

## Avoid protocol drift

Use one canonical settings file or shared evaluation function for baseline and candidate checkpoints. If settings change, label the new protocol and avoid mixing its scores with earlier results without explanation.

Generation settings are part of the benchmark definition, not cosmetic implementation details.