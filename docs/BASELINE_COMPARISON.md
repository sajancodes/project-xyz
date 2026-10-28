# Baseline Comparison Procedure

A candidate checkpoint should be compared with a known baseline under the same evaluation conditions.

## Procedure

1. Record the exact baseline and candidate checkpoint identifiers.
2. Run both with the same benchmark version, prompt wrapper, decoding settings, and scoring code.
3. Save raw outputs separately and label them with checkpoint and protocol metadata.
4. Compare overall score, per-category results, invalid-output rate, and representative examples.
5. Review regressions as well as improvements.
6. Repeat the comparison if results are noisy or sensitive to sampling.
7. Record a keep/reject/retest decision and its evidence.

## Fair comparison

Do not compare scores produced by different test sets or incompatible scoring rules as though they were one controlled experiment. If a protocol change is necessary, report both the change and its consequences.

## Promotion bar

A higher aggregate score alone is not sufficient if important capabilities regress, the result depends on leakage, or the evaluation cannot be reproduced. Follow the promotion rules in the [Checkpoint Guide](CHECKPOINT_GUIDE.md).

Keep a baseline available and immutable so future experiments have a stable reference.