# Random Seeds and Reproducibility

Seeds help control sources of randomness, but they do not make every training run perfectly deterministic.

## Record seeds for

- Dataset shuffling and split creation.
- Synthetic-data generation.
- Model initialization where relevant.
- Training framework and data-loader randomness.
- Sampled evaluation generation.

Record which seed controls which component; a single global seed may not cover every source of randomness.

## Reproducibility limits

GPU kernels, distributed execution, library versions, hardware, data order, and nondeterministic operations can still produce differences. Record the environment and deterministic settings where supported.

For evaluation comparisons, prefer a stable protocol and preserve raw outputs. If sampling is used, repeat runs or report variance when feasible.

Do not claim an experiment is exactly reproducible merely because a seed was set. State the setup and the checks performed.