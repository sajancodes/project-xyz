# Training Logging Requirements

Useful logs make a run diagnosable after a disconnect and allow results to be compared later.

## Minimum fields

- Experiment identifier and Git commit SHA.
- Start/end time and environment versions.
- Base model/checkpoint and configuration path.
- Dataset versions and split.
- Step, training loss, learning rate, and elapsed time.
- Checkpoint save events and paths.
- Validation results and warnings.
- Interruptions, restarts, OOMs, and configuration changes.

## Log quality

Use structured or consistently formatted output where possible. Keep secrets and private data out of logs. Avoid dumping entire training examples if they may contain sensitive information.

Save logs to persistent storage for temporary environments. Confirm the log path before a long run and periodically verify that the file is growing.

## Interpreting logs

Training loss describes the optimized training objective; it does not establish generalization. Compare it with held-out evaluation and raw outputs. Mark missing metrics as unavailable rather than estimating them.

Pair the log with the [Experiment Report Template](EXPERIMENT_REPORT_TEMPLATE.md) and preserve the exact config used.