# Experiment Lifecycle

Use the same lifecycle for small smoke tests and full training runs so results remain traceable.

1. **Define:** write down the question, expected outcome, baseline, and success criteria before changing settings.
2. **Prepare:** select a track, commit SHA, configuration, dataset version, and fresh output directory.
3. **Validate inputs:** inspect schemas, sample records, split boundaries, and tokenizer behavior.
4. **Smoke test:** verify device access, forward/backward execution, loss computation, and checkpoint writing on a small run.
5. **Train:** preserve logs and checkpoints; record interruptions and configuration changes.
6. **Evaluate:** use the documented protocol and the same baseline settings.
7. **Review:** inspect aggregate and per-category scores, raw examples, and regressions.
8. **Decide:** keep, repeat, reject, or promote the experiment with an evidence-based explanation.

## Required artifacts

At minimum, retain the configuration, Git commit SHA, dataset/source description, environment versions, training log, checkpoint path, evaluation outputs, and a short report. Use the [Experiment Report Template](EXPERIMENT_REPORT_TEMPLATE.md).

A run is not successful merely because it completed. Success means the stated criteria were measured and met without unacceptable regressions.