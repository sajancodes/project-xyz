# Operations Documentation Index

Use these documents together when planning, running, debugging, or reporting a Project XYZ experiment.

| Need | Start here | Why |
|---|---|---|
| Prepare a repeatable run | [Reproducibility Checklist](REPRODUCIBILITY_CHECKLIST.md) | Captures code, data, environment, checkpoint, and evaluation details. |
| Diagnose a failed run | [Training Failure Triage](FAILURE_TRIAGE.md) | Covers GPU access, OOM, runtime interruption, regressions, and leakage. |
| Record results consistently | [Experiment Report Template](EXPERIMENT_REPORT_TEMPLATE.md) | Keeps measured results, negative findings, and evidence status together. |
| Understand scoring | [Evaluation Guide](EVALUATION_GUIDE.md) | Defines protocols, categories, and the composite metric. |
| Protect and verify model states | [Checkpoint Guide](CHECKPOINT_GUIDE.md) | Documents checkpoint lineage and promotion rules. |
| Run Track B on Colab | [Colab Training Guide](COLAB_TRAINING.md) | Describes notebook execution and persistent checkpoints. |
| Understand data sources and risks | [Data Guide](DATA_GUIDE.md) | Explains dataset roles, formats, and leakage risks. |

## Suggested workflow

1. Before a run, complete the [Reproducibility Checklist](REPRODUCIBILITY_CHECKLIST.md).
2. Keep each run's configuration, logs, checkpoints, and raw evaluation outputs in its own output location.
3. If something fails, use [Training Failure Triage](FAILURE_TRIAGE.md) and preserve the relevant logs.
4. After evaluation, fill in the [Experiment Report Template](EXPERIMENT_REPORT_TEMPLATE.md).
5. Promote a checkpoint only after comparing the same held-out protocol and reviewing regressions.

These guides describe a process, not proof that any particular experiment has run or passed. Record unmeasured outcomes as **NOT MEASURED** rather than inferring success from training loss.
