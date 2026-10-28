# Experiment Reproducibility Checklist

Use this checklist for every training or evaluation run. The goal is to make results comparable and to distinguish verified measurements from assumptions.

## Before training

- [ ] Record the Git commit SHA and working branch.
- [ ] Identify the track: Track A (small model trained from scratch) or Track B (Qwen 4B QLoRA). Do not mix their configurations or artifacts.
- [ ] Record the exact configuration file, base model/checkpoint, tokenizer, dataset sources, and dataset split.
- [ ] Record Python, PyTorch, CUDA, Transformers, PEFT, TRL, and bitsandbytes versions as applicable.
- [ ] Confirm the selected device is the intended GPU; record GPU model and available VRAM.
- [ ] Run the relevant dataset/schema checks and a small forward/backward smoke test before a long run.
- [ ] Choose a new output directory. Never overwrite the known-good checkpoint or its backup.

## During training

- [ ] Save logs and checkpoints to persistent storage when using a temporary runtime such as Colab.
- [ ] Record step, loss, learning rate, effective batch size, sequence length, and any resume point.
- [ ] If resuming, verify the checkpoint path and step before starting.
- [ ] Treat a completed process as evidence that the run finished, not evidence that the model improved.

## Before evaluation

- [ ] Select the exact checkpoint by path and inspect its embedded metadata where available.
- [ ] Use the same evaluation protocol when comparing checkpoints.
- [ ] Keep training, validation, and held-out test data separate.
- [ ] Confirm benchmark examples were not used to generate training data or tune the run.
- [ ] Record decoding settings, wrapper format, maximum generation length, and random seed where relevant.

## After evaluation

- [ ] Save raw outputs and machine-readable score files alongside the report.
- [ ] Report per-category scores as well as aggregate metrics.
- [ ] Report qualitative/free-form tests separately from keyword-scored benchmarks.
- [ ] Include failed runs, regressions, and limitations; do not report only the best score.
- [ ] Promote a checkpoint only after a reproducible comparison and a regression check.

## Minimum experiment record

```text
Experiment name:
Track:
Git commit:
Config:
Base model/checkpoint:
Dataset and split:
Environment (Python / framework / CUDA / GPU):
Command or notebook:
Seed and decoding settings:
Output directory:
Training result:
Evaluation protocol and result:
Known limitations:
```

## Reporting rule

Use **VERIFIED** only for results backed by saved outputs or checkpoint metadata. Use **NOT MEASURED** when the run or evaluation has not been executed, and **UNKNOWN** when evidence is missing. Do not infer a benchmark score from training loss alone.

Related: [Evaluation Guide](EVALUATION_GUIDE.md), [Checkpoint Guide](CHECKPOINT_GUIDE.md), [Colab Training Guide](COLAB_TRAINING.md).
