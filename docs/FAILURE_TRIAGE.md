# Training Failure Triage

This page is a first-pass checklist for diagnosing common issues in Project XYZ. It does not replace the track-specific training guide.

## 1. GPU is unavailable or PyTorch uses the CPU

**Check**
- Print `torch.cuda.is_available()` and `torch.version.cuda`.
- Run `nvidia-smi` in the active runtime.
- Record the installed PyTorch build and the runtime's GPU type.

**Interpretation**
- A CPU-only PyTorch build can make CUDA unavailable even when a GPU is present.
- A visible GPU in the host environment does not prove that the current Python environment can access it.

**Next step**
Resolve the environment/build mismatch first, restart the runtime if required, then rerun the GPU check and a tiny smoke test. Do not start a long training run until the intended device is confirmed.

## 2. CUDA out-of-memory (OOM)

**Check**
- Record model, quantization mode, sequence length, micro-batch size, gradient accumulation, and peak VRAM.
- Confirm that another process is not occupying the GPU.
- Verify that the model is actually loaded in the intended dtype/quantization mode.

**Next step**
Reduce memory demand in small, controlled steps (for example, lower sequence length or micro-batch size) and rerun the smoke test. Gradient accumulation can preserve effective batch size when the micro-batch is reduced, but it does not make a single forward pass fit if that pass itself exceeds memory.

Do not silently change multiple hyperparameters at once; record each change so the result remains interpretable.

## 3. Colab disconnects or runtime expires

**Check**
- Confirm that checkpoints and logs were written to persistent storage, not only the ephemeral runtime disk.
- Inspect the latest checkpoint's step and metadata.
- Verify that the checkpoint belongs to the intended experiment rather than a smoke test.

**Next step**
Reconnect, restore the documented environment, and follow the track's resume instructions. Confirm the selected checkpoint before resuming. Never assume that a directory name such as `latest` identifies the best checkpoint.

## 4. Training loss falls but evaluation gets worse

**Likely explanations**
- Overfitting or memorization.
- Catastrophic forgetting of general-language behavior.
- Training/evaluation format mismatch, such as chat-wrapped training versus plain prompts.
- Evaluation data leakage or inconsistent scoring settings.

**Next step**
Compare the same held-out benchmark and decoding protocol against the baseline. Inspect per-category scores and representative raw outputs. Do not promote a checkpoint based only on lower training loss.

## 5. Evaluation results disagree across runs

**Check**
- Temperature, top-k/top-p, seed, max-new-tokens, prompt wrapper, and checkpoint path.
- Whether the evaluation uses greedy decoding or sampling.
- Whether the compared outputs were produced by the same code/protocol version.

**Next step**
Use the canonical deterministic evaluation where applicable, save raw outputs, and clearly label stochastic/ad-hoc tests separately. Deterministic decoding reduces sampling variance but does not eliminate differences caused by code, prompts, or environment changes.

## 6. Suspected train/test leakage

**Check**
- Whether benchmark files are imported or read by the training path.
- Whether generated examples can overlap with held-out names, templates, or underlying facts.
- Whether split creation is deterministic and recorded.

**Next step**
Stop interpreting the affected score as an unbiased held-out result until the overlap is investigated. Add an explicit guard or audit and document what it checks.

## Escalation record

When a problem persists, include:
1. Exact command or notebook cell.
2. Full traceback or relevant log excerpt.
3. Git commit and config path.
4. GPU and framework versions.
5. Steps already tried and their outcomes.

Related: [Colab Training Guide](COLAB_TRAINING.md), [Evaluation Guide](EVALUATION_GUIDE.md), [Checkpoint Guide](CHECKPOINT_GUIDE.md).
