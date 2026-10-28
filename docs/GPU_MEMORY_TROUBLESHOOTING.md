# GPU Memory Troubleshooting

Use measured device and process information to diagnose memory problems instead of guessing from the model's parameter count.

## Capture first

Record the GPU model, available VRAM, model dtype/quantization, sequence length, micro-batch size, gradient accumulation, and any offload settings. Check whether other processes are using the GPU.

## Understand the main contributors

- Model weights and quantization metadata.
- Activations, which depend strongly on sequence length and batch size.
- Gradients and optimizer states for trainable parameters.
- Temporary workspaces and framework caching.
- Memory reserved for generation or evaluation.

Quantized weights reduce weight memory but do not eliminate activation and runtime memory costs.

## Change one variable at a time

Try a shorter sequence or smaller micro-batch, then rerun the same smoke test. Gradient accumulation can maintain effective batch size after reducing the micro-batch, but it does not shrink a single forward pass enough if that pass alone exceeds memory.

Use gradient checkpointing or supported memory-saving options only after confirming compatibility. Document every change in the run configuration.

## Stop conditions

Do not launch a long run while the smoke test repeatedly OOMs, the model is unexpectedly on CPU, or the effective quantization mode is uncertain. See [Failure Triage](FAILURE_TRIAGE.md).