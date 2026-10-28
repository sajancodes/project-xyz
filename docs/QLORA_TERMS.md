# QLoRA Terminology Reference

This glossary summarizes common terms used in the repository's low-memory fine-tuning workflow.

- **Base model:** pretrained model whose weights provide the starting behavior.
- **Quantization:** representing weights with fewer bits to reduce memory use, with trade-offs depending on the method.
- **NF4:** a 4-bit data type designed for quantizing normally distributed weights, commonly used in QLoRA workflows.
- **LoRA:** Low-Rank Adaptation; trains small adapter matrices instead of updating all base-model weights.
- **Rank (`r`):** dimension of the low-rank adapter update; larger values generally add capacity and trainable parameters.
- **Alpha:** scaling factor used by the LoRA implementation to scale adapter updates.
- **Dropout:** regularization applied to adapter paths when enabled.
- **Micro-batch:** number of examples processed in one forward/backward step.
- **Gradient accumulation:** combines gradients from multiple micro-batches before an optimizer update.
- **Effective batch size:** approximately micro-batch × accumulation steps × number of devices, when the training setup uses those factors.
- **Assistant-only loss:** loss is computed on assistant target tokens while user/context tokens are masked, if the dataset and trainer implement that policy correctly.
- **Checkpoint:** saved model and possibly optimizer/scheduler/RNG state.
- **Smoke test:** small run that checks whether the pipeline functions; it is not a quality benchmark.

Always consult the actual configuration and code before assuming a term is implemented in a particular way.