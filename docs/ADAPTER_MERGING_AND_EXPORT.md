# Adapter Export Considerations

Adapter training artifacts and standalone model exports are different deliverables.

## Before exporting

- Identify the exact base model revision and adapter checkpoint.
- Verify tokenizer and chat-template requirements.
- Decide whether the consumer expects a separate adapter or merged weights.
- Confirm the supported dtype and quantization format for the target runtime.
- Preserve the original adapter and base-model references.

## Validate the export

- Load the exported artifact in the intended inference environment.
- Run a small fixed prompt suite and compare outputs with the source checkpoint.
- Check generation settings, special tokens, and expected output format.
- Record export tool versions, conversion settings, file hashes where practical, and known limitations.

Merging adapters or converting quantization can change numerical behavior and memory requirements. Do not assume equivalence without testing. Keep the export separate from training checkpoints; an inference export may not contain optimizer state and may not be resumable for training.