# Checkpoint Integrity Verification

Checkpoint files are expensive artifacts. Verify that the intended file exists and is intact before evaluation, promotion, or cleanup.

## Before a run

- Confirm the parent checkpoint path and expected experiment.
- Record a cryptographic hash for important immutable artifacts where practical.
- Verify that the best checkpoint and its backup match if the project policy requires identical copies.
- Ensure there is enough disk space for the new run and its logs.

## After saving

- Confirm the file exists and has a plausible non-zero size.
- Load it using the expected loader in a small validation step.
- Inspect metadata such as training step, loss, model family, and parent.
- Recompute and compare hashes for copied artifacts when integrity matters.
- Keep the save log and checkpoint path in the experiment report.

A successful save message alone does not prove the checkpoint can be loaded or resumed. Test loading before deleting temporary files or starting a long run from it.

Never overwrite the known-good checkpoint while testing a new save path.