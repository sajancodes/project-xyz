# Safe Training Resume Procedure

Resume only after confirming which experiment and state you intend to continue.

## Before resuming

1. Identify the run and its last known completed step from logs and checkpoint metadata.
2. Confirm the checkpoint is a resumable training checkpoint, not an inference-only export or smoke-test artifact.
3. Check whether optimizer, scheduler, RNG state, and data position are available.
4. Restore a compatible environment and configuration.
5. Verify persistent storage and free disk space.
6. Run a short validation or smoke test before committing to a long session.

## What may differ

A restart may change data order, random state, learning-rate schedule, or effective training duration if state was not saved. Document missing state instead of claiming an exact continuation.

## After resuming

Check the first logged step and learning rate, confirm loss is finite, and ensure output files are written to the intended experiment directory. Do not overwrite the best checkpoint or mix unrelated experiments in one output folder.

For interrupted Colab runs, follow the [Colab Persistence Guide](COLAB_PERSISTENCE.md) and record the interruption in the experiment report.