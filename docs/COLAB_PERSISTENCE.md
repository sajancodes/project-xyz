# Google Colab Persistence Checklist

Colab runtimes can disconnect or be reclaimed. Files stored only on the temporary runtime disk may disappear.

## Before training

- [ ] Mount the intended persistent storage and confirm the mount succeeded.
- [ ] Write a small test file, read it back, and remove the test file.
- [ ] Confirm the configured output path points to persistent storage.
- [ ] Record where logs, checkpoints, configs, and evaluation outputs will be saved.
- [ ] Ensure the selected checkpoint is not just a smoke-test artifact.

## During training

- [ ] Check that checkpoints are actually appearing at the expected path.
- [ ] Confirm logs include step and loss information.
- [ ] Avoid relying on a final save that may never execute if the runtime stops.
- [ ] Keep a copy of the exact configuration and Git commit SHA.

## After a disconnect

- [ ] Reconnect and verify the persistent mount.
- [ ] Inspect the latest saved checkpoint and its metadata.
- [ ] Compare the last saved step with the log.
- [ ] Restore compatible dependencies and configuration.
- [ ] Resume only after confirming the correct experiment.

Persistent storage reduces the risk of losing artifacts but does not guarantee a run will finish. Follow the [Colab Training Guide](COLAB_TRAINING.md) for the repository-specific workflow.