# Run preflight

Before a long run, verify the Git SHA, Python environment, CUDA availability, model ID, dataset revision, output directory, and free disk space.

- Print library versions and device name.
- Confirm the expected model loads on the intended device.
- Check that the output path is writable.
- Run one tiny batch before scaling up.

Record every check as passed, failed, or not run. Do not start long training with an unresolved critical failure.
