# Checkpoint Naming Guidance

Clear names help distinguish production candidates, resumable training states, and disposable smoke tests.

## Suggested pattern

Use a run identifier and an explicit step or purpose, for example:

```text
<experiment>-step-<number>
<experiment>-latest
<experiment>-smoke-test
```

Adapt the names to the repository's existing conventions rather than renaming historical artifacts blindly.

## Rules

- Never use a smoke-test checkpoint as a model-quality baseline.
- Do not imply that `latest` means best; it usually means most recently written.
- Preserve the original best checkpoint and its backup according to the [Checkpoint Guide](CHECKPOINT_GUIDE.md).
- Record parent checkpoint, training type, step, config, and output directory.
- Keep checkpoints from separate experiments in separate directories.
- Verify embedded metadata before resuming or evaluating.

Names are useful labels but are not proof. Check metadata and file integrity before relying on a checkpoint.