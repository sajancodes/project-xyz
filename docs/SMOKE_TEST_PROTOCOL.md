# Pre-Training Smoke Test Protocol

A smoke test checks whether the training path works before expensive compute is committed.

## Minimum checks

1. Confirm the expected GPU is visible to the active Python environment.
2. Load the intended tokenizer and model configuration.
3. Inspect trainable parameter counts after adapter setup.
4. Run a tiny batch through forward and backward passes.
5. Confirm loss is finite and gradients reach intended trainable parameters.
6. Save a test checkpoint to a disposable directory and load it again.
7. Confirm the run does not exceed the expected memory budget.
8. Run dataset schema and formatting checks.

## Keep it small

Use a clearly labeled smoke-test output path and a small number of examples/steps. Never promote its checkpoint based on the fact that it loaded or produced finite loss.

## Record the outcome

Save environment versions, model ID, config, device, loss, peak memory if available, and any warnings. Mark each check passed, failed, or not run.

If any critical check fails, diagnose it before starting the full training job. See [Failure Triage](FAILURE_TRIAGE.md) for common failure modes.