# Configuration Management

Treat each training configuration as an experiment input, not as an informal collection of notebook edits.

## Rules

- Keep the effective values for every run in a saved configuration file or report.
- Record defaults that affect behavior, including sequence length, batch size, gradient accumulation, learning rate, optimizer, scheduler, precision, quantization, LoRA settings, seed, and output path.
- Avoid changing multiple unrelated parameters in a single comparison.
- Save a copy of the exact configuration beside the logs and checkpoint metadata.
- Record environment variables that affect execution, but never commit API keys, tokens, or other secrets.
- When resuming, verify that the saved optimizer and scheduler state is compatible with the intended configuration.

## Comparing configurations

Use a diff to identify exactly what changed between experiments. State the hypothesis for each change and define the metric that would support or reject it.

Do not assume two runs are comparable because their folder names look similar. Confirm the actual configuration and evaluation protocol.

Related: [Reproducibility Checklist](REPRODUCIBILITY_CHECKLIST.md), [Experiment Report Template](EXPERIMENT_REPORT_TEMPLATE.md).