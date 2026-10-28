# Train/Test Leakage Audit

Use this audit before treating a benchmark score as evidence of generalization.

## Inspect the data path

1. Trace every file and generator read by the training pipeline.
2. Confirm evaluation files are not imported into training or preprocessing.
3. Check cached artifacts and notebook variables for stale benchmark content.
4. Review whether synthetic training examples reuse held-out facts, names, templates, or answers.
5. Check exact duplicates and, when feasible, normalized or near-duplicate text.

## Inspect the evaluation process

- Confirm the checkpoint was not selected repeatedly against the final test set.
- Confirm benchmark-specific prompt edits were not made after reviewing test outputs without disclosure.
- Separate validation results used for development from final held-out results.
- Keep raw prompts and outputs so unexpected overlaps can be investigated.

## Findings

Record the dataset versions, files inspected, matching method, number of matches, false-positive handling, and unresolved risks. A zero-match result only applies to the method and data actually checked.

If possible leakage is found, label the affected score as potentially contaminated until investigated. Do not silently keep reporting it as an unbiased test result.