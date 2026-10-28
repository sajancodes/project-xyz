# Dataset Validation Checklist

Run these checks before training on a new dataset version or materially changed generator.

## Structural checks

- [ ] Confirm the expected file format and encoding.
- [ ] Validate required fields and field types.
- [ ] Count empty, malformed, truncated, and duplicate records.
- [ ] Confirm that each example has a valid target and that prompt/answer boundaries are preserved.
- [ ] Check for unexpectedly long examples and tokenizer truncation.

## Content checks

- [ ] Review a random sample and edge cases manually.
- [ ] Look for repeated templates, placeholder text, malformed Unicode, and accidental markup.
- [ ] Check for contradictory targets or answers copied into prompts.
- [ ] Verify source/license constraints and remove data that should not be used.
- [ ] Review for private or sensitive information before committing or sharing data.

## Split checks

- [ ] Confirm train/validation/test split counts.
- [ ] Check exact and near-duplicate overlap where practical.
- [ ] Ensure benchmark questions or answers were not used to construct training examples.
- [ ] Record the split seed or deterministic assignment rule.

## Record the result

Save the dataset version, command/script, counts, warnings, sample-review notes, and Git SHA. A passing schema test alone does not establish semantic quality or absence of leakage.

Related: [Data Guide](DATA_GUIDE.md), [Dataset Card Template](DATASET_CARD_TEMPLATE.md).