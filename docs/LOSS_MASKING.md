# Assistant-Only Loss Masking Checks

Assistant-only loss masking is intended to compute training loss on assistant target tokens while excluding user prompts or other context tokens. Whether this happens correctly depends on the dataset formatting and trainer implementation.

## Verify the implementation

- Inspect the exact tokenized example and its labels.
- Confirm context tokens are masked using the convention expected by the trainer (often label value `-100`).
- Confirm assistant answer tokens remain supervised.
- Check end-of-turn tokens and multi-message examples.
- Test examples with empty inputs, long prompts, and truncation.
- Run a small forward/backward step and confirm finite loss.

## Common failure modes

Incorrect masks can train on user content, ignore the desired answer, or shift labels relative to tokens. A successful training loop does not prove the mask is semantically correct.

## Evidence

Save a small, sanitized example showing tokens and labels, the relevant configuration, and the test result. Re-check after changing chat templates, tokenizers, collators, or trainer versions.