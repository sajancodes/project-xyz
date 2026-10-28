# Tokenizer and Prompt-Formatting Checks

Tokenizer and formatting mismatches can make an otherwise valid training run learn the wrong boundaries or behave differently at evaluation time.

## Before training

- Confirm the tokenizer belongs to the intended base model.
- Inspect special tokens and chat-template behavior.
- Decode a few tokenized examples and compare them with the source text.
- Check truncation and padding behavior at the configured sequence length.
- Verify that user and assistant segments are marked correctly for the intended loss policy.
- Confirm end-of-turn and end-of-sequence tokens are used consistently.

## Before evaluation

- Use the documented prompt wrapper for the benchmark.
- Record whether the prompt is plain text or chat-formatted.
- Confirm that stop tokens and generation boundaries are handled consistently.
- Compare raw prompts, token IDs where useful, and decoded outputs for suspicious failures.

A model trained on one wrapper may score differently on another. Report the wrapper explicitly instead of attributing all changes to model quality.