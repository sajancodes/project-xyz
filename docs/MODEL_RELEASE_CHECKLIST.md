# Model Release Checklist

Use this checklist before publishing a model checkpoint, adapter, or benchmark claim.

## Artifact identity

- [ ] Identify base model and exact checkpoint/adapter revision.
- [ ] Record intended use and supported inference format.
- [ ] Verify files load in a clean environment.
- [ ] Include tokenizer and chat-template requirements where needed.
- [ ] Preserve relevant licenses and attribution.

## Evidence

- [ ] Link the evaluation protocol and dataset version.
- [ ] Include baseline and per-category results.
- [ ] Document regressions, known failures, and unmeasured capabilities.
- [ ] Consider benchmark leakage risks.
- [ ] State hardware and inference settings needed to interpret results.

## Safety and privacy

- [ ] Review dataset/source permissions and sensitive-data risks.
- [ ] Do not publish credentials, private logs, or private examples.
- [ ] State intended and out-of-scope uses.
- [ ] Explain known limitations without overstating capability.

Record what evidence was checked and why the artifact is ready. A checkpoint should not be described as generally capable based on one benchmark alone.