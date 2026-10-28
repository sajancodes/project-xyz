# Pull Request Checklist

Use this checklist to keep changes reviewable and avoid accidental experiment or artifact damage.

## Scope

- [ ] The PR has one clear purpose.
- [ ] Unrelated refactors and generated artifacts are excluded.
- [ ] Documentation links and paths are correct.
- [ ] Configuration changes are documented.

## Validation

- [ ] Relevant tests or checks were run.
- [ ] Manual validation steps are listed when automated checks are unavailable.
- [ ] Documentation claims match the current code/configuration.
- [ ] Untested behavior is called out explicitly.

## Data and artifacts

- [ ] No secrets, credentials, or private data are included.
- [ ] Large checkpoints or generated datasets are not added unintentionally.
- [ ] Known-good checkpoints and backups remain untouched.
- [ ] Dataset and benchmark leakage risks were considered where relevant.

## Reporting

Describe what changed, why it changed, what was validated, and known limitations. Do not claim tests passed unless they were actually run. For model-quality changes, link the baseline comparison and raw evaluation evidence.