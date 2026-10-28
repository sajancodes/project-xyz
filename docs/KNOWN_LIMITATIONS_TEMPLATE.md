# Known Limitations Register

Track limitations explicitly so experiment reports and releases do not overstate model behavior.

| ID | Area | Observation | Evidence | Severity | Status | Next action |
|---|---|---|---|---|---|---|
| L-001 | Example | Replace with a measured issue | Link to test/output | Low/Medium/High | Open | Define follow-up |

## Adding an entry

- Describe an observable behavior, not a vague impression.
- Link the benchmark, raw output, issue, or reproducible test.
- State the scope: model version, prompt format, dataset, and settings.
- Distinguish confirmed problems from hypotheses.
- Record a mitigation or follow-up and update status after retesting.

## Severity guidance

- **High:** blocks intended core use or creates a major reliability concern.
- **Medium:** meaningfully affects a supported task or common category.
- **Low:** limited edge case or minor documentation/usability issue.

Do not remove a limitation because one example passes. Close it only after the original failure has been retested with evidence, noting any remaining scope.