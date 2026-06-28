# Workflow Rules for This Project

## Effective from 2026-06-28

### Rule: Propose Before Implementing

Before making any code changes to any file in this repository, the assistant MUST:

1. **Find and describe the approach** needed for the change.
2. **Identify the specific file(s)** that would be modified.
3. **Ask the user for explicit agreement** using this exact framing:

   > "Are you agreeing to implement this approach at the particular file?"

4. **Wait for the user's agreement** before editing, writing, deleting, or refactoring any code.

### Level of Detail

- A **high-level explanation** of the approach is sufficient. Do not write long, detailed implementation specs before asking.
- If the change touches **multiple files**, list each file with a **one-line explanation** of what would change there, then ask for agreement.

### Rationale

This rule exists to prevent unwanted or premature changes, especially when the user is time-constrained or when multiple approaches are possible. It ensures the user retains full control over what gets changed and where.

### Scope

This rule applies to:
- Source code files (`.py`, `.json`, `.yaml`, `.toml`, etc.)
- Configuration files
- Test files
- Scripts
- Any file that is part of the deliverable repository

It does **not** apply to:
- Read-only exploration or investigation
- Documentation files inside `.kimchi/docs/` that record decisions or plans
- Running read-only verification commands (e.g., `python --help`, `py_compile` checks)

### Current Status Notes

- The `tests/` directory has been removed by the user. Do not recreate it unless explicitly requested.
- `validate_submission.py` is hackathon-provided and should be left untouched unless the user explicitly agrees otherwise.
- `util/submission.py` should remain as a utility module for `rank.py`.
