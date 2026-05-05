# Agent Instructions - nelpy

nelpy is a neuroelectrophysiology object model and analysis library for Python.

This file mirrors the repo guidance used by Copilot so Codex and other coding
agents can follow the same expectations.

## Always-On Guidance

- Prefer minimal, targeted changes that match the user's request.
- Preserve public API compatibility unless the task explicitly calls for an API
  change.
- Preserve chaining behavior where existing methods return `self`.
- Preserve support semantics for time-domain objects when editing core behavior.
- Route edits to the owning module or subpackage before changing behavior.
- Work on a branch based on `master`, not directly on `master`.

## Task-Specific Guidance

- Bug fixes: keep changes narrow, add a regression test, and avoid unrelated
  refactors.
- Refactors: preserve observable behavior unless the task explicitly includes a
  behavior change.
- Features: keep new public API surface deliberate and consistent with existing
  naming and property patterns.
- Docs-only changes: prefer documentation validation and avoid code churn unless
  the docs are wrong because the code is wrong.

## Validation

- Python behavior changes: run targeted tests while iterating, then `ruff check`
  and relevant `pytest` coverage before finalizing.
- Formatting-only or small local code edits: run `ruff format` and `ruff check`
  on the touched area when practical.
- Docs-only changes: validate the touched documentation and run Python tooling
  only if code examples or package metadata changed.

## Repo References

- `.github/copilot-instructions.md`
- `.github/instructions/`
- `styleguide.rst`
- `developnotes.md`
