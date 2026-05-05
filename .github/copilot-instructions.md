# Copilot Instructions - nelpy

nelpy is a neuroelectrophysiology object model and analysis library for Python.

This top-level file is intentionally compact. Keep always-on guidance here and put task-specific details in modular instruction files.

Repo: https://github.com/nelpy/nelpy
Docs: https://nelpy.github.io/nelpy/

## Always-On Guardrails

- Prefer minimal, targeted changes that match the request.
- Respect public API boundaries and existing chaining behavior.
- Preserve support semantics for time-domain objects when editing core behavior.
- Route edits to the owning module or subpackage before changing behavior.
- Work on a branch based on master, not directly on master.

## Task-Specific Expectations

- Bug fixes: keep changes narrow, add a regression test, and avoid unrelated refactors.
- Refactors: preserve observable behavior unless the task explicitly includes a behavior change.
- Features: keep new public API surface deliberate and consistent with existing naming and property patterns.
- Docs-only changes: prefer documentation validation and avoid code churn unless the docs are wrong because the code is wrong.

## Validation

- Python behavior changes: run targeted tests while iterating, then run `ruff check` and relevant `pytest` coverage before finalizing.
- Formatting-only or small local code edits: run `ruff format` and `ruff check` on the touched area when practical.
- Docs-only changes: validate the touched documentation and run Python tooling only if code examples or package metadata changed.

## Task Routing

Use modular instructions under .github/instructions for focused guidance:

- Core bug-fix practices: bugfix-core-practices.instructions.md
- Testing and pytest workflow: testing-pytest-helpers.instructions.md
- API naming and visibility: api-naming-conventions.instructions.md
- Development setup and tooling: development-setup-env.instructions.md
- Docs maintenance and validation: docs-maintenance.instructions.md
- Package ownership reference: package-structure-reference.instructions.md

## Sources Of Truth

- styleguide.rst for underscore, property, and chaining conventions
- developnotes.md for package organization context
- https://github.com/nelpy/nelpy/issues
