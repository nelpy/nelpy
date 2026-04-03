# Copilot Instructions - nelpy

nelpy is a neuroelectrophysiology object model and analysis library for Python.

This top-level file is intentionally compact. Keep always-on guidance here and put task-specific details in modular instruction files.

Repo: https://github.com/nelpy/nelpy
Docs: https://nelpy.github.io/nelpy/

## Always-On Guardrails

- Keep bug fixes minimal and targeted; do not refactor unrelated behavior in the same change.
- Respect public API boundaries and existing chaining behavior.
- Preserve support semantics for time-domain objects when fixing core behavior.
- Work on a branch based on master, not directly on master.
- Before finalizing, run ruff check, ruff format, and pytest from repo root.

## Task Routing

Use modular instructions under .github/instructions for focused guidance:

- Core bug-fix practices: bugfix-core-practices.instructions.md
- Testing and pytest workflow: testing-pytest-helpers.instructions.md
- API naming and visibility: api-naming-conventions.instructions.md
- Development setup and tooling: development-setup-env.instructions.md
- Package ownership reference: package-structure-reference.instructions.md

## Sources Of Truth

- styleguide.rst
- developnotes.md
- https://github.com/nelpy/nelpy/issues