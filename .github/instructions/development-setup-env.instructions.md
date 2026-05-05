---
name: development-setup-env
description: "Use when: configuring local development environment, linters, formatters, or test commands"
---

# Development Setup And Tooling

Purpose: Keep environment and tooling setup predictable for lint, format, and test workflows.

- Python support is 3.8+, with style and tooling aligned to modern 3.10+ usage.
- Use Ruff for linting, formatting, and import sorting.
- Keep formatting and lint checks green before final validation.
- Match validation scope to the type of change instead of always running the full suite.
- Prefer targeted pytest runs while iterating, then widen coverage before finalizing behavior changes.

## Common Commands

```bash
pip install -e .
ruff check .
ruff format .
pytest
```

## Validation By Change Type

- Python behavior changes: run `ruff check .` and targeted `pytest` first, then broaden to the relevant suite before finalizing.
- Docs-only changes: validate docs files and examples; run Python tooling only when code snippets or packaging metadata changed.
- Narrow formatting or lint cleanups: run `ruff format` and `ruff check` on the touched area when practical.
