---
name: development-setup-env
description: "Use when: configuring local development environment, linters, formatters, or test commands"
---

# Development Setup And Tooling

Purpose: Keep environment and tooling setup predictable for lint, format, and test workflows.

- Python support is 3.8+, with style and tooling aligned to modern 3.10+ usage.
- Use Ruff for linting, formatting, and import sorting.
- Keep formatting and lint checks green before final validation.

## Common Commands

```bash
pip install -e .
ruff check .
ruff format .
pytest
```
