---
name: testing-pytest-helpers
description: "Use when: writing regression tests, debugging pytest failures, or validating bug fixes"
applyTo: "tests/test_*.py"
---

# Testing And Pytest Helpers

Purpose: Guide regression-focused test updates and reliable pytest debugging workflow.

- Every bug fix should include a regression test that fails before and passes after the fix.
- Prefer focused tests near the affected object or behavior.
- Run targeted tests while iterating, then run the full suite before finalizing.
- If unrelated plugin autoload breaks pytest startup, use environment-based plugin disable for that run.
- Coverage focus for fixes is primarily core and utils_ paths.

## Useful Commands

```bash
pytest
pytest tests/test_file.py
pytest -x
```

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; pytest tests/test_file.py
```
