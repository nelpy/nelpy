# Copilot Instructions — nelpy contributor (bug fixes / maintenance)

nelpy is a neuroelectrophysiology object model and analysis library for Python.
This file covers conventions for contributing bug fixes and maintenance patches.

Repo: https://github.com/nelpy/nelpy
Docs: https://nelpy.github.io/nelpy/

---

## Development Setup

```bash
# Fork nelpy on GitHub, then:
git clone https://github.com/<your-username>/nelpy.git
cd nelpy
pip install -e .

# Add upstream remote to stay in sync
git remote add upstream https://github.com/nelpy/nelpy.git
```

### Branch workflow

```bash
# Always branch from master
git checkout master
git pull upstream master
git checkout -b fix/short-description-of-bug

# After iterating until tests pass and everything looks good:
# Push branch and open PR against nelpy/nelpy master
git push origin fix/short-description-of-bug
```

---

## Package Structure

```
nelpy/
    __init__.py
    version.py
    utils.py
    filtering.py
    decoding.py
    core/           # Core data objects (EpochArray, SpikeTrain, AnalogSignal, ...)
    io/             # Data loaders (neuralynx, matlab, trodes, ...)
    plotting/       # Visualization (core.py, scalebar.py, utils.py, ...)
    hmm/            # Hidden Markov model support
    utils_/         # Internal utility helpers
    contrib/        # Contributed / experimental code
    tests/          # Test suite
```

When fixing a bug, identify which subpackage owns the relevant code before editing. Do not move logic between subpackages without strong justification.

---

## Tooling

- **Linting / formatting**: Ruff (`line-length = 88`, `target-version = "py310"`)
- **Import sorting**: Ruff (`extend-select = ["I"]`)
- **Testing**: pytest (`pytest` from repo root)
- **Python**: >= 3.8 supported; target style for py3.10+

Run before submitting a PR:

```bash
ruff check .
ruff format .
pytest
```

---

## API & Naming Conventions (from `styleguide.rst`)

### Leading underscores

- **`_variable` / `_method`**: Internal to the class — not public API. Do not call from outside the class except in other classes that explicitly share implementation knowledge (and even then, avoid it).
- An underscore method may also signal it is **not yet fully tested**. Do not promote it to public API in a bug fix without adding tests.

### No leading underscores

- **Public variables**: Likely implemented as a `@property` with bounds/type checking. When fixing bugs involving these, preserve the property pattern — do not bypass it with direct attribute access.
- **Public methods**: Intended for external use. Safe to call freely.

### Method chaining

Class methods frequently return `self` to support chaining:

```python
result = obj.do_something().do_something_else()
```

When fixing methods, preserve `return self` where it already exists. Do not change a chainable method to return `None` or a new value unless that is the explicit intent of the fix.

---

## Bug Fix Guidelines

### Understand before changing

- Read the full method and its `@property` dependencies before editing.
- Check if the method returns `self` (chainable) — preserve this.
- Check if the attribute being fixed is a public property or a private `_variable`. Fix at the right level.

### Preserve the support concept

nelpy's central design principle is that every data object carries a **support** — an `EpochArray` defining its domain of definition. This distinguishes "no data at time t" from "no record at time t". When fixing bugs involving time-based operations, always verify that `.support` is correctly propagated or updated.

### Do not widen scope

Bug fixes should be minimal and targeted. Do not refactor unrelated code, rename variables, or restructure logic in the same PR. Open a separate issue for improvements noticed along the way.

### Properties and type checking

Public-facing attributes use `@property` with validation. When fixing a property:

```python
@property
def fs(self):
    return self._fs

@fs.setter
def fs(self, val):
    if val <= 0:
        raise ValueError("fs must be positive")
    self._fs = val
```

Do not bypass the setter by writing directly to `self._fs` from outside the class.

---

## Testing

- Tests live in `nelpy/tests/`.
- Every bug fix should include a regression test that fails before the fix and passes after.
- Run the full suite with `pytest` before submitting.
- If `pytest` fails during startup due to unrelated third-party plugin imports in your local environment, disable plugin autoload for that run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest ...` (PowerShell: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; pytest ...`).
- Coverage excludes `contrib/`, `examples/`, `io/`, and some legacy files — focus test additions on `core/` and `utils_/`.

```bash
pytest                        # run all tests
pytest tests/test_core.py     # run a specific file
pytest -x                     # stop on first failure
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; pytest tests/test_core.py  # PowerShell fallback if external plugins break pytest startup
```

---

## Submitting a PR

1. Open an issue first if the bug is non-trivial or the fix touches shared infrastructure.
2. Work on a branch, not directly on `master`.
3. Keep commits focused — one bug per PR where possible.
4. Ensure `ruff check .`, `ruff format .`, and `pytest` all pass cleanly.

---

## Reference

- Style guide: `styleguide.rst` in repo root
- Dev notes: `developnotes.md` in repo root
- Issues: https://github.com/nelpy/nelpy/issues