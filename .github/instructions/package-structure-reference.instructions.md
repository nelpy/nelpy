---
name: package-structure-reference
description: "Use when: identifying which module or subpackage owns logic before making a change"
---

# Package Ownership Reference

Purpose: Route edits to the owning subpackage before making behavior changes.

- nelpy/core: core object model and array abstractions.
- nelpy/analysis: replay, ergodic, and higher-level analysis helpers.
- nelpy/auxiliary: helper classes used across public objects.
- nelpy top-level modules: user-facing algorithm and utility entry points such as `decoding.py`, `estimators.py`, `filtering.py`, `preprocessing.py`, and `utils.py`.
- nelpy/io: data loaders and importers.
- nelpy/plotting: plotting and visualization utilities.
- nelpy/utils_ and nelpy/utils: utility helpers.
- docs, README.md, and mkdocs.yml: documentation and site structure.
- tests: regression and behavior verification.

Choose the owning subpackage before editing, and avoid cross-subpackage moves unless required for correctness.
