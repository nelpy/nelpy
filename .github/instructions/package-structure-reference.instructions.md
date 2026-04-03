---
name: package-structure-reference
description: "Use when: identifying which subpackage owns logic before making a bug fix"
---

# Package Ownership Reference

Purpose: Route edits to the owning subpackage before making behavior changes.

- nelpy/core: core object model and array abstractions.
- nelpy/io: data loaders and importers.
- nelpy/plotting: plotting and visualization utilities.
- nelpy/utils_ and nelpy/utils: utility helpers.
- tests: regression and behavior verification.

Choose the owning subpackage before editing, and avoid cross-subpackage moves unless required for correctness.
