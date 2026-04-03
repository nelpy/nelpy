---
name: bugfix-core-practices
description: "Use when: fixing bugs in core data objects, support propagation, properties, or chainable methods"
applyTo: "nelpy/core/*.py"
---

# Core Bug Fix Practices

Purpose: Keep core-object bug fixes behavior-safe, minimal, and support-aware.

- Keep fixes narrow and local to the bug. Do not refactor unrelated code in the same change.
- Read the full method and related properties before editing behavior.
- Preserve chainability patterns where methods currently return self.
- Respect public property boundaries: do not bypass public validation through private attributes from outside a class.
- Preserve nelpy's support concept: operations over time should keep support semantics correct.
- Avoid moving logic across subpackages unless the bug cannot be fixed in place.
