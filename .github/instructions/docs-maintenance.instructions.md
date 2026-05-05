---
name: docs-maintenance
description: "Use when: editing docs, README content, mkdocs configuration, or doc-adjacent examples"
---

# Docs Maintenance

Purpose: Keep documentation changes accurate, scoped, and aligned with the codebase.

- Prefer fixing documentation in place instead of changing code unless the docs reveal a real code defect.
- Keep examples consistent with the public API and current branch conventions.
- When documenting behavior, defer to the code and tests over stale prose.
- Avoid introducing new terminology for existing concepts such as support, epochs, or chaining unless the change intentionally clarifies them.
- For docs-only edits, validate the touched pages and examples; do not force unrelated Python test runs.
