---
name: api-naming-conventions
description: "Use when: deciding public vs private API, naming methods or attributes, or preserving property contracts"
---

# API And Naming Conventions

Purpose: Preserve public/private boundaries and stable property contracts in API-facing code.

- Underscore-prefixed variables and methods are internal implementation details.
- Public attributes should remain property-based where validation or invariants are enforced.
- Do not expose internal helpers as public API in a bug fix unless explicitly requested.
- Public methods are external API; preserve compatibility for existing behavior.
- Preserve method chaining behavior where existing methods return self.
