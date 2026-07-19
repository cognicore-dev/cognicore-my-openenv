---
name: remember
description: Store a fact, decision, preference, or context into CogniCore persistent memory.
usage: /remember <text> [--category <category>] [--scope user|project]
examples:
  - /remember This project uses FastAPI and PostgreSQL
  - /remember I prefer verbose code comments --scope user
  - /remember Always use pytest for tests in this repo --scope project --category testing
---

Store the given text into CogniCore persistent memory using `cognicore_remember`.

Parse the input as follows:
- The main text is everything before any `--` flags.
- `--category` sets the category (default: "general").
- `--scope` sets the scope: "user" (personal preference, default) or "project" (codebase-specific).

After storing, confirm the memory was saved with its ID and a one-sentence summary.
