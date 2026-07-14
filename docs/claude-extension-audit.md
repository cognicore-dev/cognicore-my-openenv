# Claude Desktop Extension — Audit Report

## Objectives
1. Build a local-first, production-ready Claude Desktop Extension for CogniCore.
2. Provide a simple mechanism for Claude to remember user preferences, facts, and project constraints persistently.
3. Reuse existing CogniCore infrastructure without duplicating code.

## Architecture Decisions

1. **Storage Engine:** `SQLiteMemoryBackend`
   - Chosen because it provides zero-dependency FTS5 full-text search and robust local file persistence (`memory.db`).
   - `TFIDFMemoryBackend` was evaluated but rejected because it stores large TF-IDF matrices in JSON, which doesn't scale as well for a long-lived extension.
   - `BasicEmbeddingBackend` and `MultiHopMemoryBackend` were rejected as defaults because they require heavy ML dependencies (`numpy`, `sentence-transformers`), which breaks the "simple, one-click" requirement. However, `sentence-transformers` is supported optionally if installed.

2. **Server Interface:** New `cognicore.extension.server`
   - The existing `cognicore.mcp.server` is heavily tailored for agent-runtime cognition (storing execution outcomes, successes/failures, scanning threats). 
   - We created a dedicated extension server with exactly 5 tools designed for factual memory: `cognicore_remember`, `cognicore_recall`, `cognicore_forget`, `cognicore_list`, `cognicore_stats`.
   - The extension server reuses the underlying `SQLiteMemoryBackend` directly.

3. **Packaging Strategy:** `mcpb` Bundle with `uv`
   - Claude Desktop supports `uv` as a server type natively.
   - Our bundle `cognicore-memory.mcpb` simply packages the Python source and a `pyproject.toml`. When Claude Desktop loads the extension, it uses `uv` to transparently manage dependencies (`mcp[cli]`) at runtime without shipping a massive virtual environment.

## Bug Fixes Applied to CogniCore
- **Fixed `memory/__init__.py`:** The original file imported `multihop_backend.py` unconditionally, which imports `numpy`. This caused a fatal crash if `numpy` wasn't installed. We wrapped it in a `try/except ImportError` block.
- **Exported Backends:** Exposed `SQLiteMemoryBackend` and `TFIDFMemoryBackend` in `__init__.py` so they are accessible without deep paths.

## Validation
- Tested `SQLiteMemoryBackend` integration via `test_extension.py`.
- Full backwards compatibility maintained: existing tests pass 610/611 (the single failure is a pre-existing HTML title mismatch in an unrelated studio test).
