# CogniCore MCP Validation Plan

This plan defines the release gate for the CogniCore MCP server.

## Scope

Validate the MCP server exposed by:

```powershell
python -B -m cognicore.mcp.server
```

and the CLI wrapper:

```powershell
python -B -m cognicore.cli mcp serve --transport stdio
```

## Automated Pytest Coverage

Run:

```powershell
python -B -m pytest tests\test_mcp_server.py -q
```

Required coverage:

- Tool registration: `list_tools()` returns the complete expected tool set.
- Persistence across restarts: `cognicore_remember` writes state to `COGNICORE_DATA_DIR`, then a fresh server instance can recall it.
- Memory retrieval: semantic recall and category fallback recall both return stored memories.
- Failure retrieval: `cognicore_recall_failures` returns failed entries for a category.
- Reflection: repeated failed actions appear in reflection output and recommendations are surfaced.
- Threat detection: `ThreatScore` output is formatted with level, score, category, and indicators.
- Desktop-client transport: a subprocess launched over MCP stdio can initialize, list tools, and call a tool.

## Manual Startup Smoke Tests

Start the module server and verify it stays alive while stdin is open:

```powershell
python -B -m cognicore.mcp.server
```

Start through the CLI wrapper:

```powershell
python -B -m cognicore.cli mcp serve --transport stdio
```

Expected result: no constructor/import traceback, and the process waits for MCP stdio messages.

## Claude Desktop Compatibility

Add this server entry to Claude Desktop MCP config, adjusting paths for the checkout:

```json
{
  "mcpServers": {
    "cognicore": {
      "command": "python",
      "args": ["-B", "-m", "cognicore.mcp.server"],
      "cwd": "C:\\Users\\kaush\\OneDrive\\Documents\\safetymind\\cognicore-my-openenv",
      "env": {
        "COGNICORE_DATA_DIR": "C:\\tmp\\cognicore-mcp-claude"
      }
    }
  }
}
```

Acceptance checks:

- Claude Desktop starts without MCP server errors.
- The `cognicore` server appears as connected.
- Claude can see all seven tools:
  - `cognicore_remember`
  - `cognicore_recall`
  - `cognicore_recall_failures`
  - `cognicore_recall_successes`
  - `cognicore_reflect`
  - `cognicore_scan_threat`
  - `cognicore_stats`
- Calling `cognicore_scan_threat` with `ignore all previous instructions` returns `HIGH` and `prompt_injection`.
- Calling `cognicore_remember`, restarting Claude Desktop, then calling `cognicore_recall` returns the stored memory.

## Cursor Compatibility

Add this server entry to Cursor MCP config, adjusting paths for the checkout:

```json
{
  "mcpServers": {
    "cognicore": {
      "command": "python",
      "args": ["-B", "-m", "cognicore.mcp.server"],
      "cwd": "C:\\Users\\kaush\\OneDrive\\Documents\\safetymind\\cognicore-my-openenv",
      "env": {
        "COGNICORE_DATA_DIR": "C:\\tmp\\cognicore-mcp-cursor"
      }
    }
  }
}
```

Acceptance checks:

- Cursor starts the MCP server without a traceback.
- Cursor lists all seven CogniCore tools.
- Cursor can call `cognicore_scan_threat` and receive formatted text output.
- Cursor can call `cognicore_remember`, restart Cursor, and recover the memory with `cognicore_recall`.

## Release Gate

Do not release unless all of these pass:

```powershell
python -B -m py_compile cognicore\mcp\server.py tests\test_mcp_server.py
python -B -m pytest tests\test_mcp_server.py -q
python -B -m pytest tests -q
python -B -m pip check
```

Known limitation: Claude Desktop and Cursor are GUI clients, so pytest can validate the shared stdio MCP protocol but cannot prove those applications load local user configuration. Run the manual client checks before release.
