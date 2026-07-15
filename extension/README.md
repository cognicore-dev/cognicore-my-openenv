# CogniCore Memory for Claude Desktop

Give Claude persistent, long-term memory across all your chats and projects using [CogniCore](https://github.com/cognicore-dev/cognicore-my-openenv).

This extension provides local-first, privacy-respecting memory via SQLite.

## Installation

This is a Model Context Protocol (MCP) server. You can install it in Claude Desktop using `uv`.

1. Open your `claude_desktop_config.json`:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Add the server configuration:

```json
{
  "mcpServers": {
    "cognicore-memory": {
      "command": "uvx",
      "args": [
        "cognicore-env[extension]",
        "cognicore-extension"
      ]
    }
  }
}
```

3. Restart Claude Desktop.

## Usage

Simply tell Claude what you want it to remember:
* "Remember that my project uses FastAPI and PostgreSQL."
* "My preference is always to use pytest over unittest."
* "Keep in mind that the deployment server IP is 192.168.1.100."

In future conversations, Claude will automatically search this memory to provide better context.

## Privacy & Storage

All memories are stored locally in an SQLite database at:
* `~/.cognicore/extension/memory.db`

No data is sent to external APIs or cloud services unless you explicitly configure the semantic search provider.

## Semantic Search (Optional)

By default, the extension uses fast, deterministic lexical search (FTS5). To enable semantic search using local AI models:

1. Install with the `embeddings` extra:
   ```json
   "args": [
     "cognicore-env[extension,embeddings]",
     "cognicore-extension"
   ]
   ```
2. Add the environment variable to your config:
   ```json
   "env": {
     "COGNICORE_USE_SEMANTIC": "1"
   }
   ```
