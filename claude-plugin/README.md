# CogniCore Memory Plugin

> Give Claude persistent, searchable memory across all your conversations.

---

## What it does

CogniCore Memory connects Claude to a self-hosted cloud memory backend. Every important fact, project decision, preference, or bug fix you discuss can be stored and recalled — across sessions, across projects, across time.

**Without CogniCore:** Claude forgets everything the moment a conversation ends.  
**With CogniCore:** Claude remembers your preferences, your stack, your decisions, and your solutions.

---

## Tools

| Tool | Description |
|---|---|
| `cognicore_remember` | Store a memory with optional category and scope |
| `cognicore_recall` | Search memories using BM25 semantic ranking |
| `cognicore_list` | List recently stored memories |
| `cognicore_forget` | Delete a memory by ID |

---

## Slash Commands

| Command | Description |
|---|---|
| `/remember <text>` | Store a fact or preference |
| `/recall <query>` | Search your memories |
| `/memory-list` | Browse all stored memories |
| `/forget <id>` | Delete a memory by ID |

---

## Quick Start

```
/remember This project uses FastAPI and PostgreSQL --scope project
/remember I prefer TypeScript over JavaScript --scope user
/recall what database does this project use
/memory-list
```

---

## How Search Works

Recall uses **BM25 Okapi** — the same ranking algorithm used by Elasticsearch — to find the most relevant memories for any query. Rare, specific terms (like framework names or bug descriptions) score higher than common words. Natural language queries work well:

- `"who built this"` → finds memories mentioning the person
- `"auth bug fix"` → finds debugging memories about authentication
- `"my preferred stack"` → finds user-scoped preference memories

---

## Scopes

- **`user`** — Personal preferences, global facts about you (default)
- **`project`** — Codebase-specific decisions, architectural choices

---

## Privacy

- All memories are stored in **your own Railway deployment** — no data is sent to CogniCore servers.
- Memories are isolated per user by a deterministic hash of your Claude client identity.
- You can delete any memory at any time with `/forget <id>` or `cognicore_forget`.

---

## Self-Hosting

Deploy your own CogniCore backend in under 5 minutes:

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

Set `JWT_SECRET` in your Railway environment, then update the MCP server URL in `.mcp.json`.

See the [full deployment guide](https://github.com/cognicore-dev/cognicore-my-openenv#deployment).

---

## License

MIT © CogniCore
