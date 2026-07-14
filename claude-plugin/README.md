# CogniCore Memory Plugin

CogniCore gives Claude persistent memory for important facts, preferences, constraints, project decisions, and previous solutions, with robust user-level isolation and project-scoped context.

## Installation
1. Start the Remote MCP Server: `uvicorn cognicore.extension.remote:app --host 0.0.0.0 --port 8000`
2. Get your JWT token for authentication.
3. Replace the `TEST_JWT_TOKEN_REPLACE_ME` in `.mcp.json` with your real token.
4. Add the marketplace to Claude Web and install the plugin!

## Usage
Claude will automatically remember your preferences and project details. You can also explicitly say:
- "Remember that I prefer Pytest."
- "What database does this project use?"
- "Forget the previous memory about using JavaScript."
