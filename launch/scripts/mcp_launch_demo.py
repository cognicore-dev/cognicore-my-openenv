"""
CogniCore MCP launch demo.

Shows the minimal loop:
1. Store a failure.
2. Retrieve the failure.
3. Generate a reflection.
4. Avoid the same failure.

Run:
    python -B launch/scripts/mcp_launch_demo.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile

import cognicore.mcp.server as mcp_server


async def call_tool(server, name: str, arguments: dict) -> str:
    """Call a FastMCP tool and return its text result."""
    result = await server.call_tool(name, arguments)
    if isinstance(result, tuple) and len(result) == 2:
        _, structured = result
        return structured["result"]
    return result[0].text


def reset_server_state(data_dir: str) -> None:
    """Reset module singletons so the demo is deterministic."""
    os.environ["COGNICORE_DATA_DIR"] = data_dir
    mcp_server._memory = None
    mcp_server._semantic_memory = None
    mcp_server._reflection = None
    mcp_server._detector = None
    mcp_server._data_dir = None
    for key in mcp_server._stats:
        mcp_server._stats[key] = 0


async def main() -> None:
    logging.getLogger("cognicore.mcp").setLevel(logging.WARNING)

    with tempfile.TemporaryDirectory(prefix="cognicore-mcp-demo-") as data_dir:
        reset_server_state(data_dir)
        server = mcp_server.create_mcp_server()

        category = "launch_release"
        failed_action = "skip pre-release tests"
        safer_action = "run pre-release tests"

        print("CogniCore MCP Launch Demo")
        print("=" * 28)

        print("\n1. Store a failure")
        print(
            await call_tool(
                server,
                "cognicore_remember",
                {
                    "task": "Ship the launch build",
                    "category": category,
                    "success": False,
                    "action": failed_action,
                    "error": "smoke test failure reached production",
                },
            )
        )

        print("\n2. Retrieve the failure")
        print(
            await call_tool(
                server,
                "cognicore_recall_failures",
                {"category": category, "top_k": 3},
            )
        )

        print("\n3. Generate a reflection")
        print(await call_tool(server, "cognicore_reflect", {"category": category}))

        print("\n4. Avoid the same failure")
        print(f"Proposed action was: {failed_action}")
        print(f"CogniCore memory flags that as a past failure.")
        print(f"New action: {safer_action}")
        print(
            await call_tool(
                server,
                "cognicore_remember",
                {
                    "task": "Ship the launch build safely",
                    "category": category,
                    "success": True,
                    "action": safer_action,
                    "metadata": '{"avoided_repeat_failure": true}',
                },
            )
        )

        print("\nFinal reflection")
        print(await call_tool(server, "cognicore_reflect", {"category": category}))
        print("\nDemo complete: failure stored, recalled, reflected on, and avoided.")


if __name__ == "__main__":
    asyncio.run(main())
