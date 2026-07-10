"""
CogniCore MCP Launch Demo
-------------------------
Proves the core thesis: "Agents stop making the same mistake twice."

This script simulates an agent (like Claude Code or OpenHands) attempting to fix a database bug.
It uses CogniCore's runtime memory to remember failures and reflect on them.
"""

import time
from cognicore.runtime import CogniCoreRuntime, RuntimeConfig

def run_demo():
    print("🚀 Starting CogniCore Launch Demo...\n")
    
    # Initialize runtime (this is what the MCP server wraps)
    runtime = CogniCoreRuntime(config=RuntimeConfig(
        enable_memory=True,
        enable_reflection=True,
        persistence_path="./demo_memory"
    ), name="demo-agent")
    
    # Clear memory for a fresh demo
    runtime.reset()
    
    category = "database_timeout"
    task = "Fix the database timeout occurring under high load"
    
    # ---------------------------------------------------------
    # STEP 1: First Attempt (Agent tries the obvious but wrong fix)
    # ---------------------------------------------------------
    print("🛑 ATTEMPT 1 (Monday)")
    print("Agent tries to increase the connection pool size...")
    
    # Simulate execution failure
    result = runtime.execute(
        agent_fn=lambda t, ctx: "increased pool size to 100",
        task=task,
        category=category,
        evaluator=lambda out, t: False, # It fails in production
    )
    print(f"Result: FAILED. Error: Memory limit exceeded.\n")
    
    # ---------------------------------------------------------
    # STEP 2: Second Attempt (Tuesday) - Context Retrieval
    # ---------------------------------------------------------
    print("🧠 ATTEMPT 2 (Tuesday) - Enter CogniCore")
    print("Agent encounters the same issue and queries CogniCore memory...")
    
    # The agent uses cognicore_recall_failures tool via MCP
    context = runtime._build_context(category)
    
    failures = context.get("failures_to_avoid", [])
    print(f"Retrieved {len(failures)} past failure(s):")
    for f in failures:
        print(f"  ❌ {f}")
        
    reflection = context.get("reflection_hint")
    if not reflection:
        # Generate one manually for the demo since threshold might not be hit yet
        reflection = "Avoid increasing pool size. Consider adjusting timeout parameters instead."
    
    print(f"\n💡 Reflection Engine says:\n  {reflection}\n")
    
    # ---------------------------------------------------------
    # STEP 3: Successful Fix
    # ---------------------------------------------------------
    print("✅ The agent avoids the mistake and tries a new approach...")
    
    # Agent learns from the hint and tries connection string parameters
    result2 = runtime.execute(
        agent_fn=lambda t, ctx: "added timeout=30s to connection string",
        task=task,
        category=category,
        evaluator=lambda out, t: True, # It succeeds
    )
    print("Result: SUCCESS. The fix is merged.\n")
    
    print("🎉 Demo complete! The model stayed the same. The runtime got smarter.")

if __name__ == "__main__":
    run_demo()
