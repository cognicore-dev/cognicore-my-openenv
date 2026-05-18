"""
CogniCore LangChain Integration — wraps CogniCore cognition as LangChain tools.

Usage:
    from cognicore.integrations.langchain import CogniCoreTool, CogniCoreMemoryTool

    # Add CogniCore repair to any LangChain agent
    tools = [CogniCoreTool(), CogniCoreMemoryTool()]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
"""
from typing import Optional, Type
from cognicore.runtime import CogniCoreRuntime, RuntimeConfig
from cognicore.research.persistent_store import PersistentCognitionStore

try:
    from langchain.tools import BaseTool
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Stub for when langchain isn't installed
    class BaseTool:
        name = ""
        description = ""
        def _run(self, *a, **kw): pass
    class BaseModel:
        pass
    class Field:
        def __init__(self, *a, **kw): pass


class RepairInput(BaseModel if LANGCHAIN_AVAILABLE else object):
    """Input for the CogniCore repair tool."""
    if LANGCHAIN_AVAILABLE:
        buggy_code: str = Field(description="The buggy Python code to fix")
        error_message: str = Field(default="", description="Error message or test output")
        category: str = Field(default="general", description="Bug category for memory retrieval")


class CogniCoreTool(BaseTool):
    """LangChain tool that uses CogniCore persistent memory for code repair."""
    name: str = "cognicore_repair"
    description: str = (
        "Use CogniCore's persistent cognitive memory to help repair buggy Python code. "
        "Retrieves past repair strategies, reflections, and failed approaches from memory. "
        "Input: buggy code + error message. Output: repair context and suggestions."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._runtime = CogniCoreRuntime(config=RuntimeConfig(
            reflection_min_samples=1, reflection_failure_threshold=1, memory_top_k=5,
        ), name="langchain-cognicore")
        self._persistent = PersistentCognitionStore()

    def _run(self, buggy_code: str, error_message: str = "", category: str = "general") -> str:
        # Build context from CogniCore memory
        mem_ctx = self._runtime._build_context(category)
        insights = self._persistent.get_cross_session_insights(category)

        parts = ["## CogniCore Repair Context\n"]

        # Memory hints
        if mem_ctx.get("memory"):
            fails = [e for e in mem_ctx["memory"] if not e.get("correct")]
            successes = [e for e in mem_ctx["memory"] if e.get("correct")]
            if fails:
                parts.append(f"### Past Failures ({len(fails)} in '{category}')")
                for f in fails[-3:]:
                    parts.append(f"- {f.get('predicted', 'unknown')}")
            if successes:
                parts.append(f"\n### Past Successes ({len(successes)})")
                for s in successes[-3:]:
                    parts.append(f"- {s.get('predicted', 'unknown')}")

        # Reflection
        if mem_ctx.get("reflection_hint"):
            parts.append(f"\n### Reflection\n{mem_ctx['reflection_hint']}")

        # Cross-session insights
        failed_tactics = insights.get("failed_tactics", {})
        if failed_tactics:
            parts.append("\n### Disabled Tactics (historically failed)")
            for tactic, cnt in failed_tactics.items():
                parts.append(f"- '{tactic}' failed {cnt}x")

        successful_tactics = insights.get("successful_tactics", {})
        if successful_tactics:
            parts.append("\n### Recommended Tactics")
            for tactic, cnt in successful_tactics.items():
                parts.append(f"- '{tactic}' succeeded {cnt}x")

        parts.append(f"\n### Bug Code\n```python\n{buggy_code[:500]}\n```")
        if error_message:
            parts.append(f"\n### Error\n{error_message[:200]}")

        return "\n".join(parts)

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class CogniCoreMemoryTool(BaseTool):
    """LangChain tool for storing repair outcomes in CogniCore memory."""
    name: str = "cognicore_memory"
    description: str = (
        "Store a code repair outcome in CogniCore's persistent memory. "
        "Call this after attempting a fix to record whether it worked. "
        "Input: category, tactic used, success (true/false), error if failed."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._runtime = CogniCoreRuntime(config=RuntimeConfig(
            reflection_min_samples=1, reflection_failure_threshold=1,
        ), name="langchain-memory")
        self._persistent = PersistentCognitionStore()

    def _run(self, category: str = "general", tactic: str = "unknown",
             success: bool = False, error: str = "") -> str:
        # Store in-session
        self._runtime.memory.store({
            "category": category, "correct": success,
            "predicted": f"tactic:{tactic} {'PASS' if success else 'FAIL'}"
        })
        # Store cross-session
        self._persistent.store_strategy(category, tactic, success)

        status = "SUCCESS" if success else "FAILURE"
        return f"Stored {status}: tactic='{tactic}' for category='{category}'"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


# Convenience function
def get_cognicore_tools():
    """Get all CogniCore LangChain tools."""
    return [CogniCoreTool(), CogniCoreMemoryTool()]
