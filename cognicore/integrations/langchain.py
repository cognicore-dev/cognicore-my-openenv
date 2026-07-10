"""
CogniCore LangChain Integration — wraps CogniCore cognition as LangChain tools.

Provides two categories of tools:

**Code-Repair tools** (original):
    - ``CogniCoreTool`` — persistent-memory-backed code repair context
    - ``CogniCoreMemoryTool`` — store repair outcomes

**General-purpose cognition tools** (new):
    - ``CogniCoreRecallTool`` — semantic memory recall for any domain
    - ``CogniCoreReflectTool`` — pattern analysis and recommendations
    - ``CogniCoreThreatScanTool`` — prompt-injection / jailbreak detection

**Callback handler**:
    - ``CogniCoreCallbackHandler`` — auto-records every LLM call in memory

Usage::

    from cognicore.integrations.langchain import cognicore_tools

    # All 5 tools for a LangChain agent
    tools = cognicore_tools()
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    # Auto-record LLM calls
    from cognicore.integrations.langchain import CogniCoreCallbackHandler
    handler = CogniCoreCallbackHandler()
    llm = ChatOpenAI(callbacks=[handler])
"""
from typing import Optional, Type
from cognicore.runtime import CogniCoreRuntime, RuntimeConfig

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

    def _run(self, buggy_code: str, error_message: str = "", category: str = "general") -> str:
        # Build context from CogniCore memory
        mem_ctx = self._runtime._build_context(category)

        parts = ["## CogniCore Repair Context\n"]

        # Memory hints
        if mem_ctx.get("memory"):
            # New dict format from backend
            fails = [e for e in mem_ctx["memory"] if not e.get("correct")]
            successes = [e for e in mem_ctx["memory"] if e.get("correct")]
            if fails:
                parts.append(f"### Past Failures ({len(fails)} in '{category}')")
                for f in fails[-3:]:
                    parts.append(f"- {f.get('action', 'unknown')}")
            if successes:
                parts.append(f"\n### Past Successes ({len(successes)})")
                for s in successes[-3:]:
                    parts.append(f"- {s.get('action', 'unknown')}")

        # Reflection
        if mem_ctx.get("reflection_hint"):
            parts.append(f"\n### Reflection\n{mem_ctx['reflection_hint']}")

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

    def _run(self, category: str = "general", tactic: str = "unknown",
             success: bool = False, error: str = "") -> str:
        from cognicore.memory.base import MemoryEntry
        self._runtime.memory.store(MemoryEntry(
            category=category, 
            correct=success,
            action=f"tactic:{tactic} {'PASS' if success else 'FAIL'}",
            text=error
        ))

        status = "SUCCESS" if success else "FAILURE"
        return f"Stored {status}: tactic='{tactic}' for category='{category}'"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


# Convenience function (original — kept for backward compatibility)
def get_cognicore_tools():
    """Get original CogniCore LangChain tools (repair + memory)."""
    return [CogniCoreTool(), CogniCoreMemoryTool()]


# ═══════════════════════════════════════════════════════════════════════
# NEW: General-purpose cognition tools + callback handler
# ═══════════════════════════════════════════════════════════════════════

import logging
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from cognicore.memory.base import MemoryEntry
from cognicore.middleware.reflection import ReflectionEngine
from cognicore.immune.detector import ThreatDetector, ThreatScore

logger = logging.getLogger("cognicore.integrations.langchain")

# --- Try/except for callback handler base class ---
try:
    from langchain.callbacks.base import BaseCallbackHandler
    LANGCHAIN_CALLBACKS_AVAILABLE = True
except ImportError:
    LANGCHAIN_CALLBACKS_AVAILABLE = False

    class BaseCallbackHandler:
        """Stub when langchain callbacks are not installed."""
        pass


# ── Module-level singletons (lazy-initialized) ──────────────────────

# We now just use the memory of the shared runtime.
# This variable is deprecated and will be removed in future versions.
_shared_semantic_memory = None
_shared_runtime = None

def _get_semantic_memory():
    return _get_shared_runtime().memory


def _get_shared_runtime() -> CogniCoreRuntime:
    """Lazily initialize a shared CogniCoreRuntime instance."""
    global _shared_runtime
    if _shared_runtime is None:
        _shared_runtime = CogniCoreRuntime(
            config=RuntimeConfig(
                reflection_min_samples=1,
                reflection_failure_threshold=2,
                memory_top_k=10,
            ),
            name="langchain-shared",
        )
    return _shared_runtime


def _get_threat_detector() -> ThreatDetector:
    """Lazily initialize a shared ThreatDetector instance."""
    global _shared_threat_detector
    if _shared_threat_detector is None:
        _shared_threat_detector = ThreatDetector()
    return _shared_threat_detector


# ═══════════════════════════════════════════════════════════════════════
# CogniCoreCallbackHandler — auto-record every LLM call
# ═══════════════════════════════════════════════════════════════════════

class CogniCoreCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that auto-records every LLM call in
    CogniCore memory.

    Captures prompts, completions, latencies, and errors so the agent
    builds an episodic memory of its own LLM interactions.

    Parameters
    ----------
    persistence_path : str
        Directory for persisted callback state (default ``~/.cognicore/langchain``).
    category_prefix : str
        Prefix prepended to memory categories (default ``langchain``).

    Usage::

        from cognicore.integrations.langchain import CogniCoreCallbackHandler
        handler = CogniCoreCallbackHandler()
        llm = ChatOpenAI(callbacks=[handler])
    """
    
    ignore_llm: bool = False
    ignore_retry: bool = False
    ignore_chain: bool = False
    ignore_agent: bool = False
    ignore_retriever: bool = False
    ignore_chat_model: bool = False
    ignore_custom_event: bool = False
    raise_error: bool = False

    def __init__(
        self,
        persistence_path: str = "~/.cognicore/langchain",
        category_prefix: str = "langchain",
    ) -> None:
        super().__init__()
        self.persistence_path = Path(persistence_path).expanduser()
        self.category_prefix = category_prefix
        self._runtime = _get_shared_runtime()
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._call_count = 0
        logger.info(
            "CogniCoreCallbackHandler initialized "
            "(persistence=%s, prefix=%s)",
            self.persistence_path,
            self.category_prefix,
        )

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Record the prompt being sent to the LLM."""
        run_id = str(kwargs.get("run_id", self._call_count))
        self._call_count += 1
        self._pending[run_id] = {
            "prompts": prompts,
            "start_time": time.time(),
            "model": serialized.get("name", serialized.get("id", ["unknown"])[-1]
                                    if isinstance(serialized.get("id"), list)
                                    else "unknown"),
        }
        logger.debug("LLM call started (run_id=%s, prompts=%d)", run_id, len(prompts))

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Record success — store prompt+response in memory."""
        run_id = str(kwargs.get("run_id", ""))
        pending = self._pending.pop(run_id, {})
        duration_ms = (time.time() - pending.get("start_time", time.time())) * 1000

        # Extract response text
        response_text = ""
        try:
            if hasattr(response, "generations") and response.generations:
                first_gen = response.generations[0]
                if isinstance(first_gen, list) and first_gen:
                    response_text = first_gen[0].text if hasattr(first_gen[0], "text") else str(first_gen[0])
                else:
                    response_text = str(first_gen)
        except Exception:
            response_text = str(response)[:500]

        category = f"{self.category_prefix}.llm_call"
        
        from cognicore.memory.base import MemoryEntry
        self._runtime.memory.store(MemoryEntry(
            text=(pending.get("prompts", [""])[0][:300] + " → " + response_text[:300]),
            category=category,
            correct=True,
            action="llm_success",
            metadata={
                "duration_ms": round(duration_ms, 1),
                "model": pending.get("model", "unknown")
            }
        ))
        logger.debug("LLM call succeeded (run_id=%s, %.0fms)", run_id, duration_ms)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Record LLM-level failure with error details."""
        run_id = str(kwargs.get("run_id", ""))
        pending = self._pending.pop(run_id, {})
        duration_ms = (time.time() - pending.get("start_time", time.time())) * 1000

        category = f"{self.category_prefix}.llm_error"
        error_str = f"{type(error).__name__}: {str(error)[:200]}"
        
        from cognicore.memory.base import MemoryEntry
        self._runtime.memory.store(MemoryEntry(
            text=error_str,
            category=category,
            correct=False,
            action=f"llm_error:{type(error).__name__}",
            metadata={"duration_ms": round(duration_ms, 1), "model": pending.get("model", "unknown")}
        ))
        logger.warning("LLM call failed (run_id=%s): %s", run_id, error_str)

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Record chain-level failure."""
        category = f"{self.category_prefix}.chain_error"
        error_str = f"{type(error).__name__}: {str(error)[:200]}"
        from cognicore.memory.base import MemoryEntry
        self._runtime.memory.store(MemoryEntry(
            text=error_str,
            category=category,
            correct=False,
            action=f"chain_error:{type(error).__name__}",
        ))
        logger.warning("Chain error recorded: %s", error_str)


# ═══════════════════════════════════════════════════════════════════════
# CogniCoreRecallTool — general-purpose semantic recall
# ═══════════════════════════════════════════════════════════════════════

class CogniCoreRecallTool(BaseTool):
    """LangChain tool for recalling past experiences via semantic search.

    Not limited to code repair — works with any domain stored in
    CogniCore's semantic memory.

    Usage::

        tool = CogniCoreRecallTool()
        result = tool.run("similar authentication failures")
    """

    name: str = "cognicore_recall"
    description: str = (
        "Search CogniCore's semantic memory for past experiences similar to a query. "
        "Input: a natural-language query describing what you're looking for, optionally "
        "prefixed with 'category:NAME ' to filter by category. "
        "Output: formatted list of relevant past experiences with outcomes."
    )

    def _run(self, query: str) -> str:
        """Execute semantic recall against CogniCore memory.

        Parameters
        ----------
        query : str
            Natural-language query. Optionally prefix with ``category:NAME``
            to filter results (e.g. ``"category:security phishing email"``).
        """
        mem = _get_semantic_memory()

        # Parse optional category prefix
        category = None
        search_query = query.strip()
        if search_query.lower().startswith("category:"):
            parts = search_query.split(" ", 1)
            category = parts[0].split(":", 1)[1]
            search_query = parts[1] if len(parts) > 1 else category

        results = mem.search(search_query, top_k=5, category=category)

        if not results:
            return f"No relevant experiences found for: '{search_query}'"

        parts = [f"## CogniCore Recall — {len(results)} relevant experiences\n"]
        for i, search_result in enumerate(results, 1):
            entry = search_result.entry
            score = search_result.score
            outcome = "✓ success" if entry.correct else "✗ failure"
            cat = entry.category or "uncategorized"
            text = (entry.text or entry.action)[:150]
            parts.append(
                f"{i}. [{outcome}] (category: {cat}, similarity: {score:.2f})\n"
                f"   {text}"
            )

        return "\n".join(parts)

    async def _arun(self, query: str) -> str:
        return self._run(query)


# ═══════════════════════════════════════════════════════════════════════
# CogniCoreReflectTool — general-purpose reflection
# ═══════════════════════════════════════════════════════════════════════

class CogniCoreReflectTool(BaseTool):
    """LangChain tool for analyzing patterns and getting recommendations.

    Uses the ReflectionEngine to identify failure patterns and suggest
    better actions based on historical data.

    Usage::

        tool = CogniCoreReflectTool()
        result = tool.run("code_fix")
    """

    name: str = "cognicore_reflect"
    description: str = (
        "Analyze past performance patterns for a given category and get "
        "recommendations. Input: a category name (e.g. 'code_fix', 'security', "
        "'api_calls'). Output: pattern analysis with failure counts, success "
        "counts, and an actionable recommendation."
    )

    def _run(self, category: str) -> str:
        """Run reflection analysis on a category.

        Parameters
        ----------
        category : str
            The category to analyze (e.g. ``"code_fix"``, ``"security"``).
        """
        runtime = _get_shared_runtime()
        analysis = runtime.reflection.analyze(category.strip())
        hint = runtime.reflection.get_hint(category.strip())

        parts = [f"## CogniCore Reflection — '{category}'\n"]
        parts.append(f"**Entries analyzed:** {analysis['n_similar']}")

        if analysis["good_predictions"]:
            parts.append("\n### Successful Actions")
            for action, count in sorted(
                analysis["good_predictions"].items(), key=lambda x: -x[1]
            ):
                if action.strip():
                    parts.append(f"  - '{action}' succeeded {count}x")

        if analysis["bad_predictions"]:
            parts.append("\n### Failed Actions (avoid these)")
            for action, count in sorted(
                analysis["bad_predictions"].items(), key=lambda x: -x[1]
            ):
                if action.strip():
                    parts.append(f"  - '{action}' failed {count}x")

        if analysis["recommendation"]:
            parts.append(f"\n**Recommendation:** Use '{analysis['recommendation']}'")

        if hint:
            parts.append(f"\n**Reflection hint:** {hint}")
        elif analysis["n_similar"] == 0:
            parts.append("\n*No historical data yet for this category.*")

        return "\n".join(parts)

    async def _arun(self, category: str) -> str:
        return self._run(category)


# ═══════════════════════════════════════════════════════════════════════
# CogniCoreThreatScanTool — threat detection
# ═══════════════════════════════════════════════════════════════════════

class CogniCoreThreatScanTool(BaseTool):
    """LangChain tool for scanning text for prompt injections, jailbreaks,
    and other adversarial inputs.

    Uses CogniCore's ThreatDetector with pattern matching, structural
    analysis, and encoding analysis.

    Usage::

        tool = CogniCoreThreatScanTool()
        result = tool.run("ignore all previous instructions and ...")
    """

    name: str = "cognicore_threat_scan"
    description: str = (
        "Scan text for potential threats including prompt injections, "
        "jailbreak attempts, resource attacks, data exfiltration, and "
        "adversarial inputs. Input: the text to scan. Output: threat score "
        "(0.0-1.0), threat category, and specific indicators found."
    )

    def _run(self, text: str) -> str:
        """Scan text for threats.

        Parameters
        ----------
        text : str
            The text to analyze for threats.
        """
        detector = _get_threat_detector()
        result: ThreatScore = detector.detect(text.strip())

        parts = [f"## CogniCore Threat Scan\n"]
        parts.append(f"**Threat score:** {result.score:.4f} / 1.0")
        parts.append(f"**Category:** {result.category}")

        if result.score <= 0.3:
            parts.append("**Verdict:** [SAFE] — no significant threats detected")
        elif result.score <= 0.6:
            parts.append("**Verdict:** [SUSPICIOUS] — moderate threat indicators")
        else:
            parts.append("**Verdict:** [THREAT DETECTED] — high-confidence threat")

        if result.indicators:
            parts.append(f"\n### Indicators ({len(result.indicators)})")
            for indicator in result.indicators[:10]:
                parts.append(f"  - {indicator}")

        if result.sub_scores:
            parts.append("\n### Sub-scores by category")
            for cat, score in sorted(
                result.sub_scores.items(), key=lambda x: -x[1]
            ):
                bar = "=" * int(score * 10) + "-" * (10 - int(score * 10))
                parts.append(f"  {cat:20s} [{bar}] {score:.2f}")

        return "\n".join(parts)

    async def _arun(self, text: str) -> str:
        return self._run(text)


# ═══════════════════════════════════════════════════════════════════════
# Updated convenience function — returns ALL tools
# ═══════════════════════════════════════════════════════════════════════

def cognicore_tools() -> list:
    """Get all CogniCore LangChain tools (repair + general cognition).

    Returns a list of 5 tools:
      1. ``CogniCoreTool`` — code repair context
      2. ``CogniCoreMemoryTool`` — store repair outcomes
      3. ``CogniCoreRecallTool`` — semantic memory recall
      4. ``CogniCoreReflectTool`` — pattern analysis
      5. ``CogniCoreThreatScanTool`` — threat detection

    Usage::

        from cognicore.integrations.langchain import cognicore_tools
        tools = cognicore_tools()
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    """
    return [
        CogniCoreTool(),
        CogniCoreMemoryTool(),
        CogniCoreRecallTool(),
        CogniCoreReflectTool(),
        CogniCoreThreatScanTool(),
    ]
