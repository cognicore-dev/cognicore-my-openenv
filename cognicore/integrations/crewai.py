"""
CogniCore CrewAI Integration — wraps CogniCore cognition as CrewAI tools.

Provides a complete set of cognition tools for CrewAI agents:

- ``CogniCoreRememberTool`` — store outcomes in episodic + semantic memory
- ``CogniCoreRecallTool`` — semantic similarity search over past experiences
- ``CogniCoreReflectTool`` — pattern analysis and failure-avoidance hints
- ``CogniCoreThreatScanTool`` — prompt-injection / jailbreak detection

All tools share a single CogniCoreRuntime instance (module-level singleton)
for consistent memory and reflection state across a crew's execution.

Usage::

    from cognicore.integrations.crewai import cognicore_crewai_tools

    tools = cognicore_crewai_tools()

    # Assign to a CrewAI agent
    from crewai import Agent
    analyst = Agent(
        role="Security Analyst",
        goal="Detect and respond to threats",
        tools=tools,
    )

    # Or use individual tools
    from cognicore.integrations.crewai import (
        CogniCoreRememberTool,
        CogniCoreRecallTool,
        CogniCoreReflectTool,
        CogniCoreThreatScanTool,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger("cognicore.integrations.crewai")

# ── Try/except guard for CrewAI ──────────────────────────────────────

try:
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

    class BaseTool:
        """Stub when crewai is not installed.

        Allows the module to be imported and classes to be defined
        without crewai as a hard dependency.
        """
        name: str = ""
        description: str = ""

        def _run(self, **kwargs) -> str:
            return "crewai is not installed"

# ── CogniCore imports ────────────────────────────────────────────────

from cognicore.runtime import CogniCoreRuntime, RuntimeConfig
from cognicore.memory.base import MemoryEntry
from cognicore.immune.detector import ThreatDetector, ThreatScore

# ── Module-level singletons (lazy-initialized) ──────────────────────

_shared_runtime: Optional[CogniCoreRuntime] = None
_shared_threat_detector: Optional[ThreatDetector] = None


def _get_runtime() -> CogniCoreRuntime:
    """Lazily initialize a shared CogniCoreRuntime instance."""
    global _shared_runtime
    if _shared_runtime is None:
        _shared_runtime = CogniCoreRuntime(
            config=RuntimeConfig(
                reflection_min_samples=1,
                reflection_failure_threshold=2,
                memory_top_k=10,
            ),
            name="crewai-shared",
        )
        logger.info("Shared CogniCoreRuntime initialized for CrewAI")
    return _shared_runtime


# We now just use the memory of the shared runtime.
# This variable is deprecated and will be removed in future versions.
_shared_semantic_memory = None

def _get_semantic_memory():
    return _get_runtime().memory


def _get_threat_detector() -> ThreatDetector:
    """Lazily initialize a shared ThreatDetector instance."""
    global _shared_threat_detector
    if _shared_threat_detector is None:
        _shared_threat_detector = ThreatDetector()
    return _shared_threat_detector


# ── Pydantic schemas for tool arguments ────────────────────────────────
try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel: pass
    def Field(*args, **kwargs): return None

class CogniCoreRememberSchema(BaseModel):
    text: str = Field(..., description="Description of the outcome or experience.")
    category: str = Field("general", description="Category for grouping (e.g. 'security' or 'code_fix').")
    success: bool = Field(False, description="Whether the outcome was successful.")
    action: str = Field("unknown", description="The action or tactic used.")

class CogniCoreRecallSchema(BaseModel):
    query: str = Field(..., description="Natural-language query describing what to recall.")
    category: Optional[str] = Field(None, description="If provided, only return results from this category.")

class CogniCoreReflectSchema(BaseModel):
    category: str = Field("general", description="The category to analyze (e.g. 'security', 'code_fix').")

class CogniCoreThreatScanSchema(BaseModel):
    text: str = Field(..., description="The content to scan for potential threats.")

# ═══════════════════════════════════════════════════════════════════════
# CogniCoreRememberTool — store outcomes in memory
# ═══════════════════════════════════════════════════════════════════════

class CogniCoreRememberTool(BaseTool):
    """CrewAI tool for storing task outcomes in CogniCore memory.

    Records both in the episodic memory (for reflection) and the semantic
    memory (for similarity-based recall), creating a durable experience
    log that the crew can learn from.
    """
    name: str = "cognicore_remember"
    description: str = (
        "Store an outcome or experience in CogniCore's persistent memory. "
        "This helps future tasks by recording what worked and what didn't."
    )
    args_schema: Type[BaseModel] = CogniCoreRememberSchema

    def _run(self, text: str, category: str = "general", success: bool = False, action: str = "unknown") -> str:
        if not text:
            return "Error: 'text' parameter is required."

        runtime = _get_runtime()
        
        # Store in unified memory (for reflection and recall)
        runtime.memory.store(MemoryEntry(
            text=text,
            category=category,
            correct=success,
            action=action
        ))

        status = "SUCCESS" if success else "FAILURE"
        logger.info(
            "Stored %s: action='%s' for category='%s'",
            status, action, category,
        )
        return (
            f"Stored {status}: action='{action}' for category='{category}'. "
            f"Memory now has {runtime.memory.count()} entries."
        )


# ═══════════════════════════════════════════════════════════════════════
# CogniCoreRecallTool — semantic similarity search
# ═══════════════════════════════════════════════════════════════════════

class CogniCoreRecallTool(BaseTool):
    """CrewAI tool for recalling past experiences via semantic search."""
    name: str = "cognicore_recall"
    description: str = (
        "Search CogniCore's semantic memory for past experiences similar "
        "to a query. Returns relevant past experiences with their outcomes "
        "and similarity scores."
    )
    args_schema: Type[BaseModel] = CogniCoreRecallSchema

    def _run(self, query: str, category: Optional[str] = None) -> str:
        if not query:
            return "Error: 'query' parameter is required."

        mem = _get_runtime().memory
        results = mem.search(query.strip(), top_k=5, category=category)

        if not results:
            return f"No relevant experiences found for: '{query}'"

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


# ═══════════════════════════════════════════════════════════════════════
# CogniCoreReflectTool — pattern analysis + recommendations
# ═══════════════════════════════════════════════════════════════════════

class CogniCoreReflectTool(BaseTool):
    """CrewAI tool for analyzing performance patterns and getting actionable recommendations."""
    name: str = "cognicore_reflect"
    description: str = (
        "Analyze past performance patterns for a given category. "
        "Returns a structured analysis with success/failure counts, "
        "the worst-performing actions, and a recommended action."
    )
    args_schema: Type[BaseModel] = CogniCoreReflectSchema

    def _run(self, category: str = "general") -> str:
        category = category.strip()

        runtime = _get_runtime()
        analysis = runtime.reflection.analyze(category)
        hint = runtime.reflection.get_hint(category)

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
            parts.append(
                f"\n**Recommendation:** Use '{analysis['recommendation']}'"
            )

        if hint:
            parts.append(f"\n**Reflection hint:** {hint}")
        elif analysis["n_similar"] == 0:
            parts.append("\n*No historical data yet for this category.*")

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# CogniCoreThreatScanTool — threat detection
# ═══════════════════════════════════════════════════════════════════════

class CogniCoreThreatScanTool(BaseTool):
    """CrewAI tool for scanning text for adversarial threats."""
    name: str = "cognicore_threat_scan"
    description: str = (
        "Scan text for potential threats including prompt injections, "
        "jailbreak attempts, resource attacks, and data exfiltration. "
        "Returns a threat score (0.0-1.0), category, verdict, and specific indicators."
    )
    args_schema: Type[BaseModel] = CogniCoreThreatScanSchema

    def _run(self, text: str) -> str:
        if not text:
            return "Error: 'text' parameter is required."

        detector = _get_threat_detector()
        result: ThreatScore = detector.detect(text.strip())

        parts = ["## CogniCore Threat Scan\n"]
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


# ═══════════════════════════════════════════════════════════════════════
# Convenience: get all CrewAI tools
# ═══════════════════════════════════════════════════════════════════════

def cognicore_crewai_tools() -> list:
    """Get all CogniCore CrewAI tools.

    Returns a list of 4 tools that share a single runtime instance:
      1. ``CogniCoreRememberTool`` — store outcomes
      2. ``CogniCoreRecallTool`` — semantic recall
      3. ``CogniCoreReflectTool`` — pattern analysis
      4. ``CogniCoreThreatScanTool`` — threat detection
    """
    return [
        CogniCoreRememberTool(),
        CogniCoreRecallTool(),
        CogniCoreReflectTool(),
        CogniCoreThreatScanTool(),
    ]
