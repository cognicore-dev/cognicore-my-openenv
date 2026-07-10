"""
CogniCore OpenAI Agents SDK Integration — function tools + pre-built agent.

Provides CogniCore cognition as ``@function_tool`` decorated functions
compatible with the OpenAI Agents SDK (``agents`` package):

- ``remember_outcome`` — store task outcomes in memory
- ``recall_context`` — semantic similarity search for past experiences
- ``recall_failures`` — retrieve past failures to avoid repeating them
- ``reflect_on_category`` — pattern analysis and recommendations
- ``scan_for_threats`` — prompt-injection / jailbreak detection

Pre-built agent:

- ``CogniCoreReflectionAgent`` — an Agent that analyzes failures when
  handed off to, providing data-driven debugging assistance

Usage::

    from cognicore.integrations.openai_agents import cognicore_openai_tools

    # Add all tools to an OpenAI agent
    from agents import Agent
    agent = Agent(
        name="My Agent",
        instructions="You are a helpful assistant with memory.",
        tools=cognicore_openai_tools(),
    )

    # Or use the pre-built reflection agent for handoffs
    from cognicore.integrations.openai_agents import CogniCoreReflectionAgent
    triage = Agent(
        name="Triage",
        instructions="Hand off failures to the reflection agent.",
        handoffs=[CogniCoreReflectionAgent],
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cognicore.integrations.openai_agents")

# ── Try/except guard for OpenAI Agents SDK ───────────────────────────

try:
    from agents import Agent, function_tool
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    logger.debug(
        "OpenAI Agents SDK ('agents' package) not installed. "
        "Install with: pip install openai-agents"
    )

    # Stub: no-op decorator that preserves the function
    def function_tool(fn=None, **kwargs):
        """Stub decorator when agents package is not installed."""
        if fn is not None:
            return fn
        return lambda f: f

    class Agent:
        """Stub Agent when agents package is not installed."""
        def __init__(self, **kwargs):
            self.name = kwargs.get("name", "stub")
            self.instructions = kwargs.get("instructions", "")
            self.tools = kwargs.get("tools", [])

# ── CogniCore imports ────────────────────────────────────────────────

from cognicore.runtime import CogniCoreRuntime, RuntimeConfig
from cognicore.advanced_memory import SemanticMemory
from cognicore.middleware.reflection import ReflectionEngine
from cognicore.middleware.memory import Memory
from cognicore.immune.detector import ThreatDetector, ThreatScore

# ── Module-level singletons (lazy-initialized) ──────────────────────

_shared_runtime: Optional[CogniCoreRuntime] = None
_shared_semantic_memory: Optional[SemanticMemory] = None
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
            name="openai-agents-shared",
        )
        logger.info("Shared CogniCoreRuntime initialized for OpenAI Agents")
    return _shared_runtime


def _get_semantic_memory() -> SemanticMemory:
    """Lazily initialize a shared SemanticMemory instance."""
    global _shared_semantic_memory
    if _shared_semantic_memory is None:
        _shared_semantic_memory = SemanticMemory(
            max_size=10_000, decay_rate=0.95, similarity_threshold=0.01
        )
    return _shared_semantic_memory


def _get_threat_detector() -> ThreatDetector:
    """Lazily initialize a shared ThreatDetector instance."""
    global _shared_threat_detector
    if _shared_threat_detector is None:
        _shared_threat_detector = ThreatDetector()
    return _shared_threat_detector


# ═══════════════════════════════════════════════════════════════════════
# Function tools
# ═══════════════════════════════════════════════════════════════════════

@function_tool
def remember_outcome(
    text: str,
    category: str = "general",
    success: bool = False,
    action: str = "unknown",
) -> str:
    """Store a task outcome in CogniCore's persistent memory.

    Records the experience in both episodic memory (for reflection
    pattern analysis) and semantic memory (for similarity-based recall).

    Args:
        text: Description of the outcome or experience.
        category: Category for grouping (e.g. 'security', 'code_fix').
        success: Whether the outcome was successful.
        action: The action or tactic that was used.

    Returns:
        Confirmation message with memory statistics.
    """
    runtime = _get_runtime()
    sem_mem = _get_semantic_memory()

    # Episodic memory (for reflection)
    runtime.memory.store({
        "category": category,
        "correct": success,
        "predicted": f"action:{action} {'PASS' if success else 'FAIL'}",
    })

    # Semantic memory (for recall)
    sem_mem.store({
        "text": text,
        "category": category,
        "correct": success,
        "predicted": action,
    })

    status = "SUCCESS" if success else "FAILURE"
    logger.info("Stored %s: action='%s' category='%s'", status, action, category)
    return (
        f"Stored {status}: action='{action}' for category='{category}'. "
        f"Memory: {len(runtime.memory.entries)} episodic, "
        f"{len(sem_mem.entries)} semantic entries."
    )


@function_tool
def recall_context(query: str, category: str = "") -> str:
    """Search CogniCore's semantic memory for relevant past experiences.

    Uses TF-IDF similarity to find the most relevant experiences
    matching the query. Optionally filter by category.

    Args:
        query: Natural-language query describing what to find.
        category: Optional category filter (empty string for all).

    Returns:
        Formatted list of matching experiences with outcomes and scores.
    """
    sem_mem = _get_semantic_memory()
    results = sem_mem.semantic_search(query.strip(), top_k=5)

    # Filter by category if provided
    if category:
        results = [
            (entry, score) for entry, score in results
            if entry.get("category", "").lower() == category.lower()
        ]

    if not results:
        return f"No relevant experiences found for: '{query}'"

    parts = [f"## Recall — {len(results)} relevant experiences\n"]
    for i, (entry, score) in enumerate(results, 1):
        outcome = "✓ success" if entry.get("correct") else "✗ failure"
        cat = entry.get("category", "uncategorized")
        text = entry.get("text", entry.get("predicted", ""))[:150]
        parts.append(
            f"{i}. [{outcome}] (category: {cat}, similarity: {score:.2f})\n"
            f"   {text}"
        )

    return "\n".join(parts)


@function_tool
def recall_failures(category: str = "general", limit: int = 5) -> str:
    """Retrieve past failures from CogniCore memory to avoid repeating them.

    Specifically surfaces failed experiences so the agent can learn
    what NOT to do.

    Args:
        category: Category to search for failures in.
        limit: Maximum number of failures to return.

    Returns:
        Formatted list of past failures with details.
    """
    runtime = _get_runtime()
    sem_mem = _get_semantic_memory()

    # Episodic failures
    episodic_failures = runtime.memory.retrieve(category, top_k=50)
    episodic_failures = [
        e for e in episodic_failures if e.get("correct") is False
    ][:limit]

    # Semantic failures
    semantic_results = sem_mem.semantic_search(category, top_k=20)
    semantic_failures = [
        (entry, score)
        for entry, score in semantic_results
        if entry.get("correct") is False
    ][:limit]

    if not episodic_failures and not semantic_failures:
        return f"No failures recorded for category: '{category}'"

    parts = [f"## Past Failures — '{category}'\n"]

    if episodic_failures:
        parts.append(f"### Episodic ({len(episodic_failures)} failures)")
        for i, entry in enumerate(episodic_failures, 1):
            action = entry.get("predicted", "unknown")
            parts.append(f"  {i}. {action}")

    if semantic_failures:
        parts.append(f"\n### Semantic ({len(semantic_failures)} failures)")
        for i, (entry, score) in enumerate(semantic_failures, 1):
            text = entry.get("text", entry.get("predicted", ""))[:150]
            parts.append(f"  {i}. (similarity: {score:.2f}) {text}")

    parts.append(
        "\n**Recommendation:** Avoid the actions listed above. "
        "Use `reflect_on_category` to get positive recommendations."
    )

    return "\n".join(parts)


@function_tool
def reflect_on_category(category: str) -> str:
    """Analyze past performance patterns for a category and get recommendations.

    Uses CogniCore's ReflectionEngine to identify which actions succeed
    or fail in the given category. Provides data-driven recommendations.

    Args:
        category: The category to analyze (e.g. 'security', 'code_fix').

    Returns:
        Structured analysis with success/failure counts and recommendation.
    """
    runtime = _get_runtime()
    analysis = runtime.reflection.analyze(category.strip())
    hint = runtime.reflection.get_hint(category.strip())

    parts = [f"## Reflection — '{category}'\n"]
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


@function_tool
def scan_for_threats(text: str) -> str:
    """Scan text for prompt injections, jailbreaks, and adversarial inputs.

    Uses CogniCore's ThreatDetector with pattern matching, structural
    analysis, and encoding analysis to detect threats.

    Args:
        text: The text to scan for threats.

    Returns:
        Threat score (0.0-1.0), category, verdict, and indicators.
    """
    detector = _get_threat_detector()
    result: ThreatScore = detector.detect(text.strip())

    parts = ["## Threat Scan\n"]
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
        parts.append("\n### Sub-scores")
        for cat, score in sorted(
            result.sub_scores.items(), key=lambda x: -x[1]
        ):
            bar = "=" * int(score * 10) + "-" * (10 - int(score * 10))
            parts.append(f"  {cat:20s} [{bar}] {score:.2f}")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Convenience: get all function tools
# ═══════════════════════════════════════════════════════════════════════

def cognicore_openai_tools() -> list:
    """Get all CogniCore function tools for the OpenAI Agents SDK.

    Returns a list of 5 function tools:
      1. ``remember_outcome`` — store outcomes in memory
      2. ``recall_context`` — semantic similarity search
      3. ``recall_failures`` — retrieve past failures
      4. ``reflect_on_category`` — pattern analysis
      5. ``scan_for_threats`` — threat detection

    Usage::

        from cognicore.integrations.openai_agents import cognicore_openai_tools
        from agents import Agent

        agent = Agent(
            name="My Agent",
            instructions="You are a helpful assistant with memory.",
            tools=cognicore_openai_tools(),
        )
    """
    return [
        remember_outcome,
        recall_context,
        recall_failures,
        reflect_on_category,
        scan_for_threats,
    ]


# ═══════════════════════════════════════════════════════════════════════
# Pre-built Reflection Agent
# ═══════════════════════════════════════════════════════════════════════

_REFLECTION_AGENT_INSTRUCTIONS = """\
You are a CogniCore Reflection Agent. Your purpose is to analyze failures
and provide data-driven debugging assistance.

When you receive a handoff, you should:

1. **Identify the category** of the failure from the context provided.
2. Use `reflect_on_category` to analyze historical patterns for that category.
3. Use `recall_failures` to surface specific past failures.
4. Use `recall_context` to find similar past situations and their outcomes.
5. If the input looks suspicious, use `scan_for_threats` to check.

Provide a structured analysis with:
- What has failed before in this category
- What has succeeded before
- A concrete recommendation for what to try next
- Any threats or concerns detected

Always ground your recommendations in the data from CogniCore's memory.
"""


def _build_reflection_agent() -> Agent:
    """Build the pre-configured CogniCore Reflection Agent.

    Returns
    -------
    Agent
        An OpenAI Agents SDK Agent configured with all CogniCore tools
        and instructions for failure analysis.
    """
    return Agent(
        name="CogniCore Reflection Agent",
        instructions=_REFLECTION_AGENT_INSTRUCTIONS,
        tools=[
            recall_context,
            recall_failures,
            reflect_on_category,
            scan_for_threats,
        ],
    )


# Lazy-initialized singleton for the reflection agent
_reflection_agent: Optional[Agent] = None


def get_reflection_agent() -> Agent:
    """Get the pre-built CogniCore Reflection Agent (singleton).

    Usage::

        from cognicore.integrations.openai_agents import get_reflection_agent
        from agents import Agent

        reflection = get_reflection_agent()
        triage = Agent(
            name="Triage",
            instructions="Hand off failures to the reflection agent.",
            handoffs=[reflection],
        )
    """
    global _reflection_agent
    if _reflection_agent is None:
        _reflection_agent = _build_reflection_agent()
    return _reflection_agent


# Module-level convenience (builds on first access when imported)
# Use get_reflection_agent() for lazy initialization
CogniCoreReflectionAgent = None


def _init_module_agent() -> None:
    """Initialize the module-level CogniCoreReflectionAgent.

    Called lazily — the agent is only built when explicitly accessed
    via ``get_reflection_agent()`` or when the module attribute is read.
    """
    global CogniCoreReflectionAgent
    if CogniCoreReflectionAgent is None and AGENTS_AVAILABLE:
        CogniCoreReflectionAgent = _build_reflection_agent()


def __getattr__(name: str):
    """Module-level __getattr__ for lazy initialization of CogniCoreReflectionAgent."""
    if name == "CogniCoreReflectionAgent":
        _init_module_agent()
        return CogniCoreReflectionAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
