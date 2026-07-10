"""Retrieval Planner for CogniCore's Agent Memory Operating System.

Replaces naive Top-K retrieval with intelligent planning.  The planner
inspects the incoming query and task context to decide *which* memory
types to consult, in what priority order, and with what budget — before
any actual retrieval takes place.

Four concrete implementations are provided:

* **RetrievalPlanner** — abstract base class.
* **RuleBasedPlanner** — zero-dependency, keyword-heuristic planner.
* **AdaptivePlanner** — learns from ``MemoryTrace`` history to improve
  planning decisions over time.
* **AlwaysRetrievePlanner** — trivial baseline that retrieves from every
  memory type with a fixed budget.

All implementations use only the Python standard library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from cognicore.memory.base import MemoryTrace, MemoryType, RetrievalPlan


# ======================================================================
# Keyword sets used by the rule-based planner
# ======================================================================

_ERROR_KEYWORDS: Set[str] = {
    "error", "bug", "fail", "failed", "failure", "crash", "exception",
    "traceback", "debug", "fix", "broken", "wrong", "issue", "problem",
    "mistake", "anti-pattern", "antipattern",
}

_PROCEDURAL_KEYWORDS: Set[str] = {
    "how", "steps", "step", "procedure", "process", "guide", "tutorial",
    "walkthrough", "instructions", "setup", "install", "configure",
    "build", "deploy", "run", "execute", "workflow", "recipe",
}

_PREFERENCE_KEYWORDS: Set[str] = {
    "prefer", "preference", "style", "format", "convention", "standard",
    "like", "dislike", "want", "should", "always", "never", "rule",
    "constraint", "restriction", "boundary", "limit",
}

_RECALL_KEYWORDS: Set[str] = {
    "remember", "recall", "history", "previous", "earlier", "before",
    "last", "past", "ago", "conversation", "discussed", "mentioned",
    "told", "said", "asked",
}


# ======================================================================
# Abstract base
# ======================================================================

class RetrievalPlanner(ABC):
    """Abstract base class for retrieval planners.

    A retrieval planner examines the incoming *query* and optional *task*
    context and produces a :class:`RetrievalPlan` that guides downstream
    retrieval — which memory types to search, in what order, and how
    many results to fetch.

    Subclasses must implement :meth:`plan`.
    """

    @abstractmethod
    def plan(
        self,
        query: str,
        task: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> RetrievalPlan:
        """Produce a retrieval plan for the given *query*.

        Parameters
        ----------
        query:
            The user or agent query that will be used for memory search.
        task:
            An optional high-level task description providing broader
            context (e.g. ``"code review"``, ``"write documentation"``).
        context:
            Arbitrary contextual metadata the caller wishes to pass to
            the planner (e.g. current agent state, active tool, etc.).

        Returns
        -------
        RetrievalPlan
            A plan describing *whether* to retrieve, *which* types to
            consult, and *how many* results to fetch.
        """
        ...


# ======================================================================
# Rule-based planner
# ======================================================================

class RuleBasedPlanner(RetrievalPlanner):
    """Zero-dependency heuristic retrieval planner.

    Uses keyword analysis to decide which :class:`MemoryType` values are
    most likely to be relevant, sets a retrieval budget proportional to
    estimated query complexity, and produces a human-readable reasoning
    string explaining the choices made.

    Parameters
    ----------
    default_budget:
        The baseline retrieval budget when no complexity signal is
        available.  Defaults to ``5``.
    """

    def __init__(self, default_budget: int = 5) -> None:
        self.default_budget = default_budget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        query: str,
        task: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> RetrievalPlan:
        """Build a retrieval plan using keyword heuristics.

        The method proceeds in three phases:

        1. **Gate check** — queries that are empty or very short (fewer
           than 5 characters) are rejected early with
           ``should_retrieve=False``.
        2. **Type selection** — the query is tokenised and matched
           against curated keyword sets to determine which memory types
           are most likely to be useful.
        3. **Budget estimation** — a simple complexity heuristic
           (word count + task presence) adjusts the budget up or down
           from the configured default.
        """
        # Phase 1: gate check
        stripped = query.strip()
        if len(stripped) < 5:
            return RetrievalPlan(
                should_retrieve=False,
                memory_types=[],
                budget=0,
                priority_order=[],
                reasoning=(
                    f"Query too short ({len(stripped)} chars) — skipping retrieval."
                ),
            )

        # Phase 2: determine relevant memory types
        tokens = self._tokenize(stripped)
        combined_tokens = tokens | self._tokenize(task)
        priority_order, reasoning_parts = self._classify_tokens(combined_tokens)

        memory_types = list(dict.fromkeys(priority_order))  # deduplicated, order preserved

        # Phase 3: estimate budget
        budget, budget_reason = self._estimate_budget(stripped, task)
        reasoning_parts.append(budget_reason)

        return RetrievalPlan(
            should_retrieve=True,
            memory_types=memory_types,
            budget=budget,
            priority_order=priority_order,
            reasoning=" | ".join(reasoning_parts),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Lowercase split into a token set, stripping punctuation."""
        cleaned = text.lower()
        for ch in ".,;:!?()[]{}'\"-/\\@#$%^&*~`":
            cleaned = cleaned.replace(ch, " ")
        return set(cleaned.split())

    @staticmethod
    def _classify_tokens(
        tokens: Set[str],
    ) -> tuple[list[str], list[str]]:
        """Return (priority_order, reasoning_parts) based on keyword overlap.

        Multiple keyword sets may match; results are merged in priority
        order with higher-signal matches first.
        """
        priority: list[str] = []
        reasons: list[str] = []

        error_overlap = tokens & _ERROR_KEYWORDS
        proc_overlap = tokens & _PROCEDURAL_KEYWORDS
        pref_overlap = tokens & _PREFERENCE_KEYWORDS
        recall_overlap = tokens & _RECALL_KEYWORDS

        if error_overlap:
            priority.extend([MemoryType.FAILURE.value, MemoryType.PROCEDURAL.value])
            reasons.append(
                f"Error/debug keywords detected ({', '.join(sorted(error_overlap))}) "
                "→ prioritising FAILURE, PROCEDURAL"
            )

        if proc_overlap:
            priority.extend([MemoryType.PROCEDURAL.value, MemoryType.KNOWLEDGE.value])
            reasons.append(
                f"Procedural keywords detected ({', '.join(sorted(proc_overlap))}) "
                "→ prioritising PROCEDURAL, KNOWLEDGE"
            )

        if pref_overlap:
            priority.extend([MemoryType.PREFERENCE.value, MemoryType.CONSTRAINT.value])
            reasons.append(
                f"Preference/constraint keywords detected ({', '.join(sorted(pref_overlap))}) "
                "→ prioritising PREFERENCE, CONSTRAINT"
            )

        if recall_overlap:
            priority.extend([MemoryType.EPISODIC.value, MemoryType.SEMANTIC.value])
            reasons.append(
                f"Recall keywords detected ({', '.join(sorted(recall_overlap))}) "
                "→ prioritising EPISODIC, SEMANTIC"
            )

        # Fallback if no keywords matched
        if not priority:
            priority = [
                MemoryType.SEMANTIC.value,
                MemoryType.EPISODIC.value,
                MemoryType.KNOWLEDGE.value,
            ]
            reasons.append(
                "No strong keyword signals — using default priority "
                "(SEMANTIC, EPISODIC, KNOWLEDGE)"
            )

        return priority, reasons

    def _estimate_budget(self, query: str, task: str) -> tuple[int, str]:
        """Return (budget, reason) based on query/task complexity."""
        word_count = len(query.split())
        has_task = bool(task.strip())

        if word_count <= 4 and not has_task:
            budget = 3
            reason = f"Simple query ({word_count} words, no task) → budget=3"
        elif word_count >= 12 or has_task:
            budget = 7
            reason = (
                f"Complex query ({word_count} words"
                + (", task provided" if has_task else "")
                + ") → budget=7"
            )
        else:
            budget = self.default_budget
            reason = f"Moderate query ({word_count} words) → budget={self.default_budget}"

        return budget, reason


# ======================================================================
# Adaptive planner
# ======================================================================

class AdaptivePlanner(RetrievalPlanner):
    """Retrieval planner that learns from past :class:`MemoryTrace` history.

    The adaptive planner maintains a list of completed traces and uses
    them to make better planning decisions:

    * **Type selection** — memory types that have historically produced
      *positive* outcomes for similar queries are prioritised.
    * **Budget adjustment** — if historical traces show that retrieved
      memories were largely ignored, the budget is automatically reduced
      to avoid wasting context window space.

    When no relevant history is available the planner falls back to the
    :class:`RuleBasedPlanner`.

    Parameters
    ----------
    trace_history:
        Initial list of past traces.  May be empty; new traces can be
        added via :meth:`record_trace`.
    default_budget:
        Budget passed to the fallback :class:`RuleBasedPlanner`.
    similarity_threshold:
        Minimum fraction of shared tokens between an incoming query and
        a historical query for the trace to be considered *relevant*.
    """

    def __init__(
        self,
        trace_history: Optional[List[MemoryTrace]] = None,
        default_budget: int = 5,
        similarity_threshold: float = 0.3,
    ) -> None:
        self._traces: List[MemoryTrace] = list(trace_history or [])
        self._fallback = RuleBasedPlanner(default_budget=default_budget)
        self._similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_trace(self, trace: MemoryTrace) -> None:
        """Append a completed trace to the history.

        Parameters
        ----------
        trace:
            A :class:`MemoryTrace` instance with a populated ``outcome``
            field (``"success"``, ``"failure"``, or ``"unknown"``).
        """
        self._traces.append(trace)

    @property
    def trace_history(self) -> List[MemoryTrace]:
        """Read-only view of the current trace history."""
        return list(self._traces)

    def plan(
        self,
        query: str,
        task: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> RetrievalPlan:
        """Produce a plan informed by historical trace outcomes.

        Falls back to :class:`RuleBasedPlanner` if there are no relevant
        historical traces for the given query.
        """
        stripped = query.strip()
        if len(stripped) < 5:
            return RetrievalPlan(
                should_retrieve=False,
                memory_types=[],
                budget=0,
                priority_order=[],
                reasoning=(
                    f"Query too short ({len(stripped)} chars) — skipping retrieval."
                ),
            )

        relevant_traces = self._find_relevant_traces(stripped)

        if not relevant_traces:
            fallback_plan = self._fallback.plan(query, task, context)
            fallback_plan.reasoning = (
                "No relevant trace history — falling back to rule-based planner. "
                + fallback_plan.reasoning
            )
            return fallback_plan

        # Analyse relevant traces
        type_scores = self._score_memory_types(relevant_traces)
        budget = self._compute_adaptive_budget(relevant_traces)

        # Build priority order from type scores (descending)
        scored_types = sorted(type_scores.items(), key=lambda kv: kv[1], reverse=True)
        priority_order = [t for t, _s in scored_types if _s > 0]

        # Ensure we have at least the fallback types when scores are all zero
        if not priority_order:
            fallback_plan = self._fallback.plan(query, task, context)
            priority_order = fallback_plan.priority_order

        memory_types = list(dict.fromkeys(priority_order))

        reasoning = (
            f"Adaptive plan based on {len(relevant_traces)} relevant trace(s). "
            f"Type scores: {dict(scored_types)}. "
            f"Adaptive budget: {budget}."
        )

        return RetrievalPlan(
            should_retrieve=True,
            memory_types=memory_types,
            budget=budget,
            priority_order=priority_order,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_relevant_traces(self, query: str) -> List[MemoryTrace]:
        """Return traces whose query is sufficiently similar to *query*."""
        query_tokens = RuleBasedPlanner._tokenize(query)
        if not query_tokens:
            return []

        relevant: List[MemoryTrace] = []
        for trace in self._traces:
            trace_tokens = RuleBasedPlanner._tokenize(trace.query)
            if not trace_tokens:
                continue
            overlap = len(query_tokens & trace_tokens)
            union = len(query_tokens | trace_tokens)
            similarity = overlap / union if union > 0 else 0.0
            if similarity >= self._similarity_threshold:
                relevant.append(trace)

        return relevant

    @staticmethod
    def _score_memory_types(
        traces: List[MemoryTrace],
    ) -> Dict[str, float]:
        """Score each memory type based on outcome frequency in *traces*.

        Positive outcomes add ``+1``, failures subtract ``0.5``, and
        unknown outcomes add ``+0.1`` (weak positive signal).
        """
        scores: Dict[str, float] = {}

        for trace in traces:
            if trace.plan is None:
                continue
            outcome_delta: float
            if trace.outcome == "success":
                outcome_delta = 1.0
            elif trace.outcome == "failure":
                outcome_delta = -0.5
            else:
                outcome_delta = 0.1  # unknown / neutral

            for mem_type in trace.plan.memory_types:
                scores[mem_type] = scores.get(mem_type, 0.0) + outcome_delta

        return scores

    @staticmethod
    def _compute_adaptive_budget(traces: List[MemoryTrace]) -> int:
        """Compute a budget based on how efficiently past retrievals were used.

        Efficiency is measured as the ratio of *injected* results to
        *retrieved* results across all relevant traces.  If most
        retrieved memories are being ignored (low injection rate), the
        budget is reduced.

        Returns a budget in the range [2, 10].
        """
        total_retrieved = 0
        total_injected = 0

        for trace in traces:
            total_retrieved += len(trace.retrieved)
            total_injected += len(trace.injected)

        if total_retrieved == 0:
            return 5  # no data — use default

        efficiency = total_injected / total_retrieved  # 0.0 – 1.0+

        # Map efficiency to a budget:
        #   0.0  → 2  (most results ignored, cut budget hard)
        #   0.5  → 5  (half used, default budget)
        #   1.0+ → 10 (everything used, maximise budget)
        budget = int(2 + 8 * min(efficiency, 1.0))
        return max(2, min(budget, 10))


# ======================================================================
# Always-retrieve planner (baseline)
# ======================================================================

class AlwaysRetrievePlanner(RetrievalPlanner):
    """Trivial baseline planner that always retrieves from all memory types.

    Useful for testing, benchmarking, or as a simple default when no
    intelligent planning is needed.

    Parameters
    ----------
    budget:
        Fixed retrieval budget.  Defaults to ``5``.
    """

    def __init__(self, budget: int = 5) -> None:
        self._budget = budget

    def plan(
        self,
        query: str,
        task: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> RetrievalPlan:
        """Return a plan that retrieves from every memory type.

        The only gate is an empty query — if the query is blank,
        ``should_retrieve`` is ``False``.
        """
        if not query.strip():
            return RetrievalPlan(
                should_retrieve=False,
                memory_types=[],
                budget=0,
                priority_order=[],
                reasoning="Empty query — nothing to retrieve.",
            )

        all_types = [mt.value for mt in MemoryType]

        return RetrievalPlan(
            should_retrieve=True,
            memory_types=all_types,
            budget=self._budget,
            priority_order=all_types,
            reasoning=(
                f"AlwaysRetrievePlanner: retrieving from all {len(all_types)} "
                f"memory types with budget={self._budget}."
            ),
        )
