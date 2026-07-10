"""Memory Evaluation Layer for the CogniCore Agent Memory Operating System.

This module determines whether an incoming observation should be persisted as
long-term memory or discarded as a transient log entry.  Three concrete
evaluators are provided:

* **RuleBasedEvaluator** — zero-dependency, pattern-matching heuristics.
* **LLMEvaluator** — delegates to an external LLM callable with automatic
  fallback to rule-based evaluation on failure.
* **HybridEvaluator** — rule-based first, LLM only for ambiguous cases.

All evaluators conform to the abstract ``MemoryEvaluator`` interface and
produce an ``EvaluationResult`` that downstream components (e.g. the Memory
Manager) consume to decide storage, importance, and classification.

Version 2.0 — Agent Memory Operating System
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from cognicore.memory.base import EvaluationResult, MemoryType


# ======================================================================
# Signal Definitions
# ======================================================================

# Each signal group maps a MemoryType to a tuple of:
#   (list-of-trigger-phrases, importance, creation_reason)
# Ordering matters: earlier groups take priority when multiple patterns
# match (preference before constraint, etc.).

_SIGNAL_GROUPS: List[Tuple[MemoryType, List[str], float, str]] = [
    (
        MemoryType.PREFERENCE,
        [
            "i prefer", "i like", "i want", "my favorite",
            "i'd rather", "please always", "please never",
        ],
        0.7,
        "Detected preference signal",
    ),
    (
        MemoryType.CONSTRAINT,
        [
            "must", "always", "never", "don't", "do not",
            "required", "forbidden", "rule:", "constraint:",
        ],
        0.9,
        "Detected constraint signal",
    ),
    (
        MemoryType.FAILURE,
        [
            "error", "failed", "bug", "crash",
            "mistake", "wrong", "incorrect", "broken",
        ],
        0.8,
        "Detected failure signal",
    ),
    (
        MemoryType.PROCEDURAL,
        [
            "steps:", "how to", "procedure",
            "recipe", "workflow", "to do this",
        ],
        0.7,
        "Detected procedural signal",
    ),
    (
        MemoryType.REFLECTION,
        [
            "i notice", "pattern", "insight",
            "observation", "i realized", "lesson learned",
        ],
        0.6,
        "Detected reflection signal",
    ),
    (
        MemoryType.KNOWLEDGE,
        [
            "definition:", "means", "is defined as", "refers to",
        ],
        0.6,
        "Detected knowledge signal",
    ),
]


# ======================================================================
# Abstract Evaluator
# ======================================================================

class MemoryEvaluator(ABC):
    """Abstract base class for all memory evaluators.

    Every evaluator must implement :meth:`evaluate`, which inspects an
    incoming piece of text (plus optional context metadata) and returns
    an :class:`EvaluationResult` indicating whether the text should be
    stored, what type it is, and how important it is.
    """

    @abstractmethod
    def evaluate(self, text: str, context: Dict[str, Any]) -> EvaluationResult:
        """Evaluate whether *text* should become persistent memory.

        Parameters
        ----------
        text:
            The raw observation / message / user input to evaluate.
        context:
            Arbitrary metadata that may influence the decision (e.g.
            ``{"session_id": "abc", "agent": "planner"}``).

        Returns
        -------
        EvaluationResult
            A dataclass with ``should_store``, ``memory_type``,
            ``importance``, ``creation_reason``, and ``confidence``.
        """
        ...


# ======================================================================
# Rule-Based Evaluator
# ======================================================================

class RuleBasedEvaluator(MemoryEvaluator):
    """Zero-dependency, pattern-matching memory evaluator.

    Scans the input text (case-insensitively) for predefined signal
    phrases that map to specific :class:`MemoryType` values.  If no
    signal matches, the evaluator falls back to a length-based
    heuristic:

    * Text shorter than 10 characters → **not stored**.
    * Text longer than 20 characters → stored as ``SEMANTIC`` with low
      importance (0.3).
    * Text between 10 and 20 characters with no signal → **not stored**.

    Signal groups are checked in priority order so that, for example, a
    preference signal is detected before a generic constraint keyword
    like ``"always"`` that might also appear in preference phrases.
    """

    # Minimum character thresholds
    _MIN_STORE_LEN: int = 10
    _DEFAULT_STORE_LEN: int = 20

    def evaluate(self, text: str, context: Dict[str, Any]) -> EvaluationResult:
        """Evaluate *text* against rule-based signal patterns.

        Parameters
        ----------
        text:
            The raw observation to evaluate.
        context:
            Additional metadata (unused by this evaluator but accepted
            for interface conformance).

        Returns
        -------
        EvaluationResult
        """
        # Very short texts are never stored.
        if len(text) < self._MIN_STORE_LEN:
            return EvaluationResult(
                should_store=False,
                memory_type=MemoryType.SEMANTIC.value,
                importance=0.0,
                creation_reason="Text too short to store",
                confidence=1.0,
            )

        text_lower = text.lower()

        # Walk signal groups in priority order.
        for memory_type, phrases, importance, reason in _SIGNAL_GROUPS:
            for phrase in phrases:
                if phrase in text_lower:
                    return EvaluationResult(
                        should_store=True,
                        memory_type=memory_type.value,
                        importance=importance,
                        creation_reason=reason,
                        confidence=0.8,
                    )

        # No signal matched — fall back to length heuristic.
        if len(text) > self._DEFAULT_STORE_LEN:
            return EvaluationResult(
                should_store=True,
                memory_type=MemoryType.SEMANTIC.value,
                importance=0.3,
                creation_reason="No signal detected; stored as generic semantic memory",
                confidence=0.4,
            )

        # Between _MIN_STORE_LEN and _DEFAULT_STORE_LEN with no signal.
        return EvaluationResult(
            should_store=False,
            memory_type=MemoryType.SEMANTIC.value,
            importance=0.0,
            creation_reason="Text too short and no signal detected",
            confidence=0.6,
        )


# ======================================================================
# LLM-Backed Evaluator
# ======================================================================

_LLM_PROMPT_TEMPLATE: str = (
    "You are a memory evaluation component.  Analyze the following text "
    "and decide whether it should be stored as persistent memory.\n\n"
    "TEXT:\n{text}\n\n"
    "Should this be stored as memory?  If yes, what type?  How important "
    "(0-1)?\n\n"
    "Reply with EXACTLY one JSON object on a single line:\n"
    '{{"should_store": true/false, "memory_type": "<type>", '
    '"importance": <float>, "reason": "<short reason>"}}\n\n'
    "Valid memory types: semantic, episodic, procedural, preference, "
    "constraint, failure, reflection, knowledge."
)


class LLMEvaluator(MemoryEvaluator):
    """LLM-backed memory evaluator with automatic rule-based fallback.

    Constructs a structured prompt, sends it to the supplied ``llm_fn``
    callable, and parses the JSON response.  If ``llm_fn`` is ``None``
    or raises an exception, the evaluator silently degrades to
    :class:`RuleBasedEvaluator`.

    Parameters
    ----------
    llm_fn:
        A callable ``(prompt: str) -> str`` that returns the LLM's raw
        text response.  May be ``None`` for pure-fallback operation.
    """

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None) -> None:
        self._llm_fn = llm_fn
        self._fallback = RuleBasedEvaluator()

    def evaluate(self, text: str, context: Dict[str, Any]) -> EvaluationResult:
        """Evaluate *text* via LLM, falling back to rules on failure.

        Parameters
        ----------
        text:
            The raw observation to evaluate.
        context:
            Additional metadata forwarded to the fallback evaluator
            when the LLM path is unavailable.

        Returns
        -------
        EvaluationResult
        """
        if self._llm_fn is None:
            return self._fallback.evaluate(text, context)

        try:
            prompt = _LLM_PROMPT_TEMPLATE.format(text=text)
            raw_response = self._llm_fn(prompt)
            return self._parse_llm_response(raw_response)
        except Exception:
            # Any failure — network, JSON parse, unexpected schema —
            # degrades gracefully to rule-based evaluation.
            return self._fallback.evaluate(text, context)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_llm_response(raw: str) -> EvaluationResult:
        """Parse the LLM's JSON response into an ``EvaluationResult``.

        The parser is intentionally lenient: it searches for the first
        ``{...}`` block in the response, so models that add preamble or
        trailing commentary still produce usable output.

        Raises
        ------
        ValueError
            If no valid JSON object can be extracted from *raw*.
        """
        # Extract the first JSON object from the response.
        match = re.search(r"\{[^}]+\}", raw)
        if match is None:
            raise ValueError(f"No JSON object found in LLM response: {raw!r}")

        data: Dict[str, Any] = json.loads(match.group())

        should_store = bool(data.get("should_store", False))

        # Validate and normalise memory_type.
        raw_type = str(data.get("memory_type", "semantic")).lower()
        memory_type = MemoryType.from_string(raw_type)

        importance = float(data.get("importance", 0.5))
        importance = max(0.0, min(1.0, importance))  # clamp to [0, 1]

        reason = str(data.get("reason", "LLM evaluation"))

        return EvaluationResult(
            should_store=should_store,
            memory_type=memory_type.value,
            importance=importance,
            creation_reason=reason,
            confidence=0.7,
        )


# ======================================================================
# Hybrid Evaluator
# ======================================================================

class HybridEvaluator(MemoryEvaluator):
    """Two-stage evaluator combining rules and LLM intelligence.

    **Strategy:**

    1. Run :class:`RuleBasedEvaluator` first.
    2. If the rule-based result says ``should_store=True`` *and*
       ``confidence >= 0.7``, accept it immediately — the signal is
       strong enough to skip the (potentially slow/expensive) LLM call.
    3. Otherwise, delegate to :class:`LLMEvaluator` for ambiguous or
       borderline cases.

    Parameters
    ----------
    llm_fn:
        Optional LLM callable forwarded to the internal
        :class:`LLMEvaluator`.  When ``None``, the LLM stage degrades
        to another round of rule-based evaluation (effectively making
        this evaluator equivalent to a pure :class:`RuleBasedEvaluator`).
    confidence_threshold:
        Minimum confidence from the rule-based stage to accept the
        result without LLM confirmation.  Defaults to ``0.7``.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        confidence_threshold: float = 0.7,
    ) -> None:
        self._rule_evaluator = RuleBasedEvaluator()
        self._llm_evaluator = LLMEvaluator(llm_fn)
        self._confidence_threshold = confidence_threshold

    def evaluate(self, text: str, context: Dict[str, Any]) -> EvaluationResult:
        """Evaluate *text* with a two-stage rule → LLM pipeline.

        Parameters
        ----------
        text:
            The raw observation to evaluate.
        context:
            Additional metadata forwarded to both evaluators.

        Returns
        -------
        EvaluationResult
        """
        rule_result = self._rule_evaluator.evaluate(text, context)

        # Fast path: high-confidence rule match.
        if rule_result.should_store and rule_result.confidence >= self._confidence_threshold:
            return rule_result

        # Ambiguous — ask the LLM for a second opinion.
        return self._llm_evaluator.evaluate(text, context)
