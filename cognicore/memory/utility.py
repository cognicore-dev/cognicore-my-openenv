"""Utility Scoring Engine for the CogniCore Agent Memory Operating System.

This module provides the :class:`UtilityScorer`, which tracks how useful each
memory entry is over its lifetime.  It answers three key questions:

1. **How useful is this memory?**  — weighted combination of retrievals, uses,
   positive/negative outcomes, and ignores, normalised per-retrieval.
2. **Is this memory causing negative transfer?** — when a disproportionate
   share of uses lead to worse outcomes, the memory is flagged.
3. **Should this memory be promoted or demoted?** — lifecycle recommendations
   (candidate → active → verified → archived) driven purely by evidence.

Zero external dependencies — stdlib only.

Version 2.0 — Agent Memory Operating System
"""

from __future__ import annotations

import time
from typing import List, Tuple

from cognicore.memory.base import MemoryEntry, MemoryState, MemoryType


# ======================================================================
# Utility Scoring Engine
# ======================================================================


class UtilityScorer:
    """Tracks and computes memory utility scores.

    The scorer maintains a set of configurable weights that translate raw
    counters on :class:`MemoryEntry` into a single normalised utility score
    in the range ``[-1.0, 1.0]``.  Higher scores mean the memory is proving
    useful; scores below zero suggest the memory is hurting performance.

    Parameters
    ----------
    w_used : float
        Weight applied to each *use* of the memory (regardless of outcome).
    w_positive : float
        Bonus weight for each positive outcome.
    w_negative : float
        Penalty weight for each negative outcome (should be negative).
    w_ignored : float
        Penalty weight for each time the memory was retrieved but ignored
        (should be negative or zero).
    negative_transfer_threshold : float
        Ratio of ``negative_outcomes / max(1, used_count)`` above which
        the memory is considered to cause negative transfer.
    min_retrievals_for_judgment : int
        Minimum number of retrievals before negative-transfer detection
        or decay-based archival kicks in.
    promotion_threshold : float
        Utility score at or above which an active memory is recommended
        for promotion to *verified*.
    decay_threshold : float
        Utility score below which a sufficiently-retrieved memory is
        recommended for archival.

    Examples
    --------
    >>> scorer = UtilityScorer()
    >>> entry = MemoryEntry(text="Python uses 0-based indexing.")
    >>> scorer.on_retrieval(entry)
    >>> scorer.on_used(entry, positive=True)
    >>> scorer.compute_utility(entry)
    3.0  # clamped to 1.0
    """

    def __init__(
        self,
        w_used: float = 1.0,
        w_positive: float = 2.0,
        w_negative: float = -1.5,
        w_ignored: float = -0.3,
        negative_transfer_threshold: float = 0.5,
        min_retrievals_for_judgment: int = 3,
        promotion_threshold: float = 0.6,
        decay_threshold: float = -0.3,
    ) -> None:
        self.w_used: float = w_used
        self.w_positive: float = w_positive
        self.w_negative: float = w_negative
        self.w_ignored: float = w_ignored
        self.negative_transfer_threshold: float = negative_transfer_threshold
        self.min_retrievals_for_judgment: int = min_retrievals_for_judgment
        self.promotion_threshold: float = promotion_threshold
        self.decay_threshold: float = decay_threshold

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def on_retrieval(self, entry: MemoryEntry) -> None:
        """Record that *entry* was retrieved by the retrieval pipeline.

        Increments :pyattr:`MemoryEntry.retrieval_count` and updates
        :pyattr:`MemoryEntry.last_accessed` to the current wall-clock time.

        Parameters
        ----------
        entry : MemoryEntry
            The memory entry that was just retrieved.
        """
        entry.retrieval_count += 1
        entry.last_accessed = time.time()

    def on_used(self, entry: MemoryEntry, positive: bool = True) -> None:
        """Record that *entry* was actually used by the agent.

        Increments :pyattr:`MemoryEntry.used_count`.  If *positive* is
        ``True`` the entry's :pyattr:`positive_outcomes` counter is bumped;
        otherwise :pyattr:`negative_outcomes` is incremented.  The entry's
        :pyattr:`utility_score` is recalculated automatically.

        Parameters
        ----------
        entry : MemoryEntry
            The memory entry that was used.
        positive : bool
            Whether the usage led to a positive outcome.
        """
        entry.used_count += 1
        if positive:
            entry.positive_outcomes += 1
        else:
            entry.negative_outcomes += 1
        entry.utility_score = self.compute_utility(entry)

    def on_ignored(self, entry: MemoryEntry) -> None:
        """Record that *entry* was retrieved but ultimately ignored.

        Increments :pyattr:`MemoryEntry.ignored_count` and recalculates
        :pyattr:`utility_score`.

        Parameters
        ----------
        entry : MemoryEntry
            The memory entry that was ignored after retrieval.
        """
        entry.ignored_count += 1
        entry.utility_score = self.compute_utility(entry)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute_utility(self, entry: MemoryEntry) -> float:
        """Compute the normalised utility score for *entry*.

        The formula is a weighted sum of the entry's usage counters,
        normalised by its total retrieval count:

        .. math::

            U = \\frac{
                \\text{used} \\cdot w_{\\text{used}}
                + \\text{pos} \\cdot w_{\\text{pos}}
                + \\text{neg} \\cdot w_{\\text{neg}}
                + \\text{ign} \\cdot w_{\\text{ign}}
            }{\\text{retrievals}}

        The result is clamped to the interval ``[-1.0, 1.0]``.

        Parameters
        ----------
        entry : MemoryEntry
            The memory entry to score.

        Returns
        -------
        float
            Utility score in ``[-1.0, 1.0]``.  Returns ``0.0`` when
            ``retrieval_count`` is zero (no evidence yet).
        """
        if entry.retrieval_count == 0:
            return 0.0

        raw = (
            entry.used_count * self.w_used
            + entry.positive_outcomes * self.w_positive
            + entry.negative_outcomes * self.w_negative
            + entry.ignored_count * self.w_ignored
        ) / entry.retrieval_count

        # Clamp to [-1.0, 1.0]
        return max(-1.0, min(1.0, raw))

    # ------------------------------------------------------------------
    # Negative-transfer detection
    # ------------------------------------------------------------------

    def detect_negative_transfer(self, entry: MemoryEntry) -> bool:
        """Detect whether *entry* is causing negative transfer.

        A memory exhibits negative transfer when it has been retrieved
        enough times (≥ ``min_retrievals_for_judgment``) **and** the ratio
        of negative outcomes to total uses exceeds the configured
        ``negative_transfer_threshold``.

        Parameters
        ----------
        entry : MemoryEntry
            The memory entry to evaluate.

        Returns
        -------
        bool
            ``True`` if the entry is likely causing negative transfer.
        """
        if entry.retrieval_count < self.min_retrievals_for_judgment:
            return False

        negative_ratio = entry.negative_outcomes / max(1, entry.used_count)
        return negative_ratio > self.negative_transfer_threshold

    # ------------------------------------------------------------------
    # Lifecycle recommendations
    # ------------------------------------------------------------------

    def get_promotion_recommendation(self, entry: MemoryEntry) -> str:
        """Return the recommended :class:`MemoryState` value for *entry*.

        The recommendation is determined by the following priority rules
        (first match wins):

        1. **Negative transfer detected** → ``'archived'``
        2. **High utility & currently active** → ``'verified'``
        3. **Non-negative utility & at least one retrieval & candidate**
           → ``'active'``
        4. **Low utility with sufficient evidence** → ``'archived'``
        5. **Otherwise** → current state (no change)

        Parameters
        ----------
        entry : MemoryEntry
            The memory entry to evaluate.

        Returns
        -------
        str
            One of the :class:`MemoryState` string values (``'candidate'``,
            ``'active'``, ``'verified'``, ``'archived'``, ``'deleted'``).
        """
        # Rule 1: negative transfer → archive immediately
        if self.detect_negative_transfer(entry):
            return MemoryState.ARCHIVED.value

        # Rule 2: high-utility active memory → promote to verified
        if (
            entry.utility_score >= self.promotion_threshold
            and entry.state == MemoryState.ACTIVE.value
        ):
            return MemoryState.VERIFIED.value

        # Rule 3: non-negative utility with evidence → promote candidate
        if (
            entry.utility_score >= 0
            and entry.retrieval_count >= 1
            and entry.state == MemoryState.CANDIDATE.value
        ):
            return MemoryState.ACTIVE.value

        # Rule 4: decayed utility with enough evidence → archive
        if (
            entry.utility_score < self.decay_threshold
            and entry.retrieval_count >= self.min_retrievals_for_judgment
        ):
            return MemoryState.ARCHIVED.value

        # Rule 5: no change
        return entry.state

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def batch_score(
        self, entries: List[MemoryEntry]
    ) -> List[Tuple[MemoryEntry, float, str]]:
        """Score a batch of entries and return lifecycle recommendations.

        For each entry the method computes the current utility score and
        determines the recommended lifecycle state.

        Parameters
        ----------
        entries : List[MemoryEntry]
            Memory entries to evaluate.

        Returns
        -------
        List[Tuple[MemoryEntry, float, str]]
            A list of ``(entry, utility_score, recommended_state)`` tuples,
            one per input entry, in the same order.
        """
        results: List[Tuple[MemoryEntry, float, str]] = []
        for entry in entries:
            score = self.compute_utility(entry)
            recommendation = self.get_promotion_recommendation(entry)
            results.append((entry, score, recommendation))
        return results
