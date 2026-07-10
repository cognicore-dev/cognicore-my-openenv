"""
CogniCore Reflection — Learns from past mistakes to improve future decisions.

Generalized from the AI-Safety-specific ``Reflection`` class.
Works with any ``Memory`` instance and any grouping key.

Features:
  - Analyze failure patterns per group
  - Generate natural-language hints
  - Suggest action overrides with confidence
  - Track override rate statistics
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from cognicore.memory.base import MemoryBackend


class ReflectionEngine:
    """Metacognitive reflection — learns from agent mistakes.

    Tracks which actions fail in which groups and provides hints
    to help the agent avoid repeating the same errors.

    Parameters
    ----------
    memory : Memory
        The memory instance to analyze.
    min_samples : int
        Minimum entries needed before generating hints.
    failure_threshold : int
        Minimum failures of one action type before flagging it.
    """

    def __init__(
        self,
        memory: MemoryBackend,
        min_samples: int = 2,
        failure_threshold: int = 2,
    ) -> None:
        self.memory = memory
        self.min_samples = min_samples
        self.failure_threshold = failure_threshold
        self._suggestion_count = 0
        self._override_count = 0
        self._hints_given = 0

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self, group_value: str) -> Dict[str, Any]:
        """Analyze past performance in a group.

        Returns
        -------
        dict
            Keys: ``n_similar``, ``good_predictions``, ``bad_predictions``,
            ``recommendation``.
        """
        entries = self.memory.get_by_category(category=group_value, top_k=50)

        if not entries:
            return {
                "n_similar": 0,
                "good_predictions": {},
                "bad_predictions": {},
                "recommendation": None,
            }

        good: Dict[str, int] = {}
        bad: Dict[str, int] = {}

        for entry in entries:
            # Use predicted, fall back to action, else empty string
            predicted = str(entry.action or "").strip()
            
            if entry.correct is True:
                good[predicted] = good.get(predicted, 0) + 1
            elif entry.correct is False:
                bad[predicted] = bad.get(predicted, 0) + 1

        # Filter out empty string from good predictions before recommending
        good_filtered = {k: v for k, v in good.items() if k.strip()}
        recommendation = max(good_filtered, key=good_filtered.get) if good_filtered else None

        return {
            "n_similar": len(entries),
            "good_predictions": good,
            "bad_predictions": bad,
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # Hints
    # ------------------------------------------------------------------

    def get_hint(self, group_value: str) -> Optional[str]:
        """Generate a natural-language reflection hint.

        Returns None if not enough data or no actionable pattern.
        """
        analysis = self.analyze(group_value)

        if analysis["n_similar"] < self.min_samples:
            return None

        # Filter out empty-string keys from bad predictions
        bad = {k: v for k, v in analysis["bad_predictions"].items() if k.strip()}
        if not bad:
            return None

        worst_prediction = max(bad, key=bad.get)
        fail_count = bad[worst_prediction]

        if fail_count < self.failure_threshold:
            return None

        self._hints_given += 1

        parts = [
            f"Reflection: In similar '{group_value}' tasks,",
            f"action '{worst_prediction}' was wrong {fail_count} times.",
        ]

        if analysis["recommendation"]:
            parts.append(f"Consider '{analysis['recommendation']}' instead.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Action suggestion / override
    # ------------------------------------------------------------------

    def suggest_override(
        self,
        group_value: str,
        proposed_action: str,
    ) -> Tuple[str, str, float]:
        """Suggest whether to keep or override the agent's action.

        Returns
        -------
        tuple of (final_action, source, confidence)
            ``source`` is ``"agent"`` or ``"reflection_override"``.
            ``confidence`` is 0.0–1.0 indicating how sure the override is.
        """
        self._suggestion_count += 1
        analysis = self.analyze(group_value)
        bad = analysis["bad_predictions"]
        recommendation = analysis["recommendation"]

        if proposed_action in bad and bad[proposed_action] >= self.failure_threshold:
            if recommendation and recommendation != proposed_action:
                self._override_count += 1
                # Confidence = fraction of that group's entries that were correct with this recommendation
                good = analysis["good_predictions"]
                total = analysis["n_similar"]
                confidence = good.get(recommendation, 0) / total if total > 0 else 0.5
                return recommendation, "reflection_override", confidence

        return proposed_action, "agent", 0.5

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def override_rate(self) -> float:
        """Fraction of suggestions that resulted in an override."""
        if self._suggestion_count == 0:
            return 0.0
        return self._override_count / self._suggestion_count

    def stats(self) -> Dict[str, Any]:
        return {
            "total_suggestions": self._suggestion_count,
            "overrides": self._override_count,
            "override_rate": self.override_rate,
            "hints_given": self._hints_given,
        }
