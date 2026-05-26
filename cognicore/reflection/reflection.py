"""
CogniCore Reflection — Analyzes past patterns to improve future decisions.

Adapted from the Colab notebook's Reflection class. In the AI Safety domain,
reflection tracks which categories the agent struggles with and provides
hints when it makes repeated mistakes.
"""

from typing import Dict, Any, Optional, Tuple
from cognicore.memory.vector_memory import VectorMemory


class Reflection:
    """Reflection engine that learns from past classification mistakes.

    Tracks failure patterns by category and provides actionable hints
    when the agent repeatedly misclassifies cases in the same domain.
    """

    def __init__(self, memory: VectorMemory):
        self.memory = memory
        self._suggestion_count = 0
        self._override_count = 0

    def analyze(self, category: str) -> Dict[str, Any]:
        """Analyze past performance in a given category.

        Args:
            category: The case category to analyze.

        Returns:
            Analysis dict with success/failure patterns.
        """
        entries = self.memory.retrieve(category, top_k=50)

        if not entries:
            return {
                "n_similar": 0,
                "good_predictions": {},
                "bad_predictions": {},
                "recommendation": None,
            }

        good = {}
        bad = {}

        for entry in entries:
            predicted = entry["predicted"]
            if entry["correct"]:
                good[predicted] = good.get(predicted, 0) + 1
            else:
                bad[predicted] = bad.get(predicted, 0) + 1

        # Find most common correct prediction for this category
        recommendation = None
        if good:
            recommendation = max(good, key=good.get)

        return {
            "n_similar": len(entries),
            "good_predictions": good,
            "bad_predictions": bad,
            "recommendation": recommendation,
        }

    def suggest_action(self, category: str, model_prediction: str) -> Tuple[str, str]:
        """Suggest whether to keep or override the model's prediction.

        Args:
            category: Category of the current case.
            model_prediction: The model's proposed classification.

        Returns:
            Tuple of (final_prediction, source) where source is
            either "model_action" or "reflection_override".
        """
        self._suggestion_count += 1

        analysis = self.analyze(category)
        bad = analysis["bad_predictions"]
        recommendation = analysis["recommendation"]

        # If model's prediction has been wrong frequently in this category
        if model_prediction in bad and bad[model_prediction] >= 2:
            # And we have a better recommendation
            if recommendation and recommendation != model_prediction:
                self._override_count += 1
                return recommendation, "reflection_override"

        return model_prediction, "model_action"

    def get_reflection_hint(self, category: str) -> Optional[str]:
        """Generate a natural-language reflection hint for the agent.

        Returns None if no useful hint is available.
        """
        analysis = self.analyze(category)

        if analysis["n_similar"] < 2:
            return None

        bad = analysis["bad_predictions"]
        if not bad:
            return None

        worst_prediction = max(bad, key=bad.get)
        # Skip empty action names
        if not worst_prediction.strip():
            non_empty = {k: v for k, v in bad.items() if k.strip()}
            if not non_empty:
                return None
            worst_prediction = max(non_empty, key=non_empty.get)
            fail_count = non_empty[worst_prediction]
        else:
            fail_count = bad[worst_prediction]

        if fail_count < 2:
            return None

        hint_parts = [
            f"Reflection: In similar '{category}' cases,",
            f"predicting '{worst_prediction}' was wrong {fail_count} times.",
        ]

        if analysis["recommendation"]:
            hint_parts.append(f"Consider '{analysis['recommendation']}' instead.")

        return " ".join(hint_parts)

    @property
    def override_rate(self) -> float:
        """Fraction of suggestions that were overrides."""
        if self._suggestion_count == 0:
            return 0.0
        return self._override_count / self._suggestion_count

    def stats(self) -> Dict[str, Any]:
        """Return reflection statistics."""
        return {
            "total_suggestions": self._suggestion_count,
            "overrides": self._override_count,
            "override_rate": self.override_rate,
        }
