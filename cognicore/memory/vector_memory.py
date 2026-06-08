"""
CogniCore VectorMemory — Category-based memory for AI Safety cases.

Adapted from the FAISS-based VectorMemory in the Colab notebook
to work with text-based safety cases instead of FrozenLake grid states.

Uses category-based similarity matching (lightweight, no FAISS dependency)
which is ideal for the 30-case dataset and HuggingFace Spaces deployment.
"""

import time
from typing import List, Dict, Any


class VectorMemory:
    """Memory system that stores and retrieves past safety classifications.

    For the AI Safety Monitor, "similar" cases are those in the same
    category (e.g., all malware cases, all privacy cases). This enables
    the agent to learn from past classifications in the same domain.
    """

    def __init__(self, max_size: int = 10000):
        self.entries: List[Dict[str, Any]] = []
        self.max_size = max_size
        self._stats = {
            "total_stored": 0,
            "total_retrieved": 0,
        }

    def store(
        self,
        case_id: str,
        category: str,
        predicted: str,
        ground_truth: str,
        reward: float,
        correct: bool,
        episode: int = 0,
    ) -> None:
        """Store a classification result in memory.

        Args:
            case_id: Unique identifier for the case.
            category: Category of the case (e.g., "malware", "privacy").
            predicted: Agent's predicted label.
            ground_truth: Correct label.
            reward: Reward received.
            correct: Whether the classification was correct.
            episode: Episode number.
        """
        entry = {
            "case_id": case_id,
            "category": category,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "reward": reward,
            "correct": correct,
            "episode": episode,
            "timestamp": time.time(),
        }
        self.entries.append(entry)
        self._stats["total_stored"] += 1

        # Enforce max size (FIFO eviction)
        if len(self.entries) > self.max_size:
            self.entries.pop(0)

    def retrieve(self, category: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve the most recent similar cases by category.

        Args:
            category: Category to match against.
            top_k: Maximum number of results to return.

        Returns:
            List of memory entries from the same category,
            most recent first.
        """
        self._stats["total_retrieved"] += 1
        keys_to_check = {"category", "group", "type"}
        similar = [
            e for e in self.entries
            if any(e.get(k) == category for k in keys_to_check)
        ]
        # Return most recent first
        return similar[-top_k:][::-1]

    def retrieve_successes(self, category: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve successful past classifications in this category."""
        keys_to_check = {"category", "group", "type"}
        successes = [
            e for e in self.entries
            if any(e.get(k) == category for k in keys_to_check) and e.get("correct")
        ]
        return successes[-top_k:][::-1]

    def retrieve_failures(self, category: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve failed past classifications in this category."""
        keys_to_check = {"category", "group", "type"}
        failures = [
            e for e in self.entries
            if any(e.get(k) == category for k in keys_to_check) and not e.get("correct")
        ]
        return failures[-top_k:][::-1]

    def get_context_for_observation(
        self, category: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Get memory context formatted for the observation.

        Returns a simplified view suitable for the agent observation.
        """
        recent = self.retrieve(category, top_k)
        context = []
        for entry in recent:
            context.append(
                {
                    "case_id": entry["case_id"],
                    "predicted": entry["predicted"],
                    "ground_truth": entry["ground_truth"],
                    "was_correct": entry["correct"],
                }
            )
        return context

    def stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        total = len(self.entries)
        successes = sum(1 for e in self.entries if e["correct"])
        failures = total - successes

        return {
            "total_entries": total,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / total if total > 0 else 0.0,
            "total_stored": self._stats["total_stored"],
            "total_retrieved": self._stats["total_retrieved"],
            "groups": list(set(e["category"] for e in self.entries)),
        }

    def clear(self) -> None:
        """Clear all memory entries."""
        self.entries.clear()
