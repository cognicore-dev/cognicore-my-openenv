"""
CogniCore Memory — Generalized episodic memory for any environment.

Evolved from the AI-Safety-specific ``VectorMemory`` into a
domain-agnostic memory system.  Entries are arbitrary dicts;
similarity is determined by a configurable *grouping key*.

Features:
  - Store / retrieve / filter by success or failure
  - FIFO eviction when max size is reached
  - JSON persistence for cross-episode learning
  - Statistics tracking
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List


class Memory:
    """Domain-agnostic episodic memory.

    Parameters
    ----------
    max_size : int
        Maximum entries before FIFO eviction kicks in.
    similarity_key : str
        The dict key used to group "similar" entries (e.g. ``"category"``).
    """

    def __init__(
        self,
        max_size: int = 10_000,
        similarity_key: str = "category",
    ) -> None:
        self.entries: List[Dict[str, Any]] = []
        self.max_size = max_size
        self.similarity_key = similarity_key
        self._stats = {
            "total_stored": 0,
            "total_retrieved": 0,
        }

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(self, entry: Dict[str, Any], *, episode: int = 0) -> None:
        """Store an experience entry in memory.

        The entry must contain at least the ``similarity_key`` field and
        a ``"correct"`` boolean.  Everything else is free-form.

        Parameters
        ----------
        entry : dict
            Arbitrary key-value data.  Must include the similarity key
            and ``"correct"``.
        episode : int
            Current episode number (auto-stamped).
        """
        entry = {**entry, "episode": episode, "_timestamp": time.time()}
        self.entries.append(entry)
        self._stats["total_stored"] += 1

        # FIFO eviction
        if len(self.entries) > self.max_size:
            self.entries.pop(0)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        group_value: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most recent entries sharing the same group value.

        Checks both the configured ``similarity_key`` and common aliases
        (``category``, ``group``, ``type``) so entries are found regardless
        of which key the caller used when storing.

        Parameters
        ----------
        group_value : str
            Value to match.
        top_k : int
            Maximum number of results (most recent first).
        """
        self._stats["total_retrieved"] += 1
        # Check primary key and common aliases
        keys_to_check = {self.similarity_key, "category", "group", "type"}
        similar = [
            e for e in self.entries
            if any(e.get(k) == group_value for k in keys_to_check)
        ]
        return similar[-top_k:][::-1]

    def retrieve_successes(
        self, group_value: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve successful (correct=True) entries in a group."""
        keys_to_check = {self.similarity_key, "category", "group", "type"}
        successes = [
            e
            for e in self.entries
            if any(e.get(k) == group_value for k in keys_to_check) and e.get("correct")
        ]
        return successes[-top_k:][::-1]

    def retrieve_failures(
        self, group_value: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve failed (correct=False) entries in a group."""
        keys_to_check = {self.similarity_key, "category", "group", "type"}
        failures = [
            e
            for e in self.entries
            if any(e.get(k) == group_value for k in keys_to_check) and not e.get("correct")
        ]
        return failures[-top_k:][::-1]

    def get_context(self, group_value: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return a simplified context for agent observations.

        Strips internal fields (``_timestamp``, ``episode``) and returns
        a clean list suitable for injecting into observations.
        """
        recent = self.retrieve(group_value, top_k)
        context = []
        for entry in recent:
            ctx = {
                k: v
                for k, v in entry.items()
                if not k.startswith("_") and k != "episode"
            }
            context.append(ctx)
        return context

    def has_seen_group(self, group_value: str) -> bool:
        """Return True if memory has any entries for this group."""
        keys_to_check = {self.similarity_key, "category", "group", "type"}
        return any(
            any(e.get(k) == group_value for k in keys_to_check)
            for e in self.entries
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist memory to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": self.entries,
            "max_size": self.max_size,
            "similarity_key": self.similarity_key,
            "stats": self._stats,
        }
        path.write_text(json.dumps(data, indent=2, default=str))

    def load(self, path: str | Path) -> None:
        """Load memory from a JSON file."""
        path = Path(path)
        if not path.exists():
            return
        data = json.loads(path.read_text())
        self.entries = data.get("entries", [])
        self._stats = data.get("stats", self._stats)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        total = len(self.entries)
        successes = sum(1 for e in self.entries if e.get("correct"))
        failures = total - successes

        groups = set(
            e.get(self.similarity_key)
            for e in self.entries
            if e.get(self.similarity_key) is not None
        )

        return {
            "total_entries": total,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / total if total > 0 else 0.0,
            "groups": sorted(groups),
            "total_stored": self._stats["total_stored"],
            "total_retrieved": self._stats["total_retrieved"],
        }

    def clear(self) -> None:
        """Clear all entries."""
        self.entries.clear()

    def export_jsonl(self, output_path: str) -> int:
        """Export memory entries as JSONL for research / training data.

        Strips internal fields (``_*``) and writes one JSON object per line.

        Returns
        -------
        int
            Number of entries exported.
        """
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for entry in self.entries:
                clean = {k: v for k, v in entry.items() if not k.startswith("_")}
                f.write(json.dumps(clean, default=str) + "\n")
                count += 1

        return count
