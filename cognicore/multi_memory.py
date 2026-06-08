"""
CogniCore Multi-Memory — Human-like multi-tier memory architecture.

Separates memory into 4 tiers like human cognition:
  - Working memory (short-term, current task, ~7 items)
  - Episodic memory (specific experiences with context)
  - Semantic memory (general knowledge distilled from experience)
  - Procedural memory (learned action patterns)

Usage::

    from cognicore.multi_memory import CognitiveMemory

    mem = CognitiveMemory()
    mem.perceive("phishing email", category="security", correct=False, action="SAFE")
    context = mem.recall("suspicious email")
    # -> working memory + relevant episodic + semantic knowledge + action patterns
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("cognicore.multi_memory")


class WorkingMemory:
    """Short-term memory — current task context (~7 items, like human STM)."""

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self._items: deque = deque(maxlen=capacity)

    def push(self, item: Dict[str, Any]) -> None:
        self._items.append({**item, "_wm_time": time.time()})

    def get_all(self) -> List[Dict]:
        return list(self._items)

    def most_recent(self, n: int = 3) -> List[Dict]:
        return list(self._items)[-n:]

    def clear(self) -> None:
        self._items.clear()

    def query_relevant(self, query: str, n: int = 3) -> List[Dict]:
        """Return items most relevant to the query, based on keyword overlap."""
        if not query or not self._items:
            return self.most_recent(n)
        query_tokens = set(query.lower().split())
        scored = []
        for item in self._items:
            text = str(item.get("text", "")) + " " + str(item.get("category", ""))
            item_tokens = set(text.lower().split())
            overlap = len(query_tokens & item_tokens)
            scored.append((item, overlap))
        # Sort by overlap descending, break ties by recency (later = more recent)
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored and scored[0][1] == 0:
            # No overlap at all — fall back to most recent
            return self.most_recent(n)
        return [item for item, _ in scored[:n]]

    @property
    def size(self) -> int:
        return len(self._items)


class EpisodicMemory:
    """Episodic memory — specific experiences with full context.

    Stores rich records of what happened, when, and the outcome.
    """

    def __init__(self, max_size: int = 5000):
        self.entries: List[Dict[str, Any]] = []
        self.max_size = max_size

    def store(self, episode: Dict[str, Any]) -> None:
        self.entries.append(
            {
                **episode,
                "_em_time": time.time(),
                "_em_id": len(self.entries),
            }
        )
        if len(self.entries) > self.max_size:
            self.entries.pop(0)

    def recall_by_category(self, category: str, limit: int = 5) -> List[Dict]:
        matches = [e for e in self.entries if e.get("category") == category]
        return matches[-limit:]

    def recall_failures(self, limit: int = 10) -> List[Dict]:
        return [e for e in self.entries if not e.get("correct", True)][-limit:]

    def recall_successes(self, limit: int = 10) -> List[Dict]:
        return [e for e in self.entries if e.get("correct", False)][-limit:]

    @property
    def size(self) -> int:
        return len(self.entries)


class SemanticKnowledge:
    """Semantic memory — distilled knowledge from experiences.

    Stores generalizations: "phishing = usually UNSAFE", not individual cases.
    """

    def __init__(self):
        self.facts: Dict[str, Dict[str, Any]] = {}
        # {category: {best_action, confidence, times_seen, accuracy}}

    def learn(self, category: str, action: str, correct: bool) -> None:
        if category not in self.facts:
            self.facts[category] = {
                "actions": defaultdict(lambda: {"correct": 0, "wrong": 0}),
                "times_seen": 0,
                "total_correct": 0,
            }

        fact = self.facts[category]
        fact["times_seen"] += 1
        if correct:
            fact["total_correct"] += 1
            fact["actions"][action]["correct"] += 1
        else:
            fact["actions"][action]["wrong"] += 1

    def get_best_action(self, category: str) -> Optional[Tuple[str, float]]:
        """Return (best_action, confidence) for a category."""
        if category not in self.facts:
            return None
        fact = self.facts[category]
        if not fact["actions"]:
            return None

        best_action = None
        best_score = -1
        for action, stats in fact["actions"].items():
            total = stats["correct"] + stats["wrong"]
            if total > 0:
                score = stats["correct"] / total
                if score > best_score:
                    best_score = score
                    best_action = action

        if best_action is None:
            return None
        confidence = best_score * min(
            1.0, fact["times_seen"] / 5
        )  # confidence grows with experience
        return best_action, confidence

    def get_knowledge(self, category: str) -> Optional[Dict]:
        if category not in self.facts:
            return None
        fact = self.facts[category]
        return {
            "category": category,
            "times_seen": fact["times_seen"],
            "accuracy": fact["total_correct"] / fact["times_seen"]
            if fact["times_seen"]
            else 0,
            "best_action": self.get_best_action(category),
        }

    @property
    def categories_known(self) -> int:
        return len(self.facts)


class ProceduralMemory:
    """Procedural memory — learned action patterns (if-then rules).

    Automatically builds rules from experience like:
    "IF category=phishing THEN action=UNSAFE (85% confident)"
    """

    def __init__(self, min_observations: int = 3, min_confidence: float = 0.7):
        self.min_observations = min_observations
        self.min_confidence = min_confidence
        self._patterns: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )  # {category: {action: success_count}}
        self._totals: Dict[str, int] = defaultdict(int)

    def record(self, category: str, action: str, correct: bool) -> None:
        self._totals[category] += 1
        if correct:
            self._patterns[category][action] += 1

    def get_action(self, category: str) -> Optional[Tuple[str, float]]:
        """Get the procedurally learned action for a category.

        Returns None if insufficient data or confidence.
        """
        total = self._totals.get(category, 0)
        if total < self.min_observations:
            return None

        patterns = self._patterns.get(category, {})
        if not patterns:
            return None

        best_action = max(patterns, key=patterns.get)
        confidence = patterns[best_action] / total

        if confidence < self.min_confidence:
            return None

        return best_action, confidence

    def list_rules(self) -> List[Dict]:
        """List all learned procedural rules."""
        rules = []
        for cat in self._patterns:
            result = self.get_action(cat)
            if result:
                action, conf = result
                rules.append(
                    {
                        "rule": f"IF category='{cat}' THEN action='{action}'",
                        "confidence": conf,
                        "observations": self._totals[cat],
                    }
                )
        return sorted(rules, key=lambda r: -r["confidence"])


class CognitiveMemory:
    """Multi-tier cognitive memory system (human-like).

    Integrates 4 memory types and routes decisions based on
    the best available memory signal.
    """

    def __init__(
        self,
        working_capacity: int = 7,
        episodic_max: int = 5000,
        procedural_min_obs: int = 3,
    ):
        self.working = WorkingMemory(capacity=working_capacity)
        self.episodic = EpisodicMemory(max_size=episodic_max)
        self.semantic = SemanticKnowledge()
        self.procedural = ProceduralMemory(min_observations=procedural_min_obs)

    def perceive(
        self,
        text: str,
        category: str,
        correct: bool,
        action: str = "",
        **extra,
    ) -> None:
        """Process an experience through all memory tiers."""
        record = {
            "text": text,
            "category": category,
            "correct": correct,
            "action": action,
            **extra,
        }

        # Working memory: push latest
        self.working.push(record)

        # Episodic: store full experience
        self.episodic.store(record)

        # Semantic: distill knowledge
        self.semantic.learn(category, action, correct)

        # Procedural: build action patterns
        self.procedural.record(category, action, correct)

    def recall(self, query: str = "", category: str = "") -> Dict[str, Any]:
        """Recall from all memory tiers for a given context.

        Returns a unified context with contributions from each tier.
        """
        result = {
            "working_memory": self.working.query_relevant(query, 3),
            "episodic": [],
            "semantic": None,
            "procedural": None,
            "recommended_action": None,
            "confidence": 0.0,
            "sources_used": [],
        }

        # Episodic recall
        if category:
            eps = self.episodic.recall_by_category(category, limit=3)
            result["episodic"] = [
                {k: v for k, v in e.items() if not k.startswith("_")} for e in eps
            ]
            if eps:
                result["sources_used"].append("episodic")

        # Semantic knowledge
        if category:
            knowledge = self.semantic.get_knowledge(category)
            result["semantic"] = knowledge
            if knowledge:
                result["sources_used"].append("semantic")

        # Procedural rule
        if category:
            proc_result = self.procedural.get_action(category)
            if proc_result:
                action, conf = proc_result
                result["procedural"] = {"action": action, "confidence": conf}
                result["sources_used"].append("procedural")

        # Determine recommended action (priority: procedural > semantic > episodic)
        if result["procedural"]:
            result["recommended_action"] = result["procedural"]["action"]
            result["confidence"] = result["procedural"]["confidence"]
        elif result["semantic"] and result["semantic"].get("best_action"):
            best = result["semantic"]["best_action"]
            result["recommended_action"] = best[0]
            result["confidence"] = best[1]
        elif result["episodic"]:
            # Most recent correct episodic memory
            correct_eps = [e for e in result["episodic"] if e.get("correct")]
            if correct_eps:
                result["recommended_action"] = correct_eps[-1].get("action")
                result["confidence"] = 0.5

        return result

    def stats(self) -> Dict[str, Any]:
        """Statistics across all memory tiers."""
        return {
            "working_memory": self.working.size,
            "episodic_memories": self.episodic.size,
            "semantic_categories": self.semantic.categories_known,
            "procedural_rules": len(self.procedural.list_rules()),
            "total_entries": (self.working.size + self.episodic.size),
        }

    def print_state(self):
        """Print current memory state across all tiers."""
        s = self.stats()
        logger.info("\n  Cognitive Memory State:")
        print(
            f"    Working Memory:   {s['working_memory']} items (capacity: {self.working.capacity})"
        )
        logger.info(f"    Episodic Memory:  {s['episodic_memories']} experiences")
        logger.info(f"    Semantic Memory:  {s['semantic_categories']} categories known")
        logger.info(f"    Procedural Rules: {s['procedural_rules']} learned rules")

        rules = self.procedural.list_rules()
        if rules:
            logger.info("\n    Learned rules:")
            for r in rules[:5]:
                print(
                    f"      {r['rule']} ({r['confidence']:.0%}, n={r['observations']})"
                )


# ---------------------------------------------------------------------------
# Opt 3: Unified Memory — Cross-system sync
# ---------------------------------------------------------------------------


class UnifiedMemory:
    """Unified wrapper that writes to all memory systems simultaneously.

    Solves the fragmentation problem when an agent uses Memory,
    SemanticMemory, and CognitiveMemory independently.

    Usage::

        from cognicore.multi_memory import UnifiedMemory

        mem = UnifiedMemory()
        mem.store(text="phishing email", category="security",
                  correct=False, action="SAFE")
        context = mem.recall("suspicious email", category="security")
        # -> combined results from all 3 memory systems
    """

    def __init__(
        self,
        working_capacity: int = 7,
        episodic_max: int = 5000,
        semantic_decay: float = 0.95,
        memory_max_size: int = 10_000,
    ):
        from cognicore.middleware.memory import Memory
        from cognicore.advanced_memory import SemanticMemory

        self.basic = Memory(max_size=memory_max_size)
        self.semantic = SemanticMemory(
            max_size=memory_max_size, decay_rate=semantic_decay
        )
        self.cognitive = CognitiveMemory(
            working_capacity=working_capacity, episodic_max=episodic_max
        )

    def store(
        self,
        text: str,
        category: str,
        correct: bool,
        action: str = "",
        **extra,
    ) -> None:
        """Store an experience across all 3 memory systems."""
        entry = {
            "text": text,
            "category": category,
            "correct": correct,
            "action": action,
            **extra,
        }

        # Basic memory
        self.basic.store(entry)

        # Semantic memory (TF-IDF indexed)
        self.semantic.store(entry)

        # Cognitive memory (4-tier human-like)
        self.cognitive.perceive(text, category, correct, action, **extra)

    def recall(
        self, query: str = "", category: str = "", top_k: int = 5
    ) -> Dict[str, Any]:
        """Recall from all memory systems, merged into one context.

        Returns
        -------
        dict
            Keys: ``basic``, ``semantic``, ``cognitive``, ``best_actions``,
            ``worst_actions``, ``recommended_action``, ``confidence``.
        """
        result: Dict[str, Any] = {
            "basic": [],
            "semantic": [],
            "cognitive": {},
            "best_actions": [],
            "worst_actions": [],
            "recommended_action": None,
            "confidence": 0.0,
        }

        # Basic memory
        if category:
            result["basic"] = self.basic.retrieve(category, top_k=top_k)

        # Semantic memory
        search_query = query or category
        if search_query:
            result["semantic"] = self.semantic.recall(search_query, top_k=top_k)
            result["best_actions"] = self.semantic.best_actions(
                search_query, top_k=3
            )
            result["worst_actions"] = self.semantic.worst_actions(
                search_query, top_k=3
            )

        # Cognitive memory
        cog = self.cognitive.recall(query=query, category=category)
        result["cognitive"] = cog
        result["recommended_action"] = cog.get("recommended_action")
        result["confidence"] = cog.get("confidence", 0.0)

        return result

    def export_jsonl(self, output_path: str) -> int:
        """Export all semantic memory entries as JSONL."""
        return self.semantic.export_jsonl(output_path)

    def stats(self) -> Dict[str, Any]:
        """Combined statistics from all memory systems."""
        return {
            "basic": self.basic.stats(),
            "semantic": self.semantic.stats(),
            "cognitive": self.cognitive.stats(),
        }
