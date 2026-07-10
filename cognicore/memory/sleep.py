import time
import logging
from typing import Dict, List, Optional, Callable, Any

from cognicore.memory.base import MemoryEntry, MemoryState, MemoryType, MemoryBackend

logger = logging.getLogger("cognicore.sleep")

class SleepProcessor:
    """Consolidates and optimizes memories offline, mimicking biological sleep."""

    def __init__(self, 
                 backend: MemoryBackend, 
                 llm_fn: Optional[Callable[[str], str]] = None,
                 similarity_threshold: float = 0.7):
        self.backend = backend
        self.llm_fn = llm_fn
        self.similarity_threshold = similarity_threshold

    def sleep(self) -> Dict[str, int]:
        """Perform a full sleep consolidation cycle.
        
        Returns:
            Dict containing the counts of operations performed.
        """
        logger.info("Starting memory consolidation (sleep cycle)...")
        stats = {"merged": 0, "archived_contradictions": 0, "compressed_episodes": 0}
        
        entries = self.backend.get_all()
        if not entries:
            logger.info("No memories found to consolidate.")
            return stats

        # 1. Deduplication
        stats["merged"] = self._deduplicate(entries)
        
        # Reload entries after deduplication
        entries = self.backend.get_all()
        
        # 2. Contradiction Resolution
        stats["archived_contradictions"] = self._resolve_contradictions(entries)
        
        # Reload entries after contradiction resolution
        entries = self.backend.get_all()
        
        # 3. Episodic Compression
        stats["compressed_episodes"] = self._compress_episodes(entries)
        
        logger.info(f"Sleep cycle completed: {stats}")
        return stats

    def _deduplicate(self, entries: List[MemoryEntry]) -> int:
        """Merge semantically highly similar entries."""
        merged_count = 0
        # Group entries by type and category
        groups: Dict[str, List[MemoryEntry]] = {}
        for e in entries:
            if e.state == MemoryState.ARCHIVED.value:
                continue
            # Skip episodic memories for deduplication
            if e.memory_type == MemoryType.EPISODIC.value:
                continue
            key = f"{e.memory_type}:{e.category}"
            groups.setdefault(key, []).append(e)

        for key, group_entries in groups.items():
            to_remove = set()
            for i in range(len(group_entries)):
                e1 = group_entries[i]
                if e1.entry_id in to_remove:
                    continue
                
                for j in range(i + 1, len(group_entries)):
                    e2 = group_entries[j]
                    if e2.entry_id in to_remove:
                        continue
                    
                    if self._are_contradictory(e1.text, e2.text):
                        continue
                        
                    similarity = self._jaccard_similarity(e1.text, e2.text)
                    if similarity >= self.similarity_threshold:
                        # Merge e2 into e1
                        e1.retrieval_count += e2.retrieval_count
                        e1.used_count += e2.used_count
                        e1.ignored_count += e2.ignored_count
                        e1.positive_outcomes += e2.positive_outcomes
                        e1.negative_outcomes += e2.negative_outcomes
                        
                        # Recalculate utility (average of both utility scores)
                        e1.utility_score = (e1.utility_score + e2.utility_score) / 2.0
                        
                        self.backend.update(
                            e1.entry_id,
                            retrieval_count=e1.retrieval_count,
                            used_count=e1.used_count,
                            ignored_count=e1.ignored_count,
                            positive_outcomes=e1.positive_outcomes,
                            negative_outcomes=e1.negative_outcomes,
                            utility_score=e1.utility_score
                        )
                        
                        self.backend.delete(e2.entry_id)
                        to_remove.add(e2.entry_id)
                        merged_count += 1
                        
        return merged_count

    def _resolve_contradictions(self, entries: List[MemoryEntry]) -> int:
        """Find and archive contradictory constraint or preference memories."""
        archived_count = 0
        candidates = [
            e for e in entries 
            if e.memory_type in (MemoryType.PREFERENCE.value, MemoryType.CONSTRAINT.value) 
            and e.state != MemoryState.ARCHIVED.value
        ]
        
        # Group by category
        by_cat: Dict[str, List[MemoryEntry]] = {}
        for e in candidates:
            by_cat.setdefault(e.category, []).append(e)
            
        for cat, cat_entries in by_cat.items():
            for i in range(len(cat_entries)):
                e1 = cat_entries[i]
                if e1.state == MemoryState.ARCHIVED.value:
                    continue
                    
                for j in range(i + 1, len(cat_entries)):
                    e2 = cat_entries[j]
                    if e2.state == MemoryState.ARCHIVED.value:
                        continue
                        
                    if self._are_contradictory(e1.text, e2.text):
                        # Archive the one with lower utility score or recency
                        if e1.utility_score >= e2.utility_score:
                            self.backend.update(e2.entry_id, state=MemoryState.ARCHIVED.value)
                            e2.state = MemoryState.ARCHIVED.value
                        else:
                            self.backend.update(e1.entry_id, state=MemoryState.ARCHIVED.value)
                            e1.state = MemoryState.ARCHIVED.value
                        archived_count += 1
                        
        return archived_count

    def _compress_episodes(self, entries: List[MemoryEntry]) -> int:
        """Compress old episodic execution histories into high-level semantic rules."""
        compressed_count = 0
        episodic = [
            e for e in entries 
            if e.memory_type == MemoryType.EPISODIC.value 
            and e.state != MemoryState.ARCHIVED.value
        ]
        
        by_cat: Dict[str, List[MemoryEntry]] = {}
        for e in episodic:
            by_cat.setdefault(e.category, []).append(e)
            
        for cat, cat_entries in by_cat.items():
            if len(cat_entries) < 5:
                continue
                
            cat_entries.sort(key=lambda e: e.timestamp)
            # Compress all but the last 2 episodic memories
            to_compress = cat_entries[:-2]
            if not to_compress:
                continue
                
            if self.llm_fn:
                texts = [f"- {e.text} (Success: {e.correct})" for e in to_compress]
                prompt = (
                    "Summarize the following agent execution history into a concise set of facts or rules (semantic memory).\n"
                    "Focus on what worked and what failed:\n" + "\n".join(texts)
                )
                try:
                    summary_text = self.llm_fn(prompt)
                except Exception as e:
                    logger.warning(f"LLM summarization failed: {e}. Falling back to simple heuristic.")
                    summary_text = self._fallback_summarize(to_compress)
            else:
                summary_text = self._fallback_summarize(to_compress)
                
            # Store the new semantic entry
            new_entry = MemoryEntry(
                text=summary_text,
                memory_type=MemoryType.SEMANTIC.value,
                state=MemoryState.ACTIVE.value,
                category=cat,
                creation_reason="Sleep compression of episodic memory",
                timestamp=time.time(),
                last_accessed=time.time()
            )
            self.backend.store(new_entry)
            
            # Archive the compressed episodic memories
            for e in to_compress:
                self.backend.update(e.entry_id, state=MemoryState.ARCHIVED.value)
                compressed_count += 1
                
        return compressed_count

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        a = set(text1.lower().split())
        b = set(text2.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _are_contradictory(self, text1: str, text2: str) -> bool:
        t1 = text1.lower()
        t2 = text2.lower()
        
        # Check opposing key terms in overlapping context
        words1 = set(t1.split())
        words2 = set(t2.split())
        
        # If there is decent overlap in context words
        overlap = words1 & words2
        if len(overlap) >= 2:
            opposites = [
                ("space", "tab"), ("spaces", "tabs"), 
                ("dark", "light"), ("always", "never"),
                ("true", "false"), ("yes", "no")
            ]
            for op1, op2 in opposites:
                if (op1 in words1 and op2 in words2) or (op2 in words1 and op1 in words2):
                    return True
                    
        # Negation check
        w_list1 = [w for w in t1.split() if w.isalnum()]
        w_list2 = [w for w in t2.split() if w.isalnum()]
        negations = {"not", "never", "no", "don't", "avoid", "cannot"}
        
        if len(overlap) >= max(len(w_list1), len(w_list2)) - 2:
            has_neg1 = any(n in w_list1 for n in negations)
            has_neg2 = any(n in w_list2 for n in negations)
            if has_neg1 != has_neg2:
                return True
                
        return False

    def _fallback_summarize(self, entries: List[MemoryEntry]) -> str:
        """Fallback rule-based summarization of episodic histories."""
        successes = [e.text for e in entries if e.correct]
        failures = [e.text for e in entries if not e.correct]
        
        summary = "Consolidated History:\n"
        if successes:
            summary += f"- Successful patterns: {', '.join(successes[:3])}\n"
        if failures:
            summary += f"- Avoid failures: {', '.join(failures[:3])}\n"
        return summary
