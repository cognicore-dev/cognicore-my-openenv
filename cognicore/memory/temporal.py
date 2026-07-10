from typing import List, Optional, Dict
from cognicore.memory.base import SearchResult, MemoryBackend
import time
import math

class TemporalResolutionEngine:
    def __init__(self, backend: MemoryBackend):
        self.backend = backend

    def resolve(self, results: List[SearchResult], question_timestamp: Optional[float] = None) -> List[SearchResult]:
        """Apply temporal filters, supersession logic, and time-decay to search results."""
        if not results:
            return []

        resolved_results = []
        
        for res in results:
            entry = res.entry
            
            # 1. Temporal Filter (Future Leakage Prevention)
            if question_timestamp is not None and entry.timestamp > 0:
                if entry.timestamp > question_timestamp:
                    continue # Drop future memories
                    
            # 2. Conflict Resolution (Supersedes DAG)
            current_entry = entry
            
            while True:
                if hasattr(self.backend, "get_superseding"):
                    next_entry = self.backend.get_superseding(current_entry.entry_id)
                    if next_entry:
                        if question_timestamp is not None and next_entry.timestamp > 0:
                            if next_entry.timestamp > question_timestamp:
                                break
                        current_entry = next_entry
                    else:
                        break
                else:
                    break
                    
            terminal_res = SearchResult(
                entry=current_entry,
                score=res.score, 
                source="temporal" if current_entry.entry_id != entry.entry_id else res.source
            )
            
            # 3. Preference Time-Decay Scoring
            if terminal_res.entry.memory_type == "preference" and terminal_res.entry.timestamp > 0:
                now = time.time()
                age = max(0, now - terminal_res.entry.timestamp)
                half_life = 30 * 24 * 3600 
                decay_factor = math.exp(-0.693 * (age / half_life)) if half_life > 0 else 1.0
                terminal_res.score = terminal_res.score * decay_factor
                
            resolved_results.append(terminal_res)
            
        unique_results = {}
        for r in resolved_results:
            if r.entry.entry_id not in unique_results or r.score > unique_results[r.entry.entry_id].score:
                unique_results[r.entry.entry_id] = r
                
        final_list = list(unique_results.values())
        final_list.sort(key=lambda x: x.score, reverse=True)
        return final_list
