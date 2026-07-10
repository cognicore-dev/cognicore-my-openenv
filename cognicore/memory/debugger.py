import json
from typing import Dict, List, Optional, Any

from cognicore.memory.base import (
    MemoryEntry, MemoryState, MemoryType, MemoryBackend, 
    SearchResult, MemoryTrace
)
from cognicore.memory.utility import UtilityScorer


class MemoryDebugger:
    """Developer tooling for inspecting and understanding memory behavior."""
    
    def __init__(self, backend: MemoryBackend, scorer: Optional[UtilityScorer] = None, traces: Optional[List[MemoryTrace]] = None):
        self.backend = backend
        self.scorer = scorer or UtilityScorer()
        self._traces = traces if traces is not None else []

    def explain_response(self, trace: MemoryTrace) -> Dict[str, Any]:
        """Explain why memory produced a specific response."""
        retrieved_details = []
        for result in trace.retrieved:
            entry = result.entry
            retrieved_details.append({
                'text': entry.text,
                'score': result.score,
                'source': result.source,
                'memory_type': entry.memory_type,
                'utility_score': entry.utility_score
            })
            
        reasoning = f"Planner decided to retrieve. Budget was {trace.plan.budget if trace.plan else 'unknown'}."
        if trace.plan and trace.plan.reasoning:
            reasoning = trace.plan.reasoning
            
        retrieval_improved = None
        if trace.outcome:
            retrieval_improved = (trace.outcome == 'success')
            
        plan_dict = {}
        if trace.plan:
            plan_dict = {
                'should_retrieve': trace.plan.should_retrieve,
                'budget': trace.plan.budget,
                'memory_types': trace.plan.memory_types,
                'priority_order': trace.plan.priority_order
            }
            
        return {
            'query': trace.query,
            'plan': plan_dict,
            'retrieved_count': len(trace.retrieved),
            'retrieved': retrieved_details,
            'outcome': trace.outcome,
            'retrieval_improved_outcome': retrieval_improved,
            'reasoning': reasoning
        }

    def get_trace_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get simplified trace history."""
        history = []
        for trace in reversed(self._traces[-limit:]):
            history.append({
                'trace_id': trace.trace_id,
                'query': trace.query,
                'retrieved_count': len(trace.retrieved),
                'outcome': trace.outcome,
                'timestamp': trace.timestamp
            })
        return history

    def get_utility_distribution(self) -> Dict[str, Any]:
        """Analyze utility scores across all memories."""
        entries = self.backend.get_all()
        total_entries = len(entries)
        
        if total_entries == 0:
            return {
                'total_entries': 0,
                'mean_utility': 0.0,
                'positive_utility_count': 0,
                'negative_utility_count': 0,
                'zero_utility_count': 0,
                'top_useful': [],
                'least_useful': []
            }
            
        total_utility = 0.0
        positive = 0
        negative = 0
        zero = 0
        
        for entry in entries:
            u = entry.utility_score
            total_utility += u
            if u > 0: positive += 1
            elif u < 0: negative += 1
            else: zero += 1
            
        sorted_entries = sorted(entries, key=lambda e: e.utility_score, reverse=True)
        
        def format_entry(e):
            return {
                'text': e.text,
                'utility_score': e.utility_score,
                'retrieval_count': e.retrieval_count
            }
            
        top_useful = [format_entry(e) for e in sorted_entries[:10]]
        least_useful = [format_entry(e) for e in sorted_entries[-10:] if e.utility_score < 0]
        
        return {
            'total_entries': total_entries,
            'mean_utility': total_utility / total_entries,
            'positive_utility_count': positive,
            'negative_utility_count': negative,
            'zero_utility_count': zero,
            'top_useful': top_useful,
            'least_useful': least_useful
        }

    def get_negative_transfers(self) -> List[Dict[str, Any]]:
        """Find memories that are actively harming performance."""
        entries = self.backend.get_all()
        negatives = []
        
        for entry in entries:
            if self.scorer.detect_negative_transfer(entry):
                negatives.append({
                    'text': entry.text,
                    'utility_score': entry.utility_score,
                    'retrieval_count': entry.retrieval_count,
                    'negative_outcomes': entry.negative_outcomes,
                    'state': entry.state
                })
                
        return negatives

    def get_memory_timeline(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get chronological view of memory creation."""
        entries = self.backend.get_all()
        sorted_entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)[:limit]
        
        return [{
            'entry_id': e.entry_id,
            'text': e.text,
            'memory_type': e.memory_type,
            'state': e.state,
            'timestamp': e.timestamp,
            'utility_score': e.utility_score
        } for e in sorted_entries]

    def get_state_summary(self) -> Dict[str, int]:
        """Get counts of entries by state."""
        entries = self.backend.get_all()
        summary = {}
        for entry in entries:
            summary[entry.state] = summary.get(entry.state, 0) + 1
        return summary

    def get_type_summary(self) -> Dict[str, int]:
        """Get counts of entries by type."""
        entries = self.backend.get_all()
        summary = {}
        for entry in entries:
            summary[entry.memory_type] = summary.get(entry.memory_type, 0) + 1
        return summary

    def export_debug_report(self, path: str) -> None:
        """Export comprehensive debug report to JSON."""
        report = {
            'utility_distribution': self.get_utility_distribution(),
            'state_summary': self.get_state_summary(),
            'type_summary': self.get_type_summary(),
            'negative_transfers': self.get_negative_transfers(),
            'recent_history': self.get_trace_history(limit=20)
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
