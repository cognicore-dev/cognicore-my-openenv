import time
import uuid
from typing import Dict, List, Optional, Any, Callable

from cognicore.memory.base import (
    MemoryEntry, MemoryState, MemoryType, MemoryBackend, 
    SearchResult, RetrievalPlan, MemoryTrace, EvaluationResult
)
from cognicore.memory.evaluator import MemoryEvaluator, RuleBasedEvaluator
from cognicore.memory.planner import RetrievalPlanner, RuleBasedPlanner
from cognicore.memory.utility import UtilityScorer


class MemoryLifecycleManager:
    """Central orchestrator for the memory operating system.
    
    Provides the complete memory lifecycle:
    observe → evaluate → store → retrieve → measure → promote/decay
    """
    
    def __init__(self, 
                 backend: MemoryBackend,
                 evaluator: Optional[MemoryEvaluator] = None,
                 planner: Optional[RetrievalPlanner] = None,
                 scorer: Optional[UtilityScorer] = None):
        self.backend = backend
        self.evaluator = evaluator or RuleBasedEvaluator()
        self.planner = planner or RuleBasedPlanner()
        self.scorer = scorer or UtilityScorer()
        self._traces: List[MemoryTrace] = []
        self._trace_counter = 0

    def observe(self, text: str, context: Optional[Dict] = None, 
                source_component: str = '', source_agent: str = '', 
                source_task: str = '') -> Optional[str]:
        """Process an observation and decide whether to store it as memory."""
        context = context or {}
        evaluation = self.evaluator.evaluate(text, context)
        
        if not evaluation.should_store:
            return None
            
        entry = MemoryEntry(
            text=text,
            memory_type=evaluation.memory_type,
            importance=evaluation.importance,
            creation_reason=evaluation.creation_reason,
            confidence=evaluation.confidence,
            source_component=source_component,
            source_agent=source_agent,
            source_task=source_task,
            state=MemoryState.CANDIDATE.value,
            timestamp=time.time(),
            last_accessed=time.time()
        )
        
        entry_id = self.backend.store(entry)
        return entry_id

    def retrieve(self, query: str, task: str = '', context: Optional[Dict] = None, top_k: int = 5) -> List[SearchResult]:
        """Retrieve memories using the intelligent planner."""
        context = context or {}
        plan = self.planner.plan(query, task, context)
        
        if not plan.should_retrieve:
            return []
            
        # Use plan's budget, cap with provided top_k if necessary, or just use budget
        budget = min(plan.budget, top_k) if plan.budget > 0 else top_k
        results = self.backend.search(query, top_k=budget)
        
        if plan.memory_types:
            results = [r for r in results if r.entry.memory_type in plan.memory_types]
            
        # Update utility tracker for retrieved entries
        for result in results:
            self.scorer.on_retrieval(result.entry)
            self.backend.update(
                result.entry.entry_id, 
                retrieval_count=result.entry.retrieval_count,
                last_accessed=result.entry.last_accessed
            )
            
        trace = MemoryTrace(
            trace_id=str(uuid.uuid4()),
            query=query,
            plan=plan,
            retrieved=results,
            timestamp=time.time()
        )
        self._traces.append(trace)
        
        # In case we have an AdaptivePlanner that records traces
        if hasattr(self.planner, 'record_trace'):
            self.planner.record_trace(trace)
            
        return results

    def record_outcome(self, trace_id: str, outcome: str, task_success_delta: float = 0.0) -> None:
        """Record the outcome of a memory retrieval trace."""
        trace = self.explain(trace_id)
        if not trace:
            return
            
        trace.outcome = outcome
        trace.task_success_delta = task_success_delta
        
        for result in trace.retrieved:
            entry = result.entry
            if outcome == 'success':
                self.scorer.on_used(entry, positive=True)
            elif outcome == 'failure':
                self.scorer.on_used(entry, positive=False)
            elif outcome == 'ignored':
                self.scorer.on_ignored(entry)
                
            self.backend.update(
                entry.entry_id,
                used_count=entry.used_count,
                ignored_count=entry.ignored_count,
                positive_outcomes=entry.positive_outcomes,
                negative_outcomes=entry.negative_outcomes,
                utility_score=entry.utility_score
            )
            
        if hasattr(self.planner, 'record_trace'):
            self.planner.record_trace(trace)

    def run_lifecycle_pass(self) -> Dict[str, int]:
        """Evaluate all memories and promote/decay them based on utility."""
        counts = {'promoted': 0, 'decayed': 0, 'archived': 0, 'unchanged': 0}
        entries = self.backend.get_all()
        
        for entry in entries:
            current_state = entry.state
            recommended_state = self.scorer.get_promotion_recommendation(entry)
            
            if recommended_state != current_state:
                self.backend.update(entry.entry_id, state=recommended_state)
                
                if recommended_state == MemoryState.ARCHIVED.value:
                    counts['archived'] += 1
                elif recommended_state == MemoryState.VERIFIED.value or (current_state == MemoryState.CANDIDATE.value and recommended_state == MemoryState.ACTIVE.value):
                    counts['promoted'] += 1
                else:
                    counts['decayed'] += 1
            else:
                counts['unchanged'] += 1
                
        return counts

    def explain(self, trace_id: str) -> Optional[MemoryTrace]:
        """Retrieve a memory trace by ID."""
        for trace in self._traces:
            if trace.trace_id == trace_id:
                return trace
        return None

    def get_health_report(self) -> Dict[str, Any]:
        """Get aggregate statistics on memory health."""
        entries = self.backend.get_all()
        total_memories = len(entries)
        
        state_distribution = {}
        type_distribution = {}
        total_utility = 0.0
        negative_transfer_count = 0
        
        for entry in entries:
            state_distribution[entry.state] = state_distribution.get(entry.state, 0) + 1
            type_distribution[entry.memory_type] = type_distribution.get(entry.memory_type, 0) + 1
            total_utility += entry.utility_score
            if self.scorer.detect_negative_transfer(entry):
                negative_transfer_count += 1
                
        avg_utility = total_utility / total_memories if total_memories > 0 else 0.0
        
        return {
            'total_memories': total_memories,
            'state_distribution': state_distribution,
            'type_distribution': type_distribution,
            'avg_utility': avg_utility,
            'negative_transfer_count': negative_transfer_count,
            'total_traces': len(self._traces),
            'recent_traces': [
                {
                    'trace_id': t.trace_id, 
                    'query': t.query, 
                    'outcome': t.outcome
                } 
                for t in self._traces[-10:]
            ]
        }

    def store_direct(self, entry: MemoryEntry) -> str:
        """Bypass the evaluator and store directly."""
        return self.backend.store(entry)

    def store(self, entry: MemoryEntry) -> str:
        """Backward compatibility alias for store_direct."""
        return self.store_direct(entry)

    def search_direct(self, query: str, top_k: int = 5, **kwargs) -> List[SearchResult]:
        """Bypass the planner and search directly."""
        return self.backend.search(query, top_k=top_k, **kwargs)

    def trigger_sleep(self, llm_fn: Optional[Callable[[str], str]] = None) -> Dict[str, int]:
        """Trigger the offline memory consolidation process."""
        from cognicore.memory.sleep import SleepProcessor
        processor = SleepProcessor(backend=self.backend, llm_fn=llm_fn)
        return processor.sleep()

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[SearchResult]:
        """Backward compatibility alias for search_direct."""
        return self.search_direct(query, top_k=top_k, **kwargs)
        
    @property
    def memory(self) -> MemoryBackend:
        """Access the underlying memory backend (backward compatibility)."""
        return self.backend
        
    @property
    def traces(self) -> List[MemoryTrace]:
        """Access the memory traces."""
        return self._traces

    @property
    def entries(self) -> List[MemoryEntry]:
        """Backward compatibility for tests accessing .entries."""
        if hasattr(self.backend, 'entries'):
            return self.backend.entries
        return self.backend.get_all()

    @entries.setter
    def entries(self, value: List[MemoryEntry]) -> None:
        """Backward compatibility for tests setting .entries."""
        if hasattr(self.backend, 'entries'):
            self.backend.entries = value

    def get_by_category(self, category: str, top_k: int = 5, success_filter: Optional[bool] = None) -> List[MemoryEntry]:
        """Backward compatibility for tests calling .get_by_category()."""
        if hasattr(self.backend, 'get_by_category'):
            return self.backend.get_by_category(category, top_k, success_filter)
        return []

    def count(self) -> int:
        """Backward compatibility for .count()."""
        return self.backend.count()
