from typing import List, Dict, Any
from cognicore_benchmarks.longmemeval.v2.state_store import V2MemoryIndexes

def retrieve_evidence(
    question: str, 
    query_type: str, 
    indexes: V2MemoryIndexes,
    top_k: int,
    question_timestamp: float = None
) -> Dict[str, Any]:
    """Retrieves and routes evidence based on query type."""
    
    evidence = {
        "query_type": query_type,
        "primary_evidence": [],
        "supporting_evidence": []
    }
    
    if query_type == "knowledge_update":
        # Pull all current state facts
        evidence["primary_evidence"] = indexes.current_state.get_all_facts()
        # Fallback to semantic search on updates and facts
        if indexes.typed_store:
            results = indexes.typed_store.search(question, top_k=top_k)
            evidence["supporting_evidence"] = [r.entry for r in results if r.entry.category in ["fact", "update"]]
            
    elif query_type == "preference":
        evidence["primary_evidence"] = indexes.preference_state.get_all_preferences()
        if indexes.typed_store:
            results = indexes.typed_store.search(question, top_k=top_k)
            evidence["supporting_evidence"] = [r.entry for r in results if r.entry.category == "preference_evidence"]
            
    elif query_type == "temporal_reasoning":
        evidence["primary_evidence"] = indexes.timeline_store.get_all_events()
        if indexes.raw_store:
            results = indexes.raw_store.search(question, top_k=top_k)
            evidence["supporting_evidence"] = [r.entry for r in results]
            
    elif query_type == "assistant_recall":
        evidence["primary_evidence"] = indexes.artifact_store.get_all_artifacts()
        if indexes.typed_store:
            results = indexes.typed_store.search(question, top_k=top_k)
            evidence["supporting_evidence"] = [r.entry for r in results if r.entry.category in ["assistant_action", "artifact"]]
            
    else: # fact_lookup
        evidence["primary_evidence"] = indexes.current_state.get_all_facts()
        if indexes.raw_store:
            results = indexes.raw_store.search(question, top_k=top_k)
            evidence["supporting_evidence"] = [r.entry for r in results]
            
    return evidence
