import re

def classify_query(question: str) -> str:
    """Classifies a LongMemEval question into one of the V2 routing types."""
    q = question.lower()
    
    # 1. Knowledge Update
    if any(w in q for w in ["current", "now", "latest", "recently changed", "currently"]):
        return "knowledge_update"
        
    # 2. Preference
    if any(w in q for w in ["prefer", "like", "dislike", "favorite kind", "usually choose", "tend to"]):
        return "preference"
        
    # 3. Assistant Recall
    if any(w in q for w in ["what did you suggest", "what was the schedule", "remind me what you said", "previous chat", "you recommend"]):
        return "assistant_recall"
        
    # 4. Temporal Reasoning
    if any(w in q for w in ["before", "after", "first", "later", "earlier", "when", "which happened first", "how long after"]):
        return "temporal_reasoning"
        
    # Fallback Fact Lookup
    return "fact_lookup"
