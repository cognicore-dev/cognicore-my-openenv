from typing import Dict, Any, List

def exact_match(prediction: str, reference: str) -> bool:
    """Check if prediction exactly matches reference (ignoring case/whitespace)."""
    if not prediction or reference is None:
        return False
    return str(prediction).strip().lower() == str(reference).strip().lower()

def includes_match(prediction: str, reference: str) -> bool:
    """Check if the reference string is contained within the prediction."""
    if not prediction or reference is None:
        return False
    return str(reference).strip().lower() in str(prediction).strip().lower()

def abstention_check(prediction: str) -> bool:
    """Check if the model abstained (e.g. 'I don't know', 'not found')."""
    if not prediction:
        return True
    
    abstain_phrases = [
        "i don't know", "i do not know", "not found", "cannot find",
        "no information", "not mentioned", "don't have", "does not appear"
    ]
    pred_lower = prediction.strip().lower()
    return any(phrase in pred_lower for phrase in abstain_phrases)

def evaluate_metrics(prediction: str, expected_answer: str, expects_abstain: bool = False) -> Dict[str, Any]:
    """Calculate basic metrics for a single prediction."""
    is_abstain = abstention_check(prediction)
    
    if expects_abstain:
        score = 1.0 if is_abstain else 0.0
        return {
            "score": score,
            "exact_match": False,
            "includes_match": False,
            "abstained": is_abstain,
            "correct_abstention": True if is_abstain else False
        }
    
    if is_abstain:
        return {
            "score": 0.0,
            "exact_match": False,
            "includes_match": False,
            "abstained": True,
            "correct_abstention": False
        }
        
    exact = exact_match(prediction, expected_answer)
    includes = includes_match(prediction, expected_answer)
    
    score = 1.0 if (exact or includes) else 0.0
    
    return {
        "score": score,
        "exact_match": exact,
        "includes_match": includes,
        "abstained": False,
        "correct_abstention": False
    }
