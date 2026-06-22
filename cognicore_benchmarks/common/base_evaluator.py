from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseEvaluator(ABC):
    """Abstract base class for benchmark specific evaluation logic."""
    
    @abstractmethod
    def evaluate(self, prediction: str, reference: Any) -> Dict[str, Any]:
        """
        Evaluate a single prediction against its reference.
        
        Args:
            prediction: The generated text from the agent.
            reference: The ground truth answer(s) or criteria.
            
        Returns:
            A dictionary of metrics (e.g., {"score": 1.0, "exact_match": True}).
        """
        pass
