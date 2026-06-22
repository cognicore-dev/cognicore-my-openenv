from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseAgentAdapter(ABC):
    """Abstract interface for benchmark agents."""
    
    @abstractmethod
    def ingest_history(self, session_data: List[Dict[str, str]]) -> None:
        """
        Process and store conversational history.
        
        Args:
            session_data: List of dicts, e.g., [{"role": "user", "content": "..."}, ...]
        """
        pass

    @abstractmethod
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a benchmark question based on the ingested history.
        
        Args:
            question: The question string.
            
        Returns:
            Dict containing:
            - 'answer': The string response
            - 'latency_s': Time taken
            - 'tokens': Tokens consumed (optional)
        """
        pass
