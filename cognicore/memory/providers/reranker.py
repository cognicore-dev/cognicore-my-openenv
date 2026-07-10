import time
from typing import List, Tuple
from sentence_transformers import CrossEncoder

class SentenceTransformerReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        
    def rerank(self, query: str, candidates: List[str]) -> List[float]:
        """
        Scores a list of candidate documents against a query.
        Returns a list of float scores corresponding to the input candidates.
        """
        if not candidates:
            return []
            
        pairs = [[query, doc] for doc in candidates]
        scores = self.model.predict(pairs)
        return scores.tolist()
