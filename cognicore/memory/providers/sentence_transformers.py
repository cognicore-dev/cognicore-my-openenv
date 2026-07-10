import logging
from typing import List, Optional

from cognicore.memory.base import EmbeddingProvider

logger = logging.getLogger(__name__)

class SentenceTransformerProvider(EmbeddingProvider):
    """
    Embedding provider using the sentence-transformers library.
    Requires `sentence-transformers` to be installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "The 'sentence-transformers' package is required for SentenceTransformerProvider. "
                "Install it with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # We can determine the dimension by running a dummy embedding
        dummy = self.model.encode(["test"])[0]
        self._dimension = len(dummy)
        logger.info(f"Initialized SentenceTransformerProvider with model '{model_name}' (dimension={self._dimension})")

    def embed(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self._dimension
        return self.model.encode([text])[0].tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension
