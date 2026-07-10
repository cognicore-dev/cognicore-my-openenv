import logging
from typing import List

from cognicore.memory.base import EmbeddingProvider

logger = logging.getLogger(__name__)

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using OpenAI's embeddings API.
    Requires `openai` to be installed and an OPENAI_API_KEY environment variable.
    """

    def __init__(self, model_name: str = "text-embedding-3-small", dimension: int = 1536, api_key: str = None):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenAIEmbeddingProvider. "
                "Install it with: pip install openai"
            )
        
        self.model_name = model_name
        self._dimension = dimension
        self.client = openai.OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAIEmbeddingProvider with model '{model_name}' (dimension={self._dimension})")

    def embed(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self._dimension
        response = self.client.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # Replace empty strings with a space to prevent OpenAI API errors
        safe_texts = [t if t else " " for t in texts]
        response = self.client.embeddings.create(input=safe_texts, model=self.model_name)
        return [data.embedding for data in response.data]

    @property
    def dimension(self) -> int:
        return self._dimension
