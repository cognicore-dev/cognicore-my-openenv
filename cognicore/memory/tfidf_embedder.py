from typing import List
from .base import EmbeddingProvider

class TFIDFEmbeddingProvider(EmbeddingProvider):
    """
    Zero-dependency TF-IDF 'embeddings' provider.
    This class is mostly a placeholder to satisfy the EmbeddingProvider interface
    for systems that strictly require an embedder. The actual TF-IDF logic
    is optimized within the TFIDFMemoryBackend using sparse dictionary vectors.
    """

    def embed(self, text: str) -> List[float]:
        # TF-IDF vectors are dynamically sized and sparse, so a dense list
        # is not a true representation without a fixed global vocabulary.
        # This returns a dummy embedding just to fulfill the protocol.
        return [0.0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] for _ in texts]

    @property
    def dimension(self) -> int:
        return 1
