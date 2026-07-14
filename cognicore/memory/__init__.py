from .base import MemoryScope, MemoryEntry, SearchResult, MemoryBackend, EmbeddingProvider

from .sqlite_backend import SQLiteMemoryBackend
from .tfidf_backend import TFIDFMemoryBackend

try:
    from .embedding_backend import BasicEmbeddingBackend
except ImportError:
    BasicEmbeddingBackend = None

try:
    from .multihop_backend import MultiHopMemoryBackend
except ImportError:
    MultiHopMemoryBackend = None

try:
    from .chroma_backend import ChromaMemoryBackend
except ImportError:
    ChromaMemoryBackend = None

try:
    from .extractor import TranscriptExtractor
except ImportError:
    TranscriptExtractor = None

try:
    from .sleep import SleepProcessor
except ImportError:
    SleepProcessor = None

__all__ = [
    "MemoryScope",
    "MemoryEntry",
    "SearchResult",
    "MemoryBackend",
    "EmbeddingProvider",
    "SQLiteMemoryBackend",
    "TFIDFMemoryBackend",
    "BasicEmbeddingBackend",
    "MultiHopMemoryBackend",
    "ChromaMemoryBackend",
    "TranscriptExtractor",
    "SleepProcessor",
]
