import logging
from typing import Any, List

from cognicore.memory.base import MemoryEntry, SearchResult
from cognicore.memory.events import event_bus

logger = logging.getLogger(__name__)

class LangfuseObserver:
    """
    Observer that logs CogniCore memory events to Langfuse.
    Requires `langfuse` to be installed.
    """

    def __init__(self, public_key: str = None, secret_key: str = None, host: str = None):
        try:
            from langfuse import Langfuse
        except ImportError:
            raise ImportError(
                "The 'langfuse' package is required for LangfuseObserver. "
                "Install it with: pip install langfuse"
            )
        
        # Will pick up LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST if kwargs are None
        self.langfuse = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        
        # Automatically register
        event_bus.subscribe("on_store", self.on_store)
        event_bus.subscribe("on_search", self.on_search)
        logger.info("LangfuseObserver initialized and subscribed to memory events.")

    def on_store(self, entry: MemoryEntry, entry_id: str, **kwargs: Any) -> None:
        """Log memory storage to Langfuse."""
        self.langfuse.trace(
            name="memory_store",
            metadata={
                "entry_id": entry_id,
                "category": entry.category,
                "scope": entry.scope.value,
                "correct": entry.correct,
            },
            input=entry.text,
            output=entry_id
        )

    def on_search(self, query: str, top_k: int, results: List[SearchResult], **kwargs: Any) -> None:
        """Log memory search to Langfuse."""
        self.langfuse.trace(
            name="memory_search",
            metadata={
                "top_k": top_k,
                "num_results": len(results),
            },
            input=query,
            output=[{"entry_id": r.entry.entry_id, "score": r.score} for r in results]
        )
