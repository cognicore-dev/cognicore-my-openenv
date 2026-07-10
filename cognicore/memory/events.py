import logging
from typing import Any, Callable, Dict, List

from cognicore.memory.base import MemoryEntry, SearchResult

logger = logging.getLogger(__name__)

class MemoryEventBus:
    """
    Publish-subscribe event bus for memory observability.
    Allows registering observers that listen to memory events
    such as store and search.
    """

    def __init__(self):
        self._observers: Dict[str, List[Callable[..., None]]] = {
            "on_store": [],
            "on_search": [],
        }

    def subscribe(self, event_name: str, callback: Callable[..., None]) -> None:
        if event_name not in self._observers:
            self._observers[event_name] = []
        self._observers[event_name].append(callback)

    def publish(self, event_name: str, **kwargs: Any) -> None:
        if event_name not in self._observers:
            return
        for callback in self._observers[event_name]:
            try:
                callback(**kwargs)
            except Exception as e:
                logger.warning(f"Error in memory event observer '{callback.__name__}': {e}")

# Global event bus singleton for the memory system
event_bus = MemoryEventBus()
