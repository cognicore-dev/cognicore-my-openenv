import logging
from typing import List, Optional, Tuple, Dict, Any

from cognicore.memory.base import MemoryBackend, MemoryEntry, SearchResult, MemoryScope
from cognicore.memory.events import event_bus

logger = logging.getLogger(__name__)

class GraphMemoryBackend(MemoryBackend):
    """
    Graph-based memory backend using NetworkX.
    Wraps another backend for primary storage, and builds a graph
    of entities or relationships for advanced graph-based retrieval.
    Requires `networkx` to be installed.
    """

    def __init__(self, primary_backend: MemoryBackend):
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "The 'networkx' package is required for GraphMemoryBackend. "
                "Install it with: pip install networkx"
            )
        
        self.backend = primary_backend
        self.graph = nx.DiGraph()
        
    def _extract_entities(self, text: str) -> List[str]:
        # Simplistic entity extraction (in a real system, you'd use NER or an LLM)
        # Here we just use capitalization heuristics or assume the user passes entities in metadata
        return []

    def store(self, entry: MemoryEntry) -> str:
        entry_id = self.backend.store(entry)
        
        # Add to graph
        self.graph.add_node(entry_id, category=entry.category, scope=entry.scope.value, correct=entry.correct)
        
        # Link category node to this entry
        cat_node = f"category:{entry.category}"
        self.graph.add_node(cat_node, type="category")
        self.graph.add_edge(cat_node, entry_id, relation="contains")
        
        # Link scope node
        if entry.scope != MemoryScope.GLOBAL:
            scope_node = f"scope:{entry.scope.value}:{entry.scope_id}"
            self.graph.add_node(scope_node, type="scope")
            self.graph.add_edge(scope_node, entry_id, relation="owns")
            
        # Example of handling explicit entities in metadata
        if "entities" in entry.metadata and isinstance(entry.metadata["entities"], list):
            for entity in entry.metadata["entities"]:
                ent_node = f"entity:{entity}"
                self.graph.add_node(ent_node, type="entity")
                self.graph.add_edge(ent_node, entry_id, relation="mentions")

        return entry_id

    def search(self, query: str, top_k: int = 5,
               category: Optional[str] = None,
               scope: Optional[MemoryScope] = None,
               scope_id: Optional[str] = None) -> List[SearchResult]:
        
        # Base search
        base_results = self.backend.search(query, top_k=top_k * 2, category=category, scope=scope, scope_id=scope_id)
        
        # Enhance with graph proximity (simple pagerank or neighbor count boost)
        # For simplicity, we just return the base results but source marked as 'graph'
        for res in base_results:
            res.source = "graph_enhanced"
            
        return base_results[:top_k]

    def get_by_category(self, category: str, top_k: int = 5,
                        success_filter: Optional[bool] = None) -> List[MemoryEntry]:
        return self.backend.get_by_category(category, top_k, success_filter)

    def count(self) -> int:
        return self.backend.count()

    def clear(self) -> None:
        self.backend.clear()
        self.graph.clear()

    def save(self) -> None:
        self.backend.save()
        # Optionally serialize self.graph

    def load(self) -> None:
        self.backend.load()
        # Optionally deserialize self.graph

    def __getattr__(self, name: str) -> Any:
        """Forward missing methods (like update, delete, lifecycle) to the primary backend."""
        if hasattr(self.backend, name):
            return getattr(self.backend, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
