import pytest
from typing import List, Optional
from cognicore.memory.base import MemoryBackend, MemoryEntry, SearchResult, MemoryScope
from cognicore.memory.tfidf_backend import TFIDFMemoryBackend
from cognicore.memory.hybrid_backend import HybridMemoryBackend

class DummyDenseBackend(MemoryBackend):
    """A dummy dense backend that returns predefined mock results for search."""
    def __init__(self):
        self.stored_entries = []
        self._count = 0

    def store(self, entry: MemoryEntry) -> str:
        if not entry.entry_id:
            entry.entry_id = f"dense_{self._count}"
            self._count += 1
        self.stored_entries.append(entry)
        return entry.entry_id

    def search(self, query: str, top_k: int = 5,
               category: Optional[str] = None,
               scope: Optional[MemoryScope] = None,
               scope_id: Optional[str] = None,
               question_timestamp: Optional[float] = None) -> List[SearchResult]:
        # Just return stored entries in reverse order as search results
        results = []
        for idx, entry in enumerate(reversed(self.stored_entries)):
            results.append(SearchResult(entry=entry, score=1.0 - (idx * 0.1), source="dense"))
        return results[:top_k]

    def get_by_category(self, category: str, top_k: int = 5,
                        success_filter: Optional[bool] = None) -> List[MemoryEntry]:
        return [e for e in self.stored_entries if e.category == category][:top_k]

    def count(self) -> int:
        return len(self.stored_entries)

    def clear(self) -> None:
        self.stored_entries.clear()

    def update(self, entry_id: str, **fields) -> bool:
        for entry in self.stored_entries:
            if entry.entry_id == entry_id:
                for k, v in fields.items():
                    if hasattr(entry, k):
                        setattr(entry, k, v)
                return True
        return False

    def delete(self, entry_id: str) -> bool:
        for i, entry in enumerate(self.stored_entries):
            if entry.entry_id == entry_id:
                self.stored_entries.pop(i)
                return True
        return False


def test_hybrid_fallback():
    # Test that it falls back to TFIDF when dense backend is None
    sparse = TFIDFMemoryBackend()
    hybrid = HybridMemoryBackend(sparse_backend=sparse, dense_backend=None)
    
    e1 = MemoryEntry(text="hello world", category="test")
    hybrid.store(e1)
    
    assert hybrid.count() == 1
    results = hybrid.search("world")
    assert len(results) == 1
    assert results[0].entry.text == "hello world"


def test_hybrid_rrf_and_boosting():
    sparse = TFIDFMemoryBackend()
    dense = DummyDenseBackend()
    hybrid = HybridMemoryBackend(sparse_backend=sparse, dense_backend=dense)
    
    # Store three entries
    e1 = MemoryEntry(text="apple pie recipe", category="cooking", utility_score=0.8)
    e2 = MemoryEntry(text="banana split recipe", category="cooking", utility_score=-0.5)
    e3 = MemoryEntry(text="cherry pie recipe", category="cooking", utility_score=0.2)
    
    hybrid.store(e1)
    hybrid.store(e2)
    hybrid.store(e3)
    
    assert hybrid.count() == 3
    assert dense.count() == 3
    
    # Perform search. Sparse search will rank them by TF-IDF (e.g. apple pie, cherry pie).
    # Dense backend (DummyDenseBackend) returns them in reverse stored order: cherry (e3), banana (e2), apple (e1).
    results = hybrid.search("recipe", top_k=3)
    
    # Verify that results are retrieved and fused
    assert len(results) == 3
    
    # e1 (apple pie) has high utility (0.8), e2 (banana) has negative utility (-0.5).
    # Apple pie should be boosted relative to banana split due to utility score
    apple_res = [r for r in results if "apple" in r.entry.text][0]
    banana_res = [r for r in results if "banana" in r.entry.text][0]
    assert apple_res.score > banana_res.score


def test_propagation():
    sparse = TFIDFMemoryBackend()
    dense = DummyDenseBackend()
    hybrid = HybridMemoryBackend(sparse_backend=sparse, dense_backend=dense)
    
    e = MemoryEntry(text="test propagation", category="sys")
    eid = hybrid.store(e)
    
    # Test update propagates to both
    assert hybrid.update(eid, category="updated_sys") is True
    assert sparse.get_by_id(eid).category == "updated_sys"
    assert dense.stored_entries[0].category == "updated_sys"
    
    # Test delete propagates to both
    assert hybrid.delete(eid) is True
    assert hybrid.count() == 0
    assert dense.count() == 0
