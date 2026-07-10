import os
import pytest
from cognicore.memory.base import MemoryEntry, MemoryState, MemoryType
from cognicore.memory.tfidf_backend import TFIDFMemoryBackend
from cognicore.memory.sleep import SleepProcessor


def test_deduplication():
    backend = TFIDFMemoryBackend()
    processor = SleepProcessor(backend=backend, similarity_threshold=0.7)
    
    # Create two duplicate entries
    e1 = MemoryEntry(
        text="The server crashed with a segfault error",
        memory_type=MemoryType.FAILURE.value,
        category="backend",
        state=MemoryState.ACTIVE.value,
        utility_score=0.5
    )
    e2 = MemoryEntry(
        text="Server crashed with a segfault error",
        memory_type=MemoryType.FAILURE.value,
        category="backend",
        state=MemoryState.ACTIVE.value,
        utility_score=0.3
    )
    
    backend.store(e1)
    backend.store(e2)
    
    assert backend.count() == 2
    
    stats = processor.sleep()
    assert stats["merged"] == 1
    
    # One entry should remain, and its properties should be combined
    remaining = backend.get_all()
    assert len(remaining) == 1
    assert remaining[0].utility_score == 0.4  # Average of 0.5 and 0.3


def test_contradiction_resolution():
    backend = TFIDFMemoryBackend()
    processor = SleepProcessor(backend=backend)
    
    # Create contradictory preference entries in the same category
    e1 = MemoryEntry(
        text="I prefer spaces over tabs",
        memory_type=MemoryType.PREFERENCE.value,
        category="formatting",
        state=MemoryState.ACTIVE.value,
        utility_score=0.8
    )
    e2 = MemoryEntry(
        text="I prefer tabs over spaces",
        memory_type=MemoryType.PREFERENCE.value,
        category="formatting",
        state=MemoryState.ACTIVE.value,
        utility_score=0.2
    )
    
    backend.store(e1)
    backend.store(e2)
    
    stats = processor.sleep()
    assert stats["archived_contradictions"] == 1
    
    # One of them should be archived
    remaining = backend.get_all()
    assert len(remaining) == 2
    
    active = [e for e in remaining if e.state == MemoryState.ACTIVE.value]
    archived = [e for e in remaining if e.state == MemoryState.ARCHIVED.value]
    
    assert len(active) == 1
    assert len(archived) == 1
    assert active[0].text == "I prefer spaces over tabs"  # Keep higher utility


def test_episodic_compression():
    backend = TFIDFMemoryBackend()
    processor = SleepProcessor(backend=backend)
    
    # Create 5 episodic memories in the same category
    for i in range(5):
        e = MemoryEntry(
            text=f"Fixed compilation issue by changing Makefile line {i}",
            memory_type=MemoryType.EPISODIC.value,
            category="build",
            state=MemoryState.ACTIVE.value,
            correct=True
        )
        backend.store(e)
        
    assert backend.count() == 5
    
    stats = processor.sleep()
    assert stats["compressed_episodes"] == 3  # All but the last 2
    
    all_entries = backend.get_all()
    # 5 original entries (3 archived, 2 active) + 1 new semantic entry = 6 total
    assert len(all_entries) == 6
    
    active_semantic = [e for e in all_entries if e.memory_type == MemoryType.SEMANTIC.value]
    assert len(active_semantic) == 1
    assert "Consolidated History" in active_semantic[0].text
