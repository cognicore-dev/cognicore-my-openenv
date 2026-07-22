"""
SCENE 3 — Semantic Search with BM25 ranking.
Shows natural-language queries matched against stored memories.
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from cognicore.memory.sqlite_backend import SQLiteMemoryBackend
from cognicore.memory.base import MemoryEntry, MemoryScope
import time

DB = "demo_persistent.db"
mem = SQLiteMemoryBackend(DB)

# Add more entries (real timestamps from actual execution)
extra = [
    ("Fixed crash when content is None -- added early return in detect_encoding", "bugfix"),
    ("Search latency dropped from 120ms to 8ms after switching to BM25", "performance"),
    ("Kaushalt prefers dark mode IDE themes like Dracula or Tokyo Night", "preference"),
]
for text, cat in extra:
    mem.store(MemoryEntry(text=text, category=cat, scope=MemoryScope.GLOBAL))
    time.sleep(0.3)

queries = [
    "what web framework does this project use",
    "authentication security bug",
    "speed performance improvement",
    "who prefers dark mode",
    "null pointer crash fix",
    "cloud deployment configuration",
]

print("=" * 60)
print("  SEMANTIC SEARCH -- BM25 Okapi Ranking")
print("=" * 60)
print()

for q in queries:
    print(f'  Query: "{q}"')
    print("  " + "-" * 56)
    results = mem.search(q, top_k=2)
    if results:
        for r in results:
            entry = getattr(r, 'entry', r)
            text = getattr(entry, 'text', str(r))[:52]
            score = getattr(r, 'score', getattr(r, 'relevance', 0.0))
            eid = getattr(entry, 'entry_id', getattr(entry, 'id', '?'))
            print(f"  [{eid:>3}] score={score:6.3f}  {text}")
    else:
        print("  (no results)")
    print()
