"""
SCENE 2a — Write memories to disk, then EXIT.
Run this first. Then run 04_persist_read.py in a FRESH terminal.
Proves memory survives process death — stored in SQLite, not RAM.
"""
from cognicore.memory.sqlite_backend import SQLiteMemoryBackend
from cognicore.memory.base import MemoryEntry, MemoryScope
import time

DB = "demo_persistent.db"
mem = SQLiteMemoryBackend(DB)

memories = [
    ("FastAPI is the web framework for this project", "stack", MemoryScope.GLOBAL),
    ("Prefer TypeScript over JavaScript", "preference", MemoryScope.GLOBAL),
    ("Bug #42: null pointer in detect_encoding fixed with early return guard", "bugfix", MemoryScope.GLOBAL),
    ("Railway deployment: set JWT_SECRET and MEMORY_DB_PATH env vars", "devops", MemoryScope.GLOBAL),
]

print("=" * 55)
print("  PROCESS A: Writing memories to disk")
print("=" * 55)
print(f"  Database: {DB}\n")

for text, category, scope in memories:
    entry = MemoryEntry(text=text, category=category, scope=scope)
    entry_id = mem.store(entry)
    ts = time.strftime("%H:%M:%S")
    print(f"  [{ts}] stored id={entry_id}  [{category}]")
    print(f"         {text[:60]}")
    print()

print(f"  Total entries written: {mem.count()}")
print()
print("  >>> Process A exiting. Run 04_persist_read.py now. <<<")
