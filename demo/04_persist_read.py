"""
SCENE 2b — Fresh process reads memories written by 03_persist_write.py.
Run this in a NEW terminal after 03_persist_write.py finishes.
No shared state. Just the DB file on disk.
"""
from cognicore.memory.sqlite_backend import SQLiteMemoryBackend
import time

DB = "demo_persistent.db"

print("=" * 55)
print("  PROCESS B: Reading from disk (fresh process)")
print("=" * 55)
print(f"  Database: {DB}")
print(f"  Timestamp: {time.strftime('%H:%M:%S')}\n")

mem = SQLiteMemoryBackend(DB)
entries = mem.get_all()

print(f"  Entries found: {len(entries)}\n")
for e in entries:
    eid = getattr(e, 'entry_id', getattr(e, 'id', '?'))
    ts = getattr(e, 'timestamp', 0)
    ts_str = time.strftime('%H:%M:%S', time.localtime(ts)) if ts else 'no-timestamp'
    text = getattr(e, 'text', str(e))
    cat = getattr(e, 'category', '—')
    print(f"  [{eid}] {ts_str}")
    print(f"        {text[:65]}")
    print(f"        category={cat}")
    print()

print("  Process A is dead. This process never imported from it.")
print("  Memory is persistent — it lives in SQLite, not RAM.")
