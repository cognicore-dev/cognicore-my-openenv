"""Shared SQLite connection helpers for CogniCore."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union


def connect_sqlite(path: Union[str, Path], timeout: float = 30.0) -> sqlite3.Connection:
    """Open a SQLite connection with contention-friendly defaults.

    For file-backed databases we enable WAL and a matching busy timeout so
    readers and writers can coexist more reliably under concurrent load.
    """
    db_path = str(path)
    conn = sqlite3.connect(db_path, timeout=timeout)

    if db_path != ":memory:":
        busy_timeout_ms = int(timeout * 1000)
        conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        
        # Retry WAL mode since it requires an exclusive lock and can bypass the busy_timeout on Windows
        import time
        for _ in range(10):
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as e:
                if "locked" in str(e):
                    time.sleep(0.1)
                else:
                    raise
                    
        conn.execute("PRAGMA synchronous=NORMAL")

    return conn
