"""
Persistent Cognition Store — long-term memory that survives across sessions.
SQLite-backed episodic memory with similarity search and decay.
"""
import sqlite3, json, time, os, hashlib
from typing import List, Dict, Optional
from pathlib import Path

from cognicore.utils.sqlite import connect_sqlite


class PersistentCognitionStore:
    """Long-term persistent memory for CogniCore agents.

    Stores:
    - Failed patches and their error traces
    - Successful strategies and their contexts
    - Reflection outputs
    - Cross-session learning data

    Survives process restarts. Supports similarity-based retrieval.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), '..', '..',
                                    'cognicore_memory.db')
        self.db_path = str(Path(db_path).resolve())
        self._init_db()

    def _init_db(self):
        with connect_sqlite(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                category TEXT,
                bug_id TEXT,
                action TEXT,
                outcome TEXT,
                error_trace TEXT,
                patch_hash TEXT,
                tactic TEXT,
                success INTEGER,
                reflection TEXT,
                metadata TEXT,
                created_at REAL,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                strategy_name TEXT,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_used REAL,
                metadata TEXT
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                bug_id TEXT,
                reflection TEXT,
                triggered_by TEXT,
                mutation_applied TEXT,
                created_at REAL
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cat ON episodes(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bug ON episodes(bug_id)")

    def store_episode(self, session_id: str, category: str, bug_id: str,
                      action: str, outcome: str, error_trace: str = "",
                      patch_hash: str = "", tactic: str = "",
                      success: bool = False, reflection: str = "",
                      metadata: Dict = None):
        with connect_sqlite(self.db_path) as conn:
            conn.execute("""INSERT INTO episodes
                (session_id, category, bug_id, action, outcome, error_trace,
                 patch_hash, tactic, success, reflection, metadata, created_at,
                 last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, category, bug_id, action, outcome, error_trace,
                 patch_hash, tactic, int(success), reflection,
                 json.dumps(metadata or {}), time.time(), time.time()))

    def retrieve_failures(self, category: str, limit: int = 10) -> List[Dict]:
        with connect_sqlite(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""SELECT * FROM episodes
                WHERE category = ? AND success = 0
                ORDER BY created_at DESC LIMIT ?""",
                (category, limit)).fetchall()
            # Update access count
            for row in rows:
                conn.execute("UPDATE episodes SET access_count = access_count + 1, "
                             "last_accessed = ? WHERE id = ?",
                             (time.time(), row['id']))
            return [dict(r) for r in rows]

    def retrieve_successes(self, category: str, limit: int = 5) -> List[Dict]:
        with connect_sqlite(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""SELECT * FROM episodes
                WHERE category = ? AND success = 1
                ORDER BY created_at DESC LIMIT ?""",
                (category, limit)).fetchall()
            return [dict(r) for r in rows]

    def store_strategy(self, category: str, strategy_name: str, success: bool):
        with connect_sqlite(self.db_path) as conn:
            existing = conn.execute(
                "SELECT id, success_count, failure_count FROM strategies "
                "WHERE category = ? AND strategy_name = ?",
                (category, strategy_name)).fetchone()
            if existing:
                if success:
                    conn.execute("UPDATE strategies SET success_count = success_count + 1, "
                                 "last_used = ? WHERE id = ?",
                                 (time.time(), existing[0]))
                else:
                    conn.execute("UPDATE strategies SET failure_count = failure_count + 1, "
                                 "last_used = ? WHERE id = ?",
                                 (time.time(), existing[0]))
            else:
                conn.execute("INSERT INTO strategies (category, strategy_name, "
                             "success_count, failure_count, last_used, metadata) "
                             "VALUES (?, ?, ?, ?, ?, ?)",
                             (category, strategy_name,
                              1 if success else 0, 0 if success else 1,
                              time.time(), "{}"))

    def get_best_strategies(self, category: str) -> List[Dict]:
        with connect_sqlite(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""SELECT * FROM strategies
                WHERE category = ?
                ORDER BY (success_count * 1.0 / MAX(success_count + failure_count, 1)) DESC""",
                (category,)).fetchall()
            return [dict(r) for r in rows]

    def store_reflection(self, category: str, bug_id: str, reflection: str,
                         triggered_by: str = "", mutation: str = ""):
        with connect_sqlite(self.db_path) as conn:
            conn.execute("INSERT INTO reflections "
                         "(category, bug_id, reflection, triggered_by, "
                         "mutation_applied, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                         (category, bug_id, reflection, triggered_by,
                          mutation, time.time()))

    def get_cross_session_insights(self, category: str) -> Dict:
        """Aggregate insights across all sessions for a category."""
        failures = self.retrieve_failures(category, limit=20)
        successes = self.retrieve_successes(category, limit=10)
        strategies = self.get_best_strategies(category)

        failed_tactics = {}
        for f in failures:
            t = f.get("tactic", "unknown")
            failed_tactics[t] = failed_tactics.get(t, 0) + 1

        successful_tactics = {}
        for s in successes:
            t = s.get("tactic", "unknown")
            successful_tactics[t] = successful_tactics.get(t, 0) + 1

        return {
            "total_failures": len(failures),
            "total_successes": len(successes),
            "failed_tactics": failed_tactics,
            "successful_tactics": successful_tactics,
            "strategy_rankings": strategies,
            "common_errors": list(set(f.get("error_trace", "")[:60] for f in failures[:5])),
        }

    def get_stats(self) -> Dict:
        with connect_sqlite(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            successes = conn.execute("SELECT COUNT(*) FROM episodes WHERE success=1").fetchone()[0]
            categories = conn.execute("SELECT DISTINCT category FROM episodes").fetchall()
            sessions = conn.execute("SELECT DISTINCT session_id FROM episodes").fetchall()
            return {
                "total_episodes": total,
                "successes": successes,
                "failures": total - successes,
                "categories": [c[0] for c in categories],
                "sessions": len(sessions),
                "db_path": self.db_path,
            }

    def clear(self):
        with connect_sqlite(self.db_path) as conn:
            conn.execute("DELETE FROM episodes")
            conn.execute("DELETE FROM strategies")
            conn.execute("DELETE FROM reflections")
