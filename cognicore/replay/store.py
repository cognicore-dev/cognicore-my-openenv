"""
Event Store — SQLite-backed persistent storage for agent events and branches.
WAL mode for concurrent reads/writes. Indexed for fast replay.
"""
import sqlite3
import json
import time
from pathlib import Path
from typing import List, Optional
import numpy as np

from cognicore.replay.recorder import AgentEvent
from cognicore.utils.sqlite import connect_sqlite

DB_DIR = Path.home() / ".cognicore" / "replay"
DB_DIR.mkdir(parents=True, exist_ok=True)


class EventStore:
    """SQLite event store with WAL mode for performance."""

    def __init__(self, db_path=None):
        self.db_path = db_path or str(DB_DIR / "events.db")
        self._init_db()

    def _init_db(self):
        conn = self._conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                branch_id TEXT DEFAULT 'main',
                parent_id TEXT DEFAULT '',
                seq INTEGER,
                step INTEGER,
                timestamp REAL,
                event_type TEXT,
                agent TEXT,
                input_hash TEXT,
                output_hash TEXT,
                input_text TEXT,
                output_text TEXT,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                cost REAL DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                state_vector BLOB,
                action TEXT,
                reward REAL DEFAULT 0,
                value_est REAL DEFAULT 0,
                model TEXT,
                temperature REAL DEFAULT 0,
                policy TEXT,
                memory_hits INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_task_seq
            ON events(task_id, seq)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_task_branch
            ON events(task_id, branch_id)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS branches (
                branch_id TEXT PRIMARY KEY,
                parent_task_id TEXT,
                branch_point INTEGER,
                modifications TEXT,
                outcome TEXT,
                created_at REAL,
                solved INTEGER DEFAULT 0,
                tokens_used INTEGER DEFAULT 0,
                cost REAL DEFAULT 0,
                duration REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_branch_task
            ON branches(parent_task_id)
        """)
        conn.commit()
        conn.close()

    def _conn(self):
        return connect_sqlite(self.db_path)

    def save_event(self, event: AgentEvent):
        """Persist a single event."""
        sv = event.state_vector
        sv_blob = sv.tobytes() if sv is not None else None

        conn = self._conn()
        conn.execute("""
            INSERT OR REPLACE INTO events
            (event_id, task_id, branch_id, parent_id, seq, step, timestamp,
             event_type, agent, input_hash, output_hash, input_text, output_text,
             tokens_in, tokens_out, cost, latency_ms, state_vector,
             action, reward, value_est, model, temperature, policy,
             memory_hits, confidence)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            event.event_id, event.task_id, event.branch_id, event.parent_id,
            event.seq, event.step, event.timestamp,
            event.event_type, event.agent, event.input_hash, event.output_hash,
            event.input_text, event.output_text,
            event.tokens_in, event.tokens_out, event.cost, event.latency_ms,
            sv_blob,
            event.action, event.reward, event.value_est,
            event.model, event.temperature, event.policy,
            event.memory_hits, event.confidence))
        conn.commit()
        conn.close()

    def load_events(self, task_id: str, branch_id: str = None,
                   from_step: int = 0,
                   to_step: int = None) -> List[AgentEvent]:
        """Load events for a task, optionally filtered by branch and step range."""
        conn = self._conn()
        query = "SELECT * FROM events WHERE task_id=?"
        params = [task_id]

        if branch_id:
            query += " AND branch_id=?"
            params.append(branch_id)
        if from_step > 0:
            query += " AND step>=?"
            params.append(from_step)
        if to_step is not None:
            query += " AND step<=?"
            params.append(to_step)

        query += " ORDER BY seq ASC"
        rows = conn.execute(query, params).fetchall()
        conn.close()
        return [self._row_to_event(r) for r in rows]

    def get_task_ids(self) -> List[str]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT DISTINCT task_id FROM events ORDER BY timestamp DESC"
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    def save_branch(self, branch_data: dict):
        """Save branch metadata."""
        conn = self._conn()
        conn.execute("""
            INSERT OR REPLACE INTO branches
            (branch_id, parent_task_id, branch_point, modifications,
             outcome, created_at, solved, tokens_used, cost, duration)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            branch_data["branch_id"],
            branch_data.get("parent_task_id", ""),
            branch_data.get("branch_point", 0),
            json.dumps(branch_data.get("modifications", {})),
            json.dumps(branch_data.get("outcome", {})),
            branch_data.get("created_at", time.time()),
            int(branch_data.get("solved", False)),
            branch_data.get("tokens_used", 0),
            branch_data.get("cost", 0.0),
            branch_data.get("duration", 0.0)))
        conn.commit()
        conn.close()

    def load_branches(self, task_id: str) -> List[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM branches WHERE parent_task_id=? ORDER BY created_at",
            (task_id,)).fetchall()
        conn.close()
        return [{
            "branch_id": r[0], "parent_task_id": r[1],
            "branch_point": r[2],
            "modifications": json.loads(r[3] or "{}"),
            "outcome": json.loads(r[4] or "{}"),
            "created_at": r[5], "solved": bool(r[6]),
            "tokens_used": r[7], "cost": r[8], "duration": r[9],
        } for r in rows]

    def get_stats(self) -> dict:
        conn = self._conn()
        total_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        total_tasks = conn.execute(
            "SELECT COUNT(DISTINCT task_id) FROM events").fetchone()[0]
        total_branches = conn.execute(
            "SELECT COUNT(*) FROM branches").fetchone()[0]
        total_tokens = conn.execute(
            "SELECT SUM(tokens_in + tokens_out) FROM events").fetchone()[0] or 0
        total_cost = conn.execute(
            "SELECT SUM(cost) FROM events").fetchone()[0] or 0
        conn.close()
        return {
            "total_events": total_events,
            "total_tasks": total_tasks,
            "total_branches": total_branches,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 4),
        }

    def delete_task(self, task_id: str):
        conn = self._conn()
        conn.execute("DELETE FROM events WHERE task_id=?", (task_id,))
        conn.execute("DELETE FROM branches WHERE parent_task_id=?", (task_id,))
        conn.commit()
        conn.close()

    def _row_to_event(self, row) -> AgentEvent:
        sv = None
        if row[17]:
            try:
                sv = np.frombuffer(row[17], dtype=np.float32)
            except Exception:
                pass
        return AgentEvent(
            event_id=row[0], task_id=row[1], branch_id=row[2],
            parent_id=row[3], seq=row[4], step=row[5], timestamp=row[6],
            event_type=row[7], agent=row[8],
            input_hash=row[9], output_hash=row[10],
            input_text=row[11], output_text=row[12],
            tokens_in=row[13], tokens_out=row[14],
            cost=row[15], latency_ms=row[16],
            state_vector=sv,
            action=row[18], reward=row[19], value_est=row[20],
            model=row[21], temperature=row[22], policy=row[23],
            memory_hits=row[24], confidence=row[25])

    def clear(self):
        conn = self._conn()
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM branches")
        conn.commit()
        conn.close()
