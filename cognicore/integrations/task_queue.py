"""
NEXUS Unified Task Queue — foundation for all enterprise integrations.
SQLite-backed priority queue with deduplication, rate limiting, and callbacks.
"""
import sqlite3, json, time, uuid, threading
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field, asdict

from cognicore.utils.sqlite import connect_sqlite

DB_PATH = Path.home() / ".cognicore" / "task_queue.db"
DB_PATH.parent.mkdir(exist_ok=True)


class TaskPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SOLVED = "solved"
    FAILED = "failed"
    DEAD = "dead"  # moved to dead letter queue


class TaskSource(Enum):
    MANUAL = "manual"
    GITHUB_ISSUE = "github_issue"
    CI_FAILURE = "ci_failure"
    SLACK = "slack"
    LINEAR = "linear"
    SCHEDULED = "scheduled"
    PR_REVIEW = "pr_review"


@dataclass
class NexusTask:
    id: str = ""
    source: str = "manual"
    repo: str = ""
    title: str = ""
    description: str = ""
    priority: int = 2
    status: str = "pending"
    policy: str = "test_first"
    max_attempts: int = 3
    budget_usd: float = 2.0
    callback_url: str = ""
    metadata: Dict = field(default_factory=dict)
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    result: Dict = field(default_factory=dict)
    attempts: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = f"nxt-{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


class NexusTaskQueue:
    """SQLite-backed priority task queue with dedup and rate limiting."""

    def __init__(self, db_path=None, max_concurrent=3):
        self.db_path = str(db_path or DB_PATH)
        self.max_concurrent = max_concurrent
        self._lock = threading.Lock()
        self._callbacks: Dict[str, List[Callable]] = {}
        self._persistent_conn = None
        self._init_db()

    def _init_db(self):
        db = self._conn()
        db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                source TEXT,
                repo TEXT,
                title TEXT,
                description TEXT,
                priority INTEGER DEFAULT 2,
                status TEXT DEFAULT 'pending',
                policy TEXT DEFAULT 'test_first',
                max_attempts INTEGER DEFAULT 3,
                budget_usd REAL DEFAULT 2.0,
                callback_url TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}',
                created_at TEXT,
                started_at TEXT DEFAULT '',
                completed_at TEXT DEFAULT '',
                result TEXT DEFAULT '{}',
                attempts INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS dead_letter (
                id TEXT PRIMARY KEY,
                task_json TEXT,
                reason TEXT,
                died_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_priority ON tasks(priority);
            CREATE INDEX IF NOT EXISTS idx_source ON tasks(source);
        """)
        if self.db_path != ":memory:":
            db.close()

    def _conn(self):
        if self.db_path == ":memory:":
            if self._persistent_conn is None:
                self._persistent_conn = sqlite3.connect(":memory:", timeout=30.0)
                self._persistent_conn.row_factory = sqlite3.Row
            return self._persistent_conn
        db = connect_sqlite(self.db_path)
        db.row_factory = sqlite3.Row
        return db

    def _close(self, db):
        if self.db_path != ":memory:":
            db.close()

    def submit(self, task: NexusTask) -> str:
        """Submit a task to the queue. Returns task ID."""
        with self._lock:
            db = self._conn()
            # Deduplication: same repo + title + pending/running = skip
            existing = db.execute(
                "SELECT id FROM tasks WHERE repo=? AND title=? AND status IN ('pending','running')",
                (task.repo, task.title)).fetchone()
            if existing:
                self._close(db)
                return existing["id"]  # already queued

            db.execute("""INSERT INTO tasks
                (id,source,repo,title,description,priority,status,policy,
                 max_attempts,budget_usd,callback_url,metadata,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (task.id, task.source, task.repo, task.title, task.description,
                 task.priority, task.status, task.policy, task.max_attempts,
                 task.budget_usd, task.callback_url, json.dumps(task.metadata),
                 task.created_at))
            db.commit()
            self._close(db)
            self._fire("task_submitted", task)
            return task.id

    def next(self) -> Optional[NexusTask]:
        """Get next task by priority. Returns None if queue empty or at capacity."""
        with self._lock:
            db = self._conn()
            # Check concurrent limit
            running = db.execute(
                "SELECT COUNT(*) as c FROM tasks WHERE status='running'").fetchone()["c"]
            if running >= self.max_concurrent:
                self._close(db)
                return None

            row = db.execute(
                "SELECT * FROM tasks WHERE status='pending' ORDER BY priority ASC, created_at ASC LIMIT 1"
            ).fetchone()
            if not row:
                self._close(db)
                return None

            task = self._row_to_task(row)
            db.execute("UPDATE tasks SET status='running', started_at=? WHERE id=?",
                      (datetime.utcnow().isoformat(), task.id))
            db.commit()
            self._close(db)
            task.status = "running"
            self._fire("task_started", task)
            return task

    def complete(self, task_id: str, solved: bool, result: dict = None):
        """Mark task as solved or failed."""
        with self._lock:
            db = self._conn()
            status = "solved" if solved else "failed"
            now = datetime.utcnow().isoformat()
            db.execute("UPDATE tasks SET status=?, completed_at=?, result=? WHERE id=?",
                      (status, now, json.dumps(result or {}), task_id))

            # Dead letter: move if failed too many times
            row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row and not solved:
                task = self._row_to_task(row)
                task.attempts += 1
                if task.attempts >= task.max_attempts:
                    db.execute("INSERT OR REPLACE INTO dead_letter VALUES (?,?,?,?)",
                              (task_id, json.dumps(asdict(task)), "max_attempts_exceeded", now))
                    db.execute("UPDATE tasks SET status='dead' WHERE id=?", (task_id,))
                else:
                    # Re-queue for retry
                    db.execute("UPDATE tasks SET status='pending', attempts=? WHERE id=?",
                              (task.attempts, task_id))

            db.commit()
            self._close(db)
            task = self._row_to_task(row) if row else NexusTask(id=task_id)
            task.status = status
            self._fire("task_completed", task)

    def list_tasks(self, status=None, source=None, limit=100) -> List[dict]:
        db = self._conn()
        q = "SELECT * FROM tasks WHERE 1=1"
        params = []
        if status:
            q += " AND status=?"
            params.append(status)
        if source:
            q += " AND source=?"
            params.append(source)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = db.execute(q, params).fetchall()
        self._close(db)
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        db = self._conn()
        total = db.execute("SELECT COUNT(*) as c FROM tasks").fetchone()["c"]
        by_status = {}
        for row in db.execute("SELECT status, COUNT(*) as c FROM tasks GROUP BY status"):
            by_status[row["status"]] = row["c"]
        by_source = {}
        for row in db.execute("SELECT source, COUNT(*) as c FROM tasks GROUP BY source"):
            by_source[row["source"]] = row["c"]
        dead = db.execute("SELECT COUNT(*) as c FROM dead_letter").fetchone()["c"]
        self._close(db)
        return {"total": total, "by_status": by_status, "by_source": by_source, "dead_letter": dead}

    def dead_letter_queue(self) -> List[dict]:
        db = self._conn()
        rows = db.execute("SELECT * FROM dead_letter ORDER BY died_at DESC").fetchall()
        self._close(db)
        return [dict(r) for r in rows]

    def clear(self, status=None):
        db = self._conn()
        if status:
            db.execute("DELETE FROM tasks WHERE status=?", (status,))
        else:
            db.execute("DELETE FROM tasks")
            db.execute("DELETE FROM dead_letter")
        db.commit()
        self._close(db)

    def on(self, event: str, callback: Callable):
        self._callbacks.setdefault(event, []).append(callback)

    def _fire(self, event: str, task: NexusTask):
        for cb in self._callbacks.get(event, []):
            try:
                cb(task)
            except Exception:
                pass

    def _row_to_task(self, row) -> NexusTask:
        return NexusTask(
            id=row["id"], source=row["source"], repo=row["repo"],
            title=row["title"], description=row["description"],
            priority=row["priority"], status=row["status"],
            policy=row["policy"], max_attempts=row["max_attempts"],
            budget_usd=row["budget_usd"], callback_url=row["callback_url"],
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=row["created_at"], started_at=row["started_at"] or "",
            completed_at=row["completed_at"] or "",
            result=json.loads(row["result"] or "{}"),
            attempts=row["attempts"])
