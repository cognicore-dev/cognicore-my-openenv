"""
Threat Memory — SQLite-backed store of all threats seen by the immune system.
Records attacks, outcomes, and provides historical context for the RL defender.
"""
import sqlite3, json, time, hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import numpy as np

from cognicore.utils.sqlite import connect_sqlite

DB_DIR = Path.home() / ".cognicore" / "immune"
DB_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ThreatRecord:
    threat_id: str = ""
    input_hash: str = ""
    category: str = ""
    action_taken: int = 0
    was_correct: bool = True
    confidence: float = 0.0
    features: Optional[np.ndarray] = None
    timestamp: float = 0.0
    metadata: Dict = field(default_factory=dict)


class ThreatMemory:
    """Persistent threat memory with SQLite storage."""

    def __init__(self, db_path=None):
        self.db_path = db_path or str(DB_DIR / "threats.db")
        self._init_db()

    def _init_db(self):
        conn = connect_sqlite(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS threats (
                threat_id TEXT PRIMARY KEY,
                input_hash TEXT,
                category TEXT,
                action_taken INTEGER,
                was_correct INTEGER,
                confidence REAL,
                features BLOB,
                timestamp REAL,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON threats(category)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON threats(timestamp)
        """)
        conn.commit()
        conn.close()

    def record_threat(self, input_text: str, category: str,
                     action_taken: int, was_correct: bool,
                     confidence: float = 0.0,
                     features: np.ndarray = None,
                     metadata: dict = None):
        """Record a threat encounter."""
        threat_id = hashlib.sha256(
            f"{input_text}{time.time()}".encode()).hexdigest()[:16]
        input_hash = hashlib.sha256(input_text.encode()).hexdigest()[:32]

        feat_blob = features.tobytes() if features is not None else b""

        conn = connect_sqlite(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO threats
            (threat_id, input_hash, category, action_taken, was_correct,
             confidence, features, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (threat_id, input_hash, category, action_taken,
              int(was_correct), confidence, feat_blob,
              time.time(), json.dumps(metadata or {})))
        conn.commit()
        conn.close()
        return threat_id

    def get_similar_threats(self, features: np.ndarray,
                           top_k: int = 5) -> List[ThreatRecord]:
        """Find threats with similar feature vectors."""
        conn = connect_sqlite(self.db_path)
        rows = conn.execute(
            "SELECT * FROM threats WHERE length(features) > 0 "
            "ORDER BY timestamp DESC LIMIT 200"
        ).fetchall()
        conn.close()

        if not rows:
            return []

        results = []
        for row in rows:
            stored_feat = np.frombuffer(row[6], dtype=np.float32)
            if len(stored_feat) == len(features):
                sim = np.dot(features, stored_feat) / (
                    np.linalg.norm(features) * np.linalg.norm(stored_feat) + 1e-8)
                rec = ThreatRecord(
                    threat_id=row[0], input_hash=row[1], category=row[2],
                    action_taken=row[3], was_correct=bool(row[4]),
                    confidence=row[5], features=stored_feat,
                    timestamp=row[7], metadata=json.loads(row[8] or "{}"))
                results.append((sim, rec))

        results.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in results[:top_k]]

    def get_threats_by_category(self, category: str) -> List[ThreatRecord]:
        conn = connect_sqlite(self.db_path)
        rows = conn.execute(
            "SELECT * FROM threats WHERE category=? ORDER BY timestamp DESC",
            (category,)).fetchall()
        conn.close()
        return [self._row_to_record(r) for r in rows]

    def get_stats(self) -> Dict:
        conn = connect_sqlite(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM threats").fetchone()[0]
        blocked = conn.execute(
            "SELECT COUNT(*) FROM threats WHERE action_taken=1").fetchone()[0]
        correct = conn.execute(
            "SELECT COUNT(*) FROM threats WHERE was_correct=1").fetchone()[0]
        cats = conn.execute(
            "SELECT category, COUNT(*) FROM threats GROUP BY category"
        ).fetchall()
        recent = conn.execute(
            "SELECT * FROM threats ORDER BY timestamp DESC LIMIT 10"
        ).fetchall()
        conn.close()

        return {
            "total_threats": total,
            "total_blocked": blocked,
            "accuracy": correct / max(total, 1),
            "categories": {c: n for c, n in cats},
            "recent": [self._row_to_record(r).__dict__ for r in recent],
        }

    def _row_to_record(self, row) -> ThreatRecord:
        feat = np.frombuffer(row[6], dtype=np.float32) if row[6] else None
        return ThreatRecord(
            threat_id=row[0], input_hash=row[1], category=row[2],
            action_taken=row[3], was_correct=bool(row[4]),
            confidence=row[5], features=feat,
            timestamp=row[7], metadata=json.loads(row[8] or "{}"))

    def clear(self):
        conn = connect_sqlite(self.db_path)
        conn.execute("DELETE FROM threats")
        conn.commit()
        conn.close()
