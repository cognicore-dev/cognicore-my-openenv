"""
Trajectory Store — records agent coordination trajectories for offline RL.
Each trajectory = (task, sequence of agent actions, rewards, outcome).
"""
import json, time, os, sqlite3
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class TrajectoryStep:
    """Single step in a coordination trajectory."""
    step: int
    agent: str
    action: str        # "generate_patch", "review", "test", etc.
    tactic: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    success: bool = False
    error: str = ""
    reward: float = 0.0  # step reward


@dataclass
class Trajectory:
    """Full coordination trajectory for one task."""
    task_id: str
    category: str
    policy: str
    seed: int = 42
    solved: bool = False
    total_attempts: int = 0
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    total_reward: float = 0.0
    wall_clock_ms: int = 0
    timestamp: float = field(default_factory=time.time)

    def compute_reward(self):
        """Compute trajectory-level reward: R(τ) = r_solve - c_attempt*n - c_token*T/1000"""
        r_solve = 1.0 if self.solved else 0.0
        c_attempt = 0.1 * self.total_attempts
        c_token = 0.01 * self.total_tokens / 1000
        c_time = 0.05 * self.wall_clock_ms / 10000
        self.total_reward = r_solve - c_attempt - c_token - c_time
        return self.total_reward

    def add_step(self, agent: str, action: str, **kwargs):
        step = TrajectoryStep(step=len(self.steps)+1, agent=agent, action=action, **kwargs)
        self.steps.append(step)
        self.total_tokens += step.tokens_in + step.tokens_out
        self.total_cost += step.cost_usd
        return step

    def to_dict(self):
        d = asdict(self)
        return d


class TrajectoryStore:
    """SQLite-backed trajectory store for offline RL training."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), '..', '..',
                                    'cognicore_trajectories.db')
        self.db_path = str(Path(db_path).resolve())
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS trajectories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                category TEXT NOT NULL,
                policy TEXT NOT NULL,
                seed INTEGER DEFAULT 42,
                solved BOOLEAN NOT NULL,
                total_attempts INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0.0,
                total_reward REAL DEFAULT 0.0,
                wall_clock_ms INTEGER DEFAULT 0,
                steps_json TEXT NOT NULL,
                timestamp REAL NOT NULL
            )""")
            conn.execute("""CREATE INDEX IF NOT EXISTS idx_traj_task
                           ON trajectories(task_id)""")
            conn.execute("""CREATE INDEX IF NOT EXISTS idx_traj_policy
                           ON trajectories(policy)""")

    def store(self, traj: Trajectory):
        traj.compute_reward()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO trajectories
                   (task_id, category, policy, seed, solved, total_attempts,
                    total_tokens, total_cost, total_reward, wall_clock_ms,
                    steps_json, timestamp)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (traj.task_id, traj.category, traj.policy, traj.seed,
                 traj.solved, traj.total_attempts, traj.total_tokens,
                 traj.total_cost, traj.total_reward, traj.wall_clock_ms,
                 json.dumps([asdict(s) for s in traj.steps]), traj.timestamp))

    def get_by_policy(self, policy: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trajectories WHERE policy=? ORDER BY timestamp",
                (policy,)).fetchall()
            return [dict(r) for r in rows]

    def get_by_task(self, task_id: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trajectories WHERE task_id=? ORDER BY timestamp",
                (task_id,)).fetchall()
            return [dict(r) for r in rows]

    def compare_policies(self) -> Dict:
        """Compare all policies by mean reward."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT policy,
                          COUNT(*) as n,
                          AVG(total_reward) as mean_reward,
                          SUM(solved) as solved,
                          AVG(total_tokens) as avg_tokens,
                          AVG(total_cost) as avg_cost
                   FROM trajectories GROUP BY policy""").fetchall()
            return {r["policy"]: dict(r) for r in rows}

    def get_stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
            solved = conn.execute("SELECT COUNT(*) FROM trajectories WHERE solved=1").fetchone()[0]
            policies = conn.execute("SELECT DISTINCT policy FROM trajectories").fetchall()
            return {
                "total_trajectories": total,
                "solved": solved,
                "policies": [p[0] for p in policies],
                "db_path": self.db_path,
            }

    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM trajectories")

    def export_for_training(self, path: str = None) -> str:
        """Export trajectories as JSONL for offline RL training."""
        if path is None:
            path = os.path.join(os.path.dirname(self.db_path), 'trajectories_export.jsonl')
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM trajectories ORDER BY timestamp").fetchall()
        with open(path, 'w') as f:
            for r in rows:
                d = dict(r)
                d['steps'] = json.loads(d.pop('steps_json', '[]'))
                f.write(json.dumps(d) + '\n')
        return path
