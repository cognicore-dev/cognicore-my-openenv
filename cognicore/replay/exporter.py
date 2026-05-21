"""
Trajectory Exporter — export events and branches for offline RL training.
Supports JSONL format for easy processing.
"""
import json
from pathlib import Path
from typing import List, Optional
from cognicore.replay.store import EventStore


class TrajectoryExporter:
    """Export stored trajectories for offline RL training."""

    def __init__(self, store: EventStore = None):
        self.store = store or EventStore()

    def export_jsonl(self, task_ids: List[str] = None,
                    output_path: str = "trajectories.jsonl") -> int:
        """Export events as JSONL. Returns number of lines written."""
        if task_ids is None:
            task_ids = self.store.get_task_ids()

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(path, "w") as f:
            for tid in task_ids:
                events = self.store.load_events(tid)
                for e in events:
                    d = e.to_dict()
                    # Remove large fields for training
                    d.pop("input_text", None)
                    d.pop("output_text", None)
                    d.pop("state_vector", None)
                    f.write(json.dumps(d) + "\n")
                    count += 1

        return count

    def export_trajectories(self, task_ids: List[str] = None,
                           output_path: str = "trajectories_full.jsonl") -> int:
        """Export full trajectories (grouped by task) as JSONL."""
        if task_ids is None:
            task_ids = self.store.get_task_ids()

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(path, "w") as f:
            for tid in task_ids:
                events = self.store.load_events(tid)
                if not events:
                    continue

                trajectory = {
                    "task_id": tid,
                    "total_events": len(events),
                    "solved": any(e.event_type == "task_solved" for e in events),
                    "total_tokens": sum(e.tokens_in + e.tokens_out for e in events),
                    "total_cost": sum(e.cost for e in events),
                    "duration": (events[-1].timestamp - events[0].timestamp
                                if len(events) > 1 else 0),
                    "events": [
                        {
                            "step": e.step,
                            "type": e.event_type,
                            "action": e.action,
                            "reward": e.reward,
                            "tokens": e.tokens_in + e.tokens_out,
                            "cost": e.cost,
                            "confidence": e.confidence,
                            "branch_id": e.branch_id,
                        }
                        for e in events
                    ],
                }
                f.write(json.dumps(trajectory) + "\n")
                count += 1

        return count

    def export_branches(self, task_ids: List[str] = None,
                       min_reward: float = None,
                       output_path: str = "branches.jsonl") -> int:
        """Export branch data for branch-decision training."""
        if task_ids is None:
            task_ids = self.store.get_task_ids()

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(path, "w") as f:
            for tid in task_ids:
                branches = self.store.load_branches(tid)
                for b in branches:
                    if min_reward is not None:
                        outcome = b.get("outcome", {})
                        if not outcome.get("solved", False):
                            continue

                    f.write(json.dumps(b) + "\n")
                    count += 1

        return count

    def export_rl_transitions(self, task_ids: List[str] = None,
                             output_path: str = "transitions.jsonl") -> int:
        """Export state-action-reward transitions for DQN training."""
        if task_ids is None:
            task_ids = self.store.get_task_ids()

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(path, "w") as f:
            for tid in task_ids:
                events = self.store.load_events(tid)
                for i in range(len(events) - 1):
                    e = events[i]
                    e_next = events[i + 1]
                    done = (e_next.event_type in
                           ("task_solved", "task_failed"))

                    transition = {
                        "task_id": tid,
                        "step": e.step,
                        "action": e.action,
                        "reward": e.reward,
                        "done": done,
                        "event_type": e.event_type,
                        "next_event_type": e_next.event_type,
                        "tokens": e.tokens_in + e.tokens_out,
                        "confidence": e.confidence,
                    }
                    f.write(json.dumps(transition) + "\n")
                    count += 1

        return count

    def get_export_stats(self) -> dict:
        """Get stats about what's available to export."""
        task_ids = self.store.get_task_ids()
        total_events = 0
        total_branches = 0
        solved = 0

        for tid in task_ids:
            events = self.store.load_events(tid)
            total_events += len(events)
            if any(e.event_type == "task_solved" for e in events):
                solved += 1
            branches = self.store.load_branches(tid)
            total_branches += len(branches)

        return {
            "total_tasks": len(task_ids),
            "total_events": total_events,
            "total_branches": total_branches,
            "solved_tasks": solved,
            "solve_rate": solved / max(len(task_ids), 1),
        }
