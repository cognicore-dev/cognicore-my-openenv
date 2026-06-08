"""
Branch Comparator — compare outcomes across branches.
Finds divergence points, calculates cost/token diffs.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from cognicore.replay.store import EventStore
from cognicore.replay.brancher import Branch, TaskBrancher


@dataclass
class Comparison:
    task_id: str = ""
    branches: List[dict] = field(default_factory=list)
    winner: str = ""                # branch_id of best outcome
    divergence_step: int = 0
    diff_summary: str = ""
    cost_diff: float = 0.0
    token_diff: int = 0
    time_diff: float = 0.0
    solve_rates: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "branches": self.branches,
            "winner": self.winner,
            "divergence_step": self.divergence_step,
            "diff_summary": self.diff_summary,
            "cost_diff": round(self.cost_diff, 4),
            "token_diff": self.token_diff,
            "time_diff": round(self.time_diff, 2),
            "solve_rates": self.solve_rates,
        }


class BranchComparator:
    """Compare branch outcomes to learn what works."""

    def __init__(self, store: EventStore = None):
        self.store = store or EventStore()
        self.brancher = TaskBrancher(self.store)

    def compare(self, task_id: str,
               branch_ids: List[str] = None) -> Comparison:
        """Compare branches for a task. If branch_ids is None, compare all."""
        all_branches = self.brancher.list_branches(task_id)
        if branch_ids:
            all_branches = [b for b in all_branches
                          if b.branch_id in branch_ids]

        if not all_branches:
            return Comparison(task_id=task_id)

        comp = Comparison(task_id=task_id)

        # Build branch summaries
        for b in all_branches:
            comp.branches.append(b.to_dict())
            comp.solve_rates[b.branch_id] = b.solved

        # Find winner (solved + cheapest)
        solved = [b for b in all_branches if b.solved]
        if solved:
            winner = min(solved, key=lambda b: b.cost)
            comp.winner = winner.branch_id
        elif all_branches:
            # No one solved — pick least-cost
            comp.winner = min(all_branches, key=lambda b: b.cost).branch_id

        # Find divergence point
        if len(all_branches) >= 2:
            comp.divergence_step = self._find_divergence(
                task_id, all_branches[0], all_branches[1])

        # Cost/token diffs
        if len(all_branches) >= 2:
            costs = [b.cost for b in all_branches]
            tokens = [b.tokens_used for b in all_branches]
            durations = [b.duration for b in all_branches]
            comp.cost_diff = max(costs) - min(costs)
            comp.token_diff = max(tokens) - min(tokens)
            comp.time_diff = max(durations) - min(durations)

        # Summary
        n_solved = sum(1 for b in all_branches if b.solved)
        comp.diff_summary = (
            f"{len(all_branches)} branches compared. "
            f"{n_solved}/{len(all_branches)} solved. "
            f"Winner: {comp.winner}. "
            f"Diverged at step {comp.divergence_step}."
        )

        return comp

    def _find_divergence(self, task_id: str,
                        branch_a: Branch, branch_b: Branch) -> int:
        """Find the step where two branches first diverge."""
        events_a = self.store.load_events(
            task_id, branch_id=branch_a.branch_id)
        events_b = self.store.load_events(
            task_id, branch_id=branch_b.branch_id)

        # Also check main branch events
        if not events_a:
            events_a = self.store.load_events(task_id, branch_id="main")
        if not events_b:
            events_b = self.store.load_events(task_id, branch_id="main")

        min_len = min(len(events_a), len(events_b))
        for i in range(min_len):
            if (events_a[i].action != events_b[i].action or
                events_a[i].output_hash != events_b[i].output_hash):
                return events_a[i].step

        # Use branch points if no event-level divergence found
        return max(branch_a.branch_point, branch_b.branch_point)

    def rank_branches(self, task_id: str) -> List[dict]:
        """Rank all branches by quality score."""
        branches = self.brancher.list_branches(task_id)
        scored = []
        for b in branches:
            score = 0.0
            if b.solved:
                score += 10.0
            # Penalize cost
            score -= b.cost * 100
            # Penalize tokens
            score -= b.tokens_used / 10000.0
            # Penalize duration
            score -= b.duration / 60.0

            scored.append({
                "branch_id": b.branch_id,
                "solved": b.solved,
                "score": round(score, 3),
                "cost": b.cost,
                "tokens": b.tokens_used,
                "duration": b.duration,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored
