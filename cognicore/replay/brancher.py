"""
Task Brancher — create branches from any point in history.
Try different decisions, compare outcomes. RL learns which branches win.
"""
import uuid
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from cognicore.replay.store import EventStore
from cognicore.replay.replayer import TaskReplayer
from cognicore.replay.recorder import AgentEvent, EventType


@dataclass
class Branch:
    branch_id: str = ""
    parent_task_id: str = ""
    branch_point: int = 0   # step number where branch diverges
    modifications: dict = field(default_factory=dict)
    outcome: dict = field(default_factory=dict)
    created_at: float = 0.0
    solved: bool = False
    tokens_used: int = 0
    cost: float = 0.0
    duration: float = 0.0
    events: List[AgentEvent] = field(default_factory=list)

    def __post_init__(self):
        if not self.branch_id:
            self.branch_id = f"br_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "branch_id": self.branch_id,
            "parent_task_id": self.parent_task_id,
            "branch_point": self.branch_point,
            "modifications": self.modifications,
            "outcome": self.outcome,
            "created_at": self.created_at,
            "solved": self.solved,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "duration": self.duration,
            "event_count": len(self.events),
        }


class TaskBrancher:
    """Create branches from any point in task history."""

    def __init__(self, store: EventStore = None, replayer: TaskReplayer = None):
        self.store = store or EventStore()
        self.replayer = replayer or TaskReplayer(self.store)

    def branch(self, task_id: str, from_step: int,
              modifications: dict = None,
              new_policy: str = None) -> Branch:
        """
        Create a new branch from a specific step in a task's history.

        Args:
            task_id: The task to branch from
            from_step: Step number to branch at
            modifications: Changes to apply at branch point
            new_policy: Different policy to use from this point

        Returns:
            Branch object with reconstructed state
        """
        # Reconstruct state at branch point
        state = self.replayer.replay_to_step(task_id, from_step)

        branch = Branch(
            parent_task_id=task_id,
            branch_point=from_step,
            modifications=modifications or {},
        )

        # Copy events up to branch point into new branch
        original_events = self.store.load_events(
            task_id, to_step=from_step)

        for event in original_events:
            branched = AgentEvent(
                task_id=task_id,
                branch_id=branch.branch_id,
                parent_id=event.branch_id,
                seq=event.seq,
                step=event.step,
                timestamp=event.timestamp,
                event_type=event.event_type,
                agent=event.agent,
                input_text=event.input_text,
                output_text=event.output_text,
                tokens_in=event.tokens_in,
                tokens_out=event.tokens_out,
                cost=event.cost,
                latency_ms=event.latency_ms,
                action=event.action,
                reward=event.reward,
                policy=new_policy or event.policy,
                confidence=event.confidence,
                state_vector=event.state_vector)
            branch.events.append(branched)

        # Record branch creation event
        branch_event = AgentEvent(
            task_id=task_id,
            branch_id=branch.branch_id,
            event_type=EventType.BRANCH_CREATED,
            step=from_step,
            action=f"branch_from_step_{from_step}",
            output_text=f"Branch {branch.branch_id} from step {from_step}")
        self.store.save_event(branch_event)

        # Save branch metadata
        self.store.save_branch(branch.to_dict())

        return branch

    def record_branch_outcome(self, branch: Branch, solved: bool,
                             tokens: int = 0, cost: float = 0.0,
                             duration: float = 0.0):
        """Record the outcome of a branch execution."""
        branch.solved = solved
        branch.tokens_used = tokens
        branch.cost = cost
        branch.duration = duration
        branch.outcome = {
            "solved": solved,
            "tokens_used": tokens,
            "cost": cost,
            "duration": duration,
        }
        self.store.save_branch(branch.to_dict())

    def list_branches(self, task_id: str) -> List[Branch]:
        """List all branches for a task."""
        raw = self.store.load_branches(task_id)
        branches = []
        for r in raw:
            b = Branch(
                branch_id=r["branch_id"],
                parent_task_id=r["parent_task_id"],
                branch_point=r["branch_point"],
                modifications=r["modifications"],
                outcome=r["outcome"],
                created_at=r["created_at"],
                solved=r["solved"],
                tokens_used=r["tokens_used"],
                cost=r["cost"],
                duration=r["duration"])
            branches.append(b)
        return branches

    def get_branch(self, task_id: str, branch_id: str) -> Optional[Branch]:
        """Get a specific branch."""
        branches = self.list_branches(task_id)
        for b in branches:
            if b.branch_id == branch_id:
                # Load events for this branch
                b.events = self.store.load_events(
                    task_id, branch_id=branch_id)
                return b
        return None
