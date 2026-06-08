"""
Task Replayer — reconstructs any past agent state from events.
Deterministic replay (from logs) or live replay (fresh LLM calls).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from cognicore.replay.recorder import AgentEvent
from cognicore.replay.store import EventStore


@dataclass
class ReplaySession:
    """A replay session that can be played, paused, stepped through."""
    events: List[AgentEvent] = field(default_factory=list)
    current_step: int = 0
    mode: str = "deterministic"
    speed: float = 1.0
    paused: bool = False

    @property
    def total_steps(self):
        return len(self.events)

    @property
    def current_event(self) -> Optional[AgentEvent]:
        if 0 <= self.current_step < len(self.events):
            return self.events[self.current_step]
        return None

    @property
    def is_done(self):
        return self.current_step >= len(self.events)

    @property
    def progress(self):
        return self.current_step / max(len(self.events), 1)

    def play(self):
        self.paused = False

    def pause(self):
        self.paused = True

    def step_forward(self) -> Optional[AgentEvent]:
        if self.current_step < len(self.events):
            event = self.events[self.current_step]
            self.current_step += 1
            return event
        return None

    def step_back(self) -> Optional[AgentEvent]:
        if self.current_step > 0:
            self.current_step -= 1
            return self.events[self.current_step]
        return None

    def jump_to(self, step: int) -> Optional[AgentEvent]:
        step = max(0, min(step, len(self.events) - 1))
        self.current_step = step
        return self.events[step] if self.events else None

    def get_state_at(self, step: int) -> dict:
        """Reconstruct full state at a given step by replaying events."""
        state = {
            "task_id": "",
            "step": step,
            "events_so_far": [],
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_cost": 0.0,
            "patches_generated": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "last_action": "",
            "last_output": "",
            "policy": "",
            "branch_id": "main",
        }

        for i, e in enumerate(self.events):
            if i > step:
                break
            state["task_id"] = e.task_id
            state["events_so_far"].append(e.event_type)
            state["total_tokens_in"] += e.tokens_in
            state["total_tokens_out"] += e.tokens_out
            state["total_cost"] += e.cost
            state["last_action"] = e.action
            state["last_output"] = e.output_text
            state["policy"] = e.policy or state["policy"]
            state["branch_id"] = e.branch_id

            if e.event_type == "patch_generated":
                state["patches_generated"] += 1
            elif e.event_type == "test_passed":
                state["tests_passed"] += 1
            elif e.event_type == "test_failed":
                state["tests_failed"] += 1

        return state

    def to_dict(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "mode": self.mode,
            "progress": round(self.progress, 3),
            "is_done": self.is_done,
            "paused": self.paused,
        }


class TaskReplayer:
    """Reconstructs any past agent state from stored events."""

    def __init__(self, store: EventStore = None):
        self.store = store or EventStore()

    def replay(self, task_id: str, mode: str = "deterministic",
              from_step: int = 0, to_step: int = None,
              branch_id: str = None,
              speed: float = 1.0) -> ReplaySession:
        """Load events and create a replay session."""
        events = self.store.load_events(
            task_id, branch_id=branch_id,
            from_step=from_step, to_step=to_step)

        session = ReplaySession(
            events=events, mode=mode, speed=speed)
        return session

    def replay_to_step(self, task_id: str, step: int,
                      branch_id: str = None) -> dict:
        """Reconstruct exact state at step N. Ready for branching."""
        session = self.replay(task_id, branch_id=branch_id, to_step=step)
        return session.get_state_at(step)

    def get_event_at(self, task_id: str, step: int) -> Optional[AgentEvent]:
        """Get the specific event at a step."""
        events = self.store.load_events(task_id, from_step=step, to_step=step)
        return events[0] if events else None

    def list_tasks(self) -> List[str]:
        return self.store.get_task_ids()

    def get_task_summary(self, task_id: str) -> dict:
        """Summary of a task's execution."""
        events = self.store.load_events(task_id)
        if not events:
            return {"task_id": task_id, "events": 0}

        return {
            "task_id": task_id,
            "total_events": len(events),
            "first_event": events[0].timestamp,
            "last_event": events[-1].timestamp,
            "duration": events[-1].timestamp - events[0].timestamp,
            "total_tokens": sum(e.tokens_in + e.tokens_out for e in events),
            "total_cost": sum(e.cost for e in events),
            "event_types": list(set(e.event_type for e in events)),
            "solved": any(e.event_type == "task_solved" for e in events),
            "branches": list(set(e.branch_id for e in events)),
        }
