"""
Timeline Visualizer — generates JSON data for the dashboard /replay page.
Timeline view, branch tree, step inspector.
"""
from typing import List, Dict
from cognicore.replay.store import EventStore
from cognicore.replay.brancher import TaskBrancher


# Color coding for event types
EVENT_COLORS = {
    "task_start": "#6366f1",      # indigo
    "memory_retrieved": "#8b5cf6", # violet
    "plan_generated": "#3b82f6",   # blue
    "patch_generated": "#10b981",  # emerald
    "patch_rejected": "#ef4444",   # red
    "test_executed": "#f59e0b",    # amber
    "test_passed": "#22c55e",      # green
    "test_failed": "#ef4444",      # red
    "branch_created": "#06b6d4",   # cyan
    "task_solved": "#22c55e",      # green
    "task_failed": "#ef4444",      # red
    "immune_blocked": "#f97316",   # orange
    "checkpoint": "#64748b",       # slate
}

EVENT_ICONS = {
    "task_start": "🚀",
    "memory_retrieved": "🧠",
    "plan_generated": "📋",
    "patch_generated": "🔧",
    "patch_rejected": "❌",
    "test_executed": "🧪",
    "test_passed": "✅",
    "test_failed": "❌",
    "branch_created": "🔀",
    "task_solved": "🎉",
    "task_failed": "💥",
    "immune_blocked": "🛡️",
    "checkpoint": "📌",
}


class TimelineVisualizer:
    """Generates JSON-serializable visualization data for the dashboard."""

    def __init__(self, store: EventStore = None):
        self.store = store or EventStore()
        self.brancher = TaskBrancher(self.store)

    def generate_timeline(self, task_id: str,
                         branch_id: str = None) -> dict:
        """Generate timeline data for a task."""
        events = self.store.load_events(task_id, branch_id=branch_id)

        timeline_items = []
        cumulative_cost = 0.0
        cumulative_tokens = 0

        for e in events:
            cumulative_cost += e.cost
            cumulative_tokens += e.tokens_in + e.tokens_out

            item = {
                "id": e.event_id,
                "step": e.step,
                "seq": e.seq,
                "timestamp": e.timestamp,
                "type": e.event_type,
                "agent": e.agent,
                "action": e.action,
                "output_preview": (e.output_text[:100] + "..."
                                  if len(e.output_text) > 100
                                  else e.output_text),
                "tokens": e.tokens_in + e.tokens_out,
                "cost": round(e.cost, 6),
                "latency_ms": e.latency_ms,
                "confidence": e.confidence,
                "branch_id": e.branch_id,
                "color": EVENT_COLORS.get(e.event_type, "#94a3b8"),
                "icon": EVENT_ICONS.get(e.event_type, "•"),
                "cumulative_cost": round(cumulative_cost, 6),
                "cumulative_tokens": cumulative_tokens,
            }
            timeline_items.append(item)

        # Cost accumulation data for chart
        cost_series = [{"step": t["step"], "cost": t["cumulative_cost"],
                       "tokens": t["cumulative_tokens"]}
                      for t in timeline_items]

        return {
            "task_id": task_id,
            "branch_id": branch_id or "main",
            "total_events": len(events),
            "total_cost": round(cumulative_cost, 6),
            "total_tokens": cumulative_tokens,
            "duration": (events[-1].timestamp - events[0].timestamp
                        if len(events) > 1 else 0),
            "solved": any(e.event_type == "task_solved" for e in events),
            "timeline": timeline_items,
            "cost_series": cost_series,
        }

    def generate_branch_tree(self, task_id: str) -> dict:
        """Generate branch tree visualization data."""
        branches = self.brancher.list_branches(task_id)
        main_events = self.store.load_events(task_id, branch_id="main")

        # Build tree
        root = {
            "id": "main",
            "label": "Main Branch",
            "solved": any(e.event_type == "task_solved" for e in main_events),
            "steps": len(main_events),
            "cost": sum(e.cost for e in main_events),
            "tokens": sum(e.tokens_in + e.tokens_out for e in main_events),
            "color": "#22c55e" if any(
                e.event_type == "task_solved" for e in main_events
            ) else "#ef4444",
            "children": [],
        }

        for b in branches:
            branch_events = self.store.load_events(
                task_id, branch_id=b.branch_id)
            child = {
                "id": b.branch_id,
                "label": f"Branch @ step {b.branch_point}",
                "branch_point": b.branch_point,
                "solved": b.solved,
                "steps": len(branch_events),
                "cost": b.cost,
                "tokens": b.tokens_used,
                "duration": b.duration,
                "modifications": b.modifications,
                "color": "#22c55e" if b.solved else "#ef4444",
                "children": [],
            }
            root["children"].append(child)

        return {
            "task_id": task_id,
            "tree": root,
            "total_branches": len(branches),
            "solved_branches": sum(1 for b in branches if b.solved),
        }

    def generate_step_detail(self, task_id: str, step: int,
                            branch_id: str = None) -> dict:
        """Generate detailed view for a specific step."""
        events = self.store.load_events(
            task_id, branch_id=branch_id,
            from_step=step, to_step=step)

        if not events:
            return {"step": step, "found": False}

        e = events[0]
        return {
            "step": step,
            "found": True,
            "event_id": e.event_id,
            "type": e.event_type,
            "agent": e.agent,
            "timestamp": e.timestamp,
            "input": e.input_text,
            "output": e.output_text,
            "action": e.action,
            "tokens_in": e.tokens_in,
            "tokens_out": e.tokens_out,
            "cost": e.cost,
            "latency_ms": e.latency_ms,
            "confidence": e.confidence,
            "reward": e.reward,
            "model": e.model,
            "policy": e.policy,
            "memory_hits": e.memory_hits,
            "branch_id": e.branch_id,
            "color": EVENT_COLORS.get(e.event_type, "#94a3b8"),
            "icon": EVENT_ICONS.get(e.event_type, "•"),
        }
