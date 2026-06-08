"""
Token Tracker — tracks token usage, enforces budgets, computes costs.
"""
from dataclasses import dataclass, field
from typing import Dict, List
import time


@dataclass
class TokenEvent:
    agent: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    task_id: str
    attempt: int
    timestamp: float = field(default_factory=time.time)


class TokenTracker:
    """Tracks all token usage across agents and tasks."""

    # Cost per 1K tokens (input+output blended)
    MODEL_COSTS = {
        "gpt-4o": 0.005, "gpt-4o-mini": 0.00015,
        "claude-3.5-sonnet": 0.006, "claude-3-haiku": 0.00025,
        "gemini-2.0-flash": 0.0001, "rule-based": 0.0,
    }

    def __init__(self, budget_tokens: int = 500000, budget_usd: float = 10.0):
        self.budget_tokens = budget_tokens
        self.budget_usd = budget_usd
        self.events: List[TokenEvent] = []

    def record(self, agent: str, model: str, tokens_in: int, tokens_out: int,
               task_id: str = "", attempt: int = 0):
        total = tokens_in + tokens_out
        rate = self.MODEL_COSTS.get(model, 0.001)
        cost = total * rate / 1000
        self.events.append(TokenEvent(
            agent=agent, model=model, tokens_in=tokens_in,
            tokens_out=tokens_out, cost_usd=cost,
            task_id=task_id, attempt=attempt))

    @property
    def total_tokens(self) -> int:
        return sum(e.tokens_in + e.tokens_out for e in self.events)

    @property
    def total_cost(self) -> float:
        return sum(e.cost_usd for e in self.events)

    @property
    def budget_remaining_tokens(self) -> int:
        return max(0, self.budget_tokens - self.total_tokens)

    @property
    def budget_remaining_usd(self) -> float:
        return max(0.0, self.budget_usd - self.total_cost)

    def is_over_budget(self) -> bool:
        return self.total_tokens >= self.budget_tokens or self.total_cost >= self.budget_usd

    def per_agent(self) -> Dict:
        stats = {}
        for e in self.events:
            if e.agent not in stats:
                stats[e.agent] = {"calls": 0, "tokens": 0, "cost": 0.0}
            stats[e.agent]["calls"] += 1
            stats[e.agent]["tokens"] += e.tokens_in + e.tokens_out
            stats[e.agent]["cost"] += e.cost_usd
        return stats

    def per_task(self) -> Dict:
        stats = {}
        for e in self.events:
            if e.task_id not in stats:
                stats[e.task_id] = {"tokens": 0, "cost": 0.0, "attempts": 0}
            stats[e.task_id]["tokens"] += e.tokens_in + e.tokens_out
            stats[e.task_id]["cost"] += e.cost_usd
            stats[e.task_id]["attempts"] = max(stats[e.task_id]["attempts"], e.attempt)
        return stats

    def summary(self) -> str:
        pa = self.per_agent()
        lines = [f"  Tokens: {self.total_tokens:,} | Cost: ${self.total_cost:.4f} | "
                 f"Budget: {self.budget_remaining_tokens:,} tokens / ${self.budget_remaining_usd:.2f}"]
        for agent, s in pa.items():
            lines.append(f"    {agent}: {s['calls']} calls, {s['tokens']:,} tokens, ${s['cost']:.4f}")
        return "\n".join(lines)
