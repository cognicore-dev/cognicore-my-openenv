"""
AXIOM Agent System — base agent, agent registry, role definitions.
Each agent is a stateless function: (context, task) → (action, artifacts).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import time


class AgentRole(Enum):
    PLANNER = "planner"
    LOCALIZER = "localizer"
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"
    VERIFIER = "verifier"
    MEMORY = "memory"


@dataclass
class AgentResult:
    """Output from an agent execution."""
    role: AgentRole
    success: bool
    output: str = ""
    artifacts: Dict = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    model: str = "rule-based"
    error: str = ""

    @property
    def total_tokens(self):
        return self.tokens_in + self.tokens_out

    @property
    def cost_usd(self):
        # Approximate costs per 1K tokens
        rates = {
            "gpt-4o": 0.005, "gpt-4o-mini": 0.00015,
            "claude-3.5-sonnet": 0.006, "claude-3-haiku": 0.00025,
            "gemini-2.0-flash": 0.0001, "rule-based": 0.0,
        }
        rate = rates.get(self.model, 0.001)
        return self.total_tokens * rate / 1000


@dataclass
class AgentContext:
    """Input context for an agent."""
    task_id: str
    task_description: str
    buggy_code: str = ""
    test_code: str = ""
    relevant_files: List[str] = field(default_factory=list)
    error_trace: str = ""
    previous_patches: List[Dict] = field(default_factory=list)
    memory_hints: List[str] = field(default_factory=list)
    reflection: str = ""
    attempt: int = 1
    budget_remaining: int = 50000  # tokens
    metadata: Dict = field(default_factory=dict)


class BaseAgent:
    """Base class for all AXIOM agents."""

    def __init__(self, role: AgentRole, model: str = "rule-based"):
        self.role = role
        self.model = model
        self.call_count = 0
        self.total_tokens = 0

    def execute(self, ctx: AgentContext) -> AgentResult:
        self.call_count += 1
        t0 = time.perf_counter()
        try:
            result = self._run(ctx)
            result.latency_ms = int((time.perf_counter() - t0) * 1000)
            result.model = self.model
            self.total_tokens += result.total_tokens
            return result
        except Exception as e:
            return AgentResult(
                role=self.role, success=False, error=str(e),
                latency_ms=int((time.perf_counter() - t0) * 1000),
                model=self.model)

    def _run(self, ctx: AgentContext) -> AgentResult:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.role.value}(model={self.model}, calls={self.call_count})"


class AgentRegistry:
    """Registry of available agents. Agents are registered by role."""

    def __init__(self):
        self._agents: Dict[AgentRole, BaseAgent] = {}

    def register(self, agent: BaseAgent):
        self._agents[agent.role] = agent

    def get(self, role: AgentRole) -> Optional[BaseAgent]:
        return self._agents.get(role)

    def all(self) -> List[BaseAgent]:
        return list(self._agents.values())

    def stats(self) -> Dict:
        return {
            a.role.value: {
                "calls": a.call_count,
                "tokens": a.total_tokens,
                "model": a.model
            } for a in self._agents.values()
        }
