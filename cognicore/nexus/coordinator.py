"""
AXIOM Coordination Bus — routes tasks to agents, manages pipeline execution.
Supports static routing (hardcoded pipeline) and learned routing (future).
"""
from typing import List, Dict, Optional, Tuple
from cognicore.nexus.agent import AgentRole, AgentContext, AgentResult, AgentRegistry
from cognicore.nexus.token_tracker import TokenTracker
import time


class CoordinationEvent:
    def __init__(self, step: int, role: AgentRole, result: AgentResult, timestamp: float = None):
        self.step = step
        self.role = role
        self.result = result
        self.timestamp = timestamp or time.time()


class CoordinationBus:
    """Routes tasks through agent pipelines with budget enforcement."""

    # Static routing policies
    POLICIES = {
        "standard": [
            AgentRole.MEMORY, AgentRole.PLANNER, AgentRole.CODER,
            AgentRole.REVIEWER, AgentRole.TESTER,
        ],
        "review_first": [
            AgentRole.MEMORY, AgentRole.PLANNER, AgentRole.CODER,
            AgentRole.REVIEWER, AgentRole.CODER, AgentRole.TESTER,
        ],
        "test_first": [
            AgentRole.MEMORY, AgentRole.PLANNER, AgentRole.CODER,
            AgentRole.TESTER, AgentRole.REVIEWER,
        ],
        "minimal": [
            AgentRole.CODER, AgentRole.TESTER,
        ],
    }

    def __init__(self, registry: AgentRegistry, tracker: TokenTracker,
                 policy: str = "standard", max_messages: int = 10):
        self.registry = registry
        self.tracker = tracker
        self.policy = policy
        self.max_messages = max_messages
        self.history: List[CoordinationEvent] = []

    def execute_pipeline(self, ctx: AgentContext,
                         max_attempts: int = 3) -> Tuple[bool, List[CoordinationEvent]]:
        """Execute the full agent pipeline for a task."""
        pipeline = self.POLICIES.get(self.policy, self.POLICIES["standard"])
        events = []

        for attempt in range(1, max_attempts + 1):
            ctx.attempt = attempt
            ctx.budget_remaining = self.tracker.budget_remaining_tokens

            if self.tracker.is_over_budget():
                break

            step = 0
            patch = None
            review_passed = True

            for role in pipeline:
                agent = self.registry.get(role)
                if agent is None:
                    continue

                step += 1
                if step > self.max_messages:
                    break

                result = agent.execute(ctx)

                # Track tokens
                self.tracker.record(
                    agent=role.value, model=result.model,
                    tokens_in=result.tokens_in, tokens_out=result.tokens_out,
                    task_id=ctx.task_id, attempt=attempt)

                event = CoordinationEvent(step, role, result)
                events.append(event)
                self.history.append(event)

                # Process results based on role
                if role == AgentRole.MEMORY:
                    if result.artifacts.get("hints"):
                        ctx.memory_hints = result.artifacts["hints"]
                    if result.artifacts.get("reflection"):
                        ctx.reflection = result.artifacts["reflection"]

                elif role == AgentRole.PLANNER:
                    if result.artifacts.get("plan"):
                        ctx.metadata["plan"] = result.artifacts["plan"]

                elif role == AgentRole.CODER:
                    patch = result.output
                    ctx.metadata["current_patch"] = patch

                elif role == AgentRole.REVIEWER:
                    review_passed = result.success
                    if not review_passed:
                        ctx.error_trace = result.output
                        break  # Back to next attempt

                elif role == AgentRole.TESTER:
                    if result.success:
                        return True, events
                    else:
                        ctx.error_trace = result.error or result.output
                        ctx.previous_patches.append({
                            "patch": patch[:300] if patch else "",
                            "error": ctx.error_trace[:200],
                            "attempt": attempt,
                        })
                        break  # Next attempt

        return False, events

    def get_stats(self) -> Dict:
        return {
            "total_steps": len(self.history),
            "policy": self.policy,
            "token_summary": self.tracker.summary(),
        }
