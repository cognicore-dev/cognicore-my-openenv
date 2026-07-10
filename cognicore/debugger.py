"""
CogniCore AI Debugger — "VS Code for AI behavior"

Debug AI agents like debugging code:
  - Set breakpoints on categories, actions, or conditions
  - Step through decisions one at a time
  - Inspect memory, rewards, and reasoning at any point
  - Modify decisions live
  - Full execution trace

Usage::

    from cognicore.debugger import AIDebugger

    dbg = AIDebugger(env)
    dbg.breakpoint(category="security")  # pause on security tasks
    dbg.breakpoint(condition=lambda obs, action: action["classification"] == "SAFE")
    trace = dbg.run(agent)
    trace.print_trace()
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import cognicore
import logging

logger = logging.getLogger("cognicore.debugger")


class Breakpoint:
    """A debugger breakpoint."""

    def __init__(
        self,
        name: str,
        category: Optional[str] = None,
        action_value: Optional[str] = None,
        condition: Optional[Callable] = None,
        on_wrong_only: bool = False,
    ):
        self.name = name
        self.category = category
        self.action_value = action_value
        self.condition = condition
        self.on_wrong_only = on_wrong_only
        self.hit_count = 0

    def should_break(self, obs: Dict, action: Dict, info: Dict) -> bool:
        """Check if this breakpoint should trigger."""
        # Category match
        if self.category and obs.get("category") != self.category:
            return False

        # Action value match
        if self.action_value:
            action_str = str(action.get("classification", action))
            if action_str != self.action_value:
                return False

        # Wrong-only filter
        if self.on_wrong_only:
            if info.get("eval_result", {}).get("correct", True):
                return False

        # Custom condition
        if self.condition:
            try:
                if not self.condition(obs, action):
                    return False
            except Exception:
                return False

        self.hit_count += 1
        return True


class DebugSnapshot:
    """A snapshot of agent state at a specific step."""

    def __init__(
        self,
        step: int,
        obs: Dict,
        action: Dict,
        reward: Any,
        info: Dict,
        breakpoint_name: str = "",
        memory_state: Dict = None,
    ):
        self.step = step
        self.obs = obs
        self.action = action
        self.reward = reward
        self.info = info
        self.breakpoint_name = breakpoint_name
        self.memory_state = memory_state or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict:
        return {
            "step": self.step,
            "category": self.obs.get("category", "?"),
            "action": self.action,
            "correct": self.info.get("eval_result", {}).get("correct", None),
            "reward": self.reward.total
            if hasattr(self.reward, "total")
            else self.reward,
            "breakpoint": self.breakpoint_name,
            "memory_entries": self.memory_state.get("total_entries", 0),
        }


class ExecutionTrace:
    """Full execution trace of a debug session."""

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.snapshots: List[DebugSnapshot] = []
        self.breakpoint_hits: List[DebugSnapshot] = []

    def add_step(self, step_data: Dict):
        self.steps.append(step_data)

    def add_snapshot(self, snapshot: DebugSnapshot):
        self.snapshots.append(snapshot)
        if snapshot.breakpoint_name:
            self.breakpoint_hits.append(snapshot)

    def get_step(self, step_num: int) -> Optional[Dict]:
        for s in self.steps:
            if s.get("step") == step_num:
                return s
        return None

    def filter_wrong(self) -> List[Dict]:
        return [s for s in self.steps if not s.get("correct", True)]

    def filter_by_category(self, category: str) -> List[Dict]:
        return [s for s in self.steps if s.get("category") == category]

    def decision_tree(self) -> Dict[str, Any]:
        """Build a decision tree from the trace."""
        tree = {}
        for s in self.steps:
            cat = s.get("category", "?")
            action = str(s.get("action", "?"))
            correct = s.get("correct", None)

            if cat not in tree:
                tree[cat] = {"actions": {}, "total": 0}
            tree[cat]["total"] += 1

            if action not in tree[cat]["actions"]:
                tree[cat]["actions"][action] = {"correct": 0, "wrong": 0}
            if correct:
                tree[cat]["actions"][action]["correct"] += 1
            else:
                tree[cat]["actions"][action]["wrong"] += 1

        return tree

    def print_trace(self, max_lines: int = 30):
        """Print formatted execution trace."""
        logger.info(f"\n{'=' * 65}")
        logger.info("  AI Debugger — Execution Trace")
        print(
            f"  {len(self.steps)} steps | {len(self.breakpoint_hits)} breakpoints hit"
        )
        logger.info(f"{'=' * 65}")

        for s in self.steps[:max_lines]:
            icon = (
                " OK" if s.get("correct") else "BRK" if s.get("breakpoint") else "ERR"
            )
            bp = f" [BP: {s['breakpoint']}]" if s.get("breakpoint") else ""
            reward = s.get("reward", 0)
            print(
                f"  {s['step']:3d} [{icon}] {s.get('category', '?'):20s} "
                f"-> {str(s.get('action', '?')):15s} "
                f"reward={reward:+.2f}{bp}"
            )

        if len(self.steps) > max_lines:
            logger.info(f"  ... ({len(self.steps) - max_lines} more steps)")

        # Breakpoint summary
        if self.breakpoint_hits:
            logger.info("\n  Breakpoint Hits:")
            for bp in self.breakpoint_hits[:10]:
                d = bp.to_dict()
                print(
                    f"    Step {d['step']}: [{d['breakpoint']}] "
                    f"category={d['category']} action={d['action']} "
                    f"correct={d['correct']}"
                )

        # Decision tree
        tree = self.decision_tree()
        if tree:
            logger.info("\n  Decision Tree:")
            for cat in sorted(tree.keys()):
                node = tree[cat]
                logger.info(f"    {cat} ({node['total']} cases):")
                for action, counts in node["actions"].items():
                    total = counts["correct"] + counts["wrong"]
                    acc = counts["correct"] / total if total else 0
                    logger.info(f"      -> {action}: {acc:.0%} ({counts['correct']}/{total})")

        logger.info(f"{'=' * 65}\n")


class AIDebugger:
    """Universal AI debugger — VS Code for AI behavior.

    Parameters
    ----------
    env_id : str
        Environment to debug in.
    difficulty : str
        Environment difficulty.
    """

    def __init__(
        self,
        env_id: str = "SafetyClassification-v1",
        difficulty: str = "easy",
        **env_kwargs,
    ):
        self.env_id = env_id
        self.difficulty = difficulty
        self.env_kwargs = env_kwargs
        self.breakpoints: List[Breakpoint] = []
        self._interceptors: List[Callable] = []

    def breakpoint(
        self,
        name: str = "",
        category: Optional[str] = None,
        action_value: Optional[str] = None,
        condition: Optional[Callable] = None,
        on_wrong_only: bool = False,
    ) -> "AIDebugger":
        """Add a breakpoint.

        Parameters
        ----------
        name : str
            Breakpoint name for identification.
        category : str or None
            Break when this category is encountered.
        action_value : str or None
            Break when agent produces this action.
        condition : callable or None
            Custom condition: fn(obs, action) -> bool
        on_wrong_only : bool
            Only break when the agent is wrong.
        """
        if not name:
            name = f"bp_{len(self.breakpoints) + 1}"
            if category:
                name = f"bp_{category}"
            elif on_wrong_only:
                name = "bp_on_wrong"

        bp = Breakpoint(
            name=name,
            category=category,
            action_value=action_value,
            condition=condition,
            on_wrong_only=on_wrong_only,
        )
        self.breakpoints.append(bp)
        return self

    def on_step(self, callback: Callable) -> "AIDebugger":
        """Register a callback for every step.

        callback(step, obs, action, reward, info) -> None
        """
        self._interceptors.append(callback)
        return self

    def run(self, agent=None, verbose: bool = True) -> ExecutionTrace:
        """Run the agent with debugging enabled.

        Parameters
        ----------
        agent : BaseAgent or None
            Agent to debug. None uses RandomAgent.
        verbose : bool
            Print step-by-step output.
        """
        env = cognicore.make(self.env_id, difficulty=self.difficulty, **self.env_kwargs)

        if agent is None:
            from cognicore.agents.base_agent import RandomAgent

            agent = RandomAgent(env.action_space)

        trace = ExecutionTrace()

        if verbose:
            logger.info(f"\n  AI Debugger: {self.env_id} ({self.difficulty})")
            logger.info(f"  Breakpoints: {len(self.breakpoints)}")
            logger.info(f"  {'-' * 50}")

        obs = env.reset()
        step = 0

        while True:
            step += 1

            # Agent acts
            action = agent.act(obs)

            # Step environment
            obs, reward, done, _, info = env.step(action)
            er = info.get("eval_result", {})

            # Build step record
            step_data = {
                "step": step,
                "category": er.get("category", obs.get("category", "?")),
                "action": action,
                "correct": er.get("correct", False),
                "reward": reward.total,
                "memory_bonus": reward.memory_bonus,
                "streak_penalty": reward.streak_penalty,
                "ground_truth": er.get("ground_truth", "?"),
                "breakpoint": None,
            }

            # Check breakpoints
            triggered_bp = None
            for bp in self.breakpoints:
                if bp.should_break(obs, action, info):
                    triggered_bp = bp
                    step_data["breakpoint"] = bp.name
                    break

            trace.add_step(step_data)

            # Create snapshot if breakpoint hit
            if triggered_bp:
                snapshot = DebugSnapshot(
                    step=step,
                    obs=obs,
                    action=action,
                    reward=reward,
                    info=info,
                    breakpoint_name=triggered_bp.name,
                    memory_state={
                        "total_entries": len(env.memory.entries)
                    },
                )
                trace.add_snapshot(snapshot)

                if verbose:
                    print(
                        f"  >>> BREAKPOINT [{triggered_bp.name}] at step {step} | "
                        f"category={step_data['category']} action={action} "
                        f"correct={step_data['correct']}"
                    )

            # Run interceptors
            for cb in self._interceptors:
                try:
                    cb(step, obs, action, reward, info)
                except Exception:
                    pass

            # Agent learns (if it can)
            if hasattr(agent, "learn"):
                agent.learn(reward, info)

            if verbose and not triggered_bp:
                icon = " OK" if step_data["correct"] else "ERR"
                print(
                    f"  {step:3d} [{icon}] {step_data['category']:20s} "
                    f"-> {str(action):30s} reward={reward.total:+.2f}"
                )

            if done:
                break

        if verbose:
            print(
                f"\n  Done: {step} steps, {len(trace.breakpoint_hits)} breakpoints hit"
            )

        return trace
