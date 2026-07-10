"""
CogniCore Base Environment — The abstract base for all cognitive environments.

Every CogniCore environment inherits from ``CogniCoreEnv``.
The base class automatically wires up all cognitive middleware:

  - **Memory** — stores and retrieves past experiences
  - **Reflection** — analyzes failure patterns, provides hints
  - **Structured Rewards** — 8-component reward signal
  - **PROPOSE → Revise** — tentative exploration before commitment
  - **Safety Monitor** — streak detection and health status

Subclasses implement four abstract methods to define their domain:

  - ``_setup()`` — configure action/observation spaces
  - ``_generate_tasks()`` — yield task data for each step
# IDEA: Add support for multi-agent swarm environments
  - ``_evaluate(action)`` — grade the agent's action
  - ``_get_obs()`` — build the raw observation dict
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from cognicore.core.types import (
    CogniCoreConfig,
    EpisodeStats,
    EvalResult,
    ProposalFeedback,
    StructuredReward,
)
from cognicore.memory.tfidf_backend import TFIDFMemoryBackend
from cognicore.memory.lifecycle import MemoryLifecycleManager
from cognicore.middleware.reflection import ReflectionEngine
from cognicore.middleware.rewards import RewardBuilder
from cognicore.middleware.propose_revise import ProposeReviseProtocol
from cognicore.middleware.safety_monitor import SafetyMonitor
from cognicore.core.errors import EpisodeFinishedError

logger = logging.getLogger("cognicore.env")


class CogniCoreEnv(ABC):
    """Abstract base class for all CogniCore environments.

    Provides a Gymnasium-compatible ``reset()`` / ``step()`` interface,
    plus CogniCore-exclusive ``propose()`` / ``revise()`` methods and
    an 8-component ``StructuredReward``.

    Parameters
    ----------
    config : CogniCoreConfig or None
        Configuration for middleware.  Uses defaults if None.
    **kwargs
        Passed to subclass ``_setup()``.

    Example
    -------
    ::

        import cognicore

        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        obs = env.reset()

        while True:
            action = {"classification": "SAFE"}
            obs, reward, done, truncated, info = env.step(action)
            print(reward)  # StructuredReward with 8 components
            if done:
                break
    """

    # Subclasses should set these in _setup()
    observation_space: Any = None
    action_space: Any = None

    def __init__(
        self,
        config: Optional[CogniCoreConfig] = None,
        **kwargs: Any,
    ) -> None:
        self.config = config or CogniCoreConfig()

        # ---- Cognitive Middleware ----
        self.backend = TFIDFMemoryBackend(
            max_size=self.config.memory_max_size,
        )
        self.memory = MemoryLifecycleManager(self.backend)
        self.reflection = ReflectionEngine(
            memory=self.backend,
            min_samples=self.config.reflection_min_samples,
            failure_threshold=self.config.reflection_failure_threshold,
        )
        self.reward_builder = RewardBuilder(
            config=self.config,
            memory=self.backend,
        )
        self.propose_revise = ProposeReviseProtocol(
            memory=self.backend,
            reflection=self.reflection,
            max_proposals=self.config.max_proposals_per_step,
        )
        self.safety_monitor = SafetyMonitor(
            streak_threshold=self.config.streak_threshold,
            streak_penalty=self.config.streak_penalty,
        )

        # ---- Episode State ----
        self._current_step: int = 0
        self._max_steps: int = 0
        self._done: bool = True
        self._truncated: bool = False
        self._episode_count: int = 0
        self._correct_count: int = 0
        self._total_reward: float = 0.0
        self._rewards: List[StructuredReward] = []
        self._step_start_time: Optional[float] = None

        # ---- Stats ----
        self._episode_memory_entries: int = 0
        self._episode_hints_given: int = 0
        self._episode_proposals: int = 0
        self._episode_improvements: int = 0

        # Let the subclass configure itself
        self._setup_kwargs = kwargs
        self._setup(**kwargs)

    # ==================================================================
    # ABSTRACT METHODS — subclasses MUST implement these
    # ==================================================================

    @abstractmethod
    def _setup(self, **kwargs: Any) -> None:
        """Configure the environment.

        Set ``self.observation_space``, ``self.action_space``, load data,
        and do any other initialization.
        """
        ...

    @abstractmethod
    def _generate_tasks(self) -> List[Any]:
        """Return the list of tasks for the current episode.

        Each item becomes the "current task" at the corresponding step.
        The number of tasks determines ``max_steps``.
        """
        ...

    @abstractmethod
    def _evaluate(self, action: Dict[str, Any]) -> EvalResult:
        """Grade the agent's action for the current task.

        Parameters
        ----------
        action : dict
            The agent's action.

        Returns
        -------
        EvalResult
            Must include ``base_score``, ``correct``, ``category``.
        """
        ...

    @abstractmethod
    def _get_obs(self) -> Dict[str, Any]:
        """Build the raw observation dict for the current step.

        The base class will inject memory context and reflection hints
        automatically — just return the domain-specific fields.
        """
        ...

    # ==================================================================
    # PUBLIC API — reset / step / propose / revise / state
    # ==================================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Reset the environment for a new episode.

        Parameters
        ----------
        seed : int or None
            Random seed (reserved for reproducibility).
        **kwargs
            Passed to ``_generate_tasks()`` indirectly (subclass can read
            from ``self`` attributes set here).

        Returns
        -------
        dict
            First observation (with memory context + reflection hints).
        """
        # Update any dynamic settings
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # Generate tasks
        self._tasks = self._generate_tasks()
        self._max_steps = len(self._tasks)
        self._current_step = 0
        self._done = False
        self._truncated = False
        self._episode_count += 1
        self._correct_count = 0
        self._total_reward = 0.0
        self._rewards = []
        self._step_start_time = None

        # Reset per-episode stats
        self._episode_memory_entries = 0
        self._episode_hints_given = 0
        self._episode_proposals = 0
        self._episode_improvements = 0

        # Reset safety monitor streak (but keep cross-episode history)
        self.safety_monitor.reset()

        return self._build_observation()

    def step(
        self,
        action: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], StructuredReward, bool, bool, Dict[str, Any]]:
        """Take one step in the environment.

        Parameters
        ----------
        action : dict
            The agent's action.

        Returns
        -------
        tuple
            ``(observation, reward, terminated, truncated, info)``
            where ``reward`` is a ``StructuredReward``.
        """
        if self._done:
            obs = self._build_observation()
            info = {
                "error": str(EpisodeFinishedError()),
                "step": self._current_step,
                "max_steps": self._max_steps,
            }
            return obs, StructuredReward(), True, False, info

        self._step_start_time = time.time()

        # Start propose-revise protocol for this step
        self.propose_revise.begin_step()

        # Get current task (accessed via self._tasks[self._current_step] when needed)

        # Evaluate the action
        eval_result = self._evaluate(action)

        # --- Cognitive Middleware ---

        # Safety monitor
        streak_penalty = self.safety_monitor.check(eval_result.correct)

        is_novel = eval_result.category != "" and len(self.backend.get_by_category(
            eval_result.category, top_k=1
        )) == 0

        # Check if agent followed reflection hint
        followed_hint = False
        if self.config.enable_reflection and eval_result.category:
            hint = self.reflection.get_hint(eval_result.category)
            if hint:
                self._episode_hints_given += 1

        # Check propose improvement
        proposal_improved = False
        if self.propose_revise.has_pending_proposal:
            action_str = self._extract_action_str(action)
            proposal_improved = self.propose_revise.check_improvement(
                action_str, eval_result.correct
            )
            if proposal_improved:
                self._episode_improvements += 1

        # Build structured reward
        confidence = action.get("confidence")
        reward = self.reward_builder.build(
            eval_result,
            streak_penalty=streak_penalty,
            followed_hint=followed_hint,
            proposal_improved=proposal_improved,
            is_novel_group=is_novel,
            confidence=confidence,
            step_start_time=self._step_start_time,
        )

        if self.config.enable_memory and eval_result.category:
            from cognicore.memory.base import MemoryEntry
            self.memory.store_direct(MemoryEntry(
                text=str(eval_result.predicted),
                category=eval_result.category,
                correct=eval_result.correct,
                action=str(eval_result.predicted),
                metadata={
                    "ground_truth": str(eval_result.ground_truth),
                    "reward": reward.total,
                    "episode": self._episode_count,
                    **eval_result.metadata,
                }
            ))
            self._episode_memory_entries += 1

        # Update counters
        if eval_result.correct:
            self._correct_count += 1
        self._total_reward += reward.total
        self._rewards.append(reward)

        # End propose-revise protocol
        self.propose_revise.end_step()

        # Advance step
        self._current_step += 1
        if self._current_step >= self._max_steps:
            self._done = True

        # Build info dict
        info = {
            "eval_result": {
                "base_score": eval_result.base_score,
                "correct": eval_result.correct,
                "ground_truth": str(eval_result.ground_truth),
                "predicted": str(eval_result.predicted),
                "category": eval_result.category,
            },
            "reward_components": reward.to_dict(),
            "wrong_streak": self.safety_monitor.get_wrong_streak(),
            "agent_status": self.safety_monitor.status(),
            "step": self._current_step,
            "max_steps": self._max_steps,
        }

        return (
            self._build_observation(),
            reward,
            self._done,
            self._truncated,
            info,
        )

    def propose(self, action: Dict[str, Any]) -> ProposalFeedback:
        """Submit a tentative action for feedback (no grading).

        Parameters
        ----------
        action : dict
            The proposed action.

        Returns
        -------
        ProposalFeedback
            Memory context, reflection hints, confidence estimate.
        """
        if not self.config.enable_propose_revise:
            raise RuntimeError(
                "PROPOSE→Revise is disabled. Set config.enable_propose_revise=True."
            )

        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        # Get current task category
        current_task = self._tasks[self._current_step]
        group_value = getattr(current_task, "category", "") or ""
        if isinstance(current_task, dict):
            group_value = current_task.get("category", "")

        self._episode_proposals += 1

        # Ensure the propose-revise protocol has an active step
        if not self.propose_revise._step_active:
            self.propose_revise.begin_step()

        return self.propose_revise.propose(action, group_value)

    def revise(
        self,
        action: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], StructuredReward, bool, bool, Dict[str, Any]]:
        """Submit a revised action after proposal feedback.

        This calls ``step()`` internally — the action IS graded.
        """
        return self.step(action)

    # ==================================================================
    # STATE / STATS
    # ==================================================================

    def state(self) -> Dict[str, Any]:
        """Return full environment state."""
        accuracy = (
            self._correct_count / self._current_step if self._current_step > 0 else 0.0
        )
        return {
            "current_step": self._current_step,
            "max_steps": self._max_steps,
            "total_reward": round(self._total_reward, 4),
            "done": self._done,
            "episode_number": self._episode_count,
            "correct_count": self._correct_count,
            "accuracy": round(accuracy, 4),
            "agent_status": self.safety_monitor.status(),
            "wrong_streak": self.safety_monitor.get_wrong_streak(),
            "memory_stats": self.memory.get_health_report(),
            "reflection_stats": self.reflection.stats(),
            "safety_stats": self.safety_monitor.stats(),
        }

    def episode_stats(self) -> EpisodeStats:
        """Return summary statistics for the current/last episode."""
        return EpisodeStats(
            episode_number=self._episode_count,
            steps=self._current_step,
            total_reward=round(self._total_reward, 4),
            accuracy=(
                round(self._correct_count / self._current_step, 4)
                if self._current_step > 0
                else 0.0
            ),
            correct_count=self._correct_count,
            memory_entries_created=self._episode_memory_entries,
            reflection_hints_given=self._episode_hints_given,
            proposals_made=self._episode_proposals,
            proposal_improvements=self._episode_improvements,
        )

    def get_score(self) -> float:
        """Return normalized score for the episode (0.0–1.0)."""
        if self._max_steps == 0:
            return 0.0
        score = self._total_reward / self._max_steps
        return round(min(max(score, 0.01), 0.99), 4)

    # ==================================================================
    # INTERNALS
    # ==================================================================

    def _build_observation(self) -> Dict[str, Any]:
        """Build observation with cognitive middleware injections."""
        if self._done or self._current_step >= len(self._tasks):
            return {
                "_done": True,
                "step": self._current_step,
                "max_steps": self._max_steps,
            }

        # Get raw observation from subclass
        obs = self._get_obs()

        # Inject memory context
        if self.config.enable_memory:
            current_task = self._tasks[self._current_step]
            group_value = getattr(current_task, "category", "") or ""
            if isinstance(current_task, dict):
                group_value = current_task.get("category", "")

            if group_value:
                obs["memory_context"] = [
                    r.entry.to_dict() for r in self.memory.retrieve(
                        query=str(current_task),
                        task=group_value,
                        top_k=self.config.memory_retrieve_top_k
                    )
                ]

        # Inject reflection hint
        if self.config.enable_reflection:
            current_task = self._tasks[self._current_step]
            group_value = getattr(current_task, "category", "") or ""
            if isinstance(current_task, dict):
                group_value = current_task.get("category", "")

            if group_value:
                hint = self.reflection.get_hint(group_value)
                if hint:
                    obs["reflection_hint"] = hint

        # Inject step info
        obs["step"] = self._current_step
        obs["max_steps"] = self._max_steps

        return obs

    @staticmethod
    def _extract_action_str(action: Dict[str, Any]) -> str:
        """Extract a string representation from an action dict."""
        # Common keys to look for
        for key in ("classification", "answer", "action", "response", "label"):
            if key in action:
                return str(action[key])
        return str(action)

    def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return (
            f"{cls}(episode={self._episode_count}, "
            f"step={self._current_step}/{self._max_steps}, "
            f"done={self._done})"
        )
