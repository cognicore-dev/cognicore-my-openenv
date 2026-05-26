"""
CogniCore Core API — Type-safe entry points for training and evaluation.

These are the primary user-facing functions. They enforce strict
type contracts and provide clear error messages.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import EpisodeStats
from cognicore.core.errors import (
    AgentInterfaceError,
    CogniCoreError,
)

logger = logging.getLogger("cognicore")


# ---------------------------------------------------------------------------
# Type contracts (runtime-checkable protocols for agents)
# ---------------------------------------------------------------------------

def _validate_agent(agent: Any) -> None:
    """Ensure the agent has the minimum required interface."""
    if not callable(getattr(agent, "act", None)):
        raise AgentInterfaceError(agent, "act")


def _validate_env(env: Any) -> None:
    """Ensure the object is a CogniCoreEnv."""
    if not isinstance(env, CogniCoreEnv):
        raise CogniCoreError(
            f"Expected a CogniCoreEnv instance, got {type(env).__name__}. "
            f"Use cognicore.make('EnvName-v1') to create an environment."
        )


def _validate_episodes(episodes: int) -> None:
    """Ensure episodes is a non-negative integer."""
    if not isinstance(episodes, int) or episodes < 0:
        raise ValueError(
            f"episodes must be a non-negative integer, got {episodes!r}."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train(
    agent: Any,
    env: CogniCoreEnv,
    episodes: int = 10,
    *,
    verbose: bool = False,
) -> Any:
    """Train an agent in a CogniCore environment.

    Parameters
    ----------
    agent : BaseAgent
        An agent implementing ``act(obs) -> action``.
        Optionally implements ``on_reward(reward)`` and
        ``on_episode_end(stats)``.
    env : CogniCoreEnv
        The environment instance (from ``cognicore.make()``).
    episodes : int
        Number of episodes to run. Must be >= 1.
    verbose : bool
        If True, log per-episode stats at INFO level.

    Returns
    -------
    agent
        The same agent (now trained).

    Raises
    ------
    AgentInterfaceError
        If the agent doesn't implement ``act()``.
    CogniCoreError
        If env is not a CogniCoreEnv or episodes < 1.
    """
    _validate_agent(agent)
    _validate_env(env)
    _validate_episodes(episodes)

    logger.info(
        "Training started: agent=%s env=%s episodes=%d",
        type(agent).__name__, type(env).__name__, episodes,
    )

    for ep in range(episodes):
        obs: Dict[str, Any] = env.reset()

        while True:
            action: Dict[str, Any] = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)

            if hasattr(agent, "on_reward"):
                agent.on_reward(reward)

            if done:
                stats: EpisodeStats = env.episode_stats()
                if hasattr(agent, "on_episode_end"):
                    agent.on_episode_end(stats)
                if verbose:
                    logger.info(
                        "  ep=%d accuracy=%.1f%% reward=%.2f memory=%d",
                        ep + 1, stats.accuracy * 100,
                        stats.total_reward, stats.memory_entries_created,
                    )
                break

    logger.info("Training complete.")
    return agent


def evaluate(
    agent: Any,
    env: CogniCoreEnv,
    episodes: int = 5,
) -> float:
    """Evaluate an agent's accuracy in a CogniCore environment.

    Parameters
    ----------
    agent : BaseAgent
        An agent implementing ``act(obs) -> action``.
    env : CogniCoreEnv
        The environment instance.
    episodes : int
        Number of episodes to evaluate. Must be >= 1.

    Returns
    -------
    float
        Average accuracy (0.0 – 1.0) across all episodes.

    Raises
    ------
    AgentInterfaceError
        If the agent doesn't implement ``act()``.
    CogniCoreError
        If env is not a CogniCoreEnv or episodes < 1.
    """
    _validate_agent(agent)
    _validate_env(env)
    _validate_episodes(episodes)

    logger.info(
        "Evaluation started: agent=%s env=%s episodes=%d",
        type(agent).__name__, type(env).__name__, episodes,
    )

    total_score: float = 0.0

    for ep in range(episodes):
        obs: Dict[str, Any] = env.reset()
        while True:
            action: Dict[str, Any] = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                total_score += env.get_score()
                break

    avg: float = total_score / episodes
    logger.info("Evaluation complete. Average score: %.4f", avg)
    return avg
