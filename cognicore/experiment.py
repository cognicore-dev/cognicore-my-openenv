"""
CogniCore Experiment — A/B testing framework for comparing agents.

Run controlled experiments comparing different agents, prompts,
or configurations on the same environments.

Usage::

    from cognicore.experiment import Experiment

    exp = Experiment("prompt-v1-vs-v2", env_id="SafetyClassification-v1")
    exp.add_variant("v1", agent_v1)
    exp.add_variant("v2", agent_v2)
    results = exp.run(episodes=10)
    results.print_comparison()
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import cognicore
from cognicore.agents.base_agent import RandomAgent
import logging

logger = logging.getLogger("cognicore.experiment")


class Experiment:
    """A/B testing framework for comparing agent variants.

    Runs each variant through the same environments and produces
    statistical comparisons.
    """

    def __init__(
        self,
        name: str,
        env_id: str = "SafetyClassification-v1",
        difficulty: str = "easy",
        **env_kwargs,
    ):
        self.name = name
        self.env_id = env_id
        self.difficulty = difficulty
        self.env_kwargs = env_kwargs
        self.variants: Dict[str, Any] = {}

    def add_variant(self, name: str, agent=None, **config):
        """Add an agent variant to the experiment.

        Parameters
        ----------
        name : str
            Variant name (e.g., "prompt-v1", "temperature-0.5")
        agent : BaseAgent or None
            Agent to test. None uses RandomAgent.
        **config : dict
            Additional config for this variant.
        """
        self.variants[name] = {"agent": agent, "config": config}

    def run(self, episodes: int = 5, verbose: bool = True) -> "ExperimentResults":
        """Run all variants and collect results.

        Parameters
        ----------
        episodes : int
            Number of episodes per variant.
        verbose : bool
            Print progress.
        """
        if verbose:
            logger.info(f"\nExperiment: {self.name}")
            logger.info(f"  Env: {self.env_id} ({self.difficulty})")
            logger.info(f"  Variants: {len(self.variants)} x {episodes} episodes")
            logger.info("-" * 60)

        all_results = {}

        for var_name, var_data in self.variants.items():
            agent = var_data["agent"]
            var_results = []

            for ep in range(episodes):
                env = cognicore.make(
                    self.env_id,
                    difficulty=self.difficulty,
                    **self.env_kwargs,
                )

                if agent is None:
                    _agent = RandomAgent(env.action_space)
                else:
                    _agent = agent

                obs = env.reset()
                total_memory = 0
                total_streak = 0
                start = time.time()

                while True:
                    action = _agent.act(obs)
                    obs, reward, done, _, info = env.step(action)
                    total_memory += reward.memory_bonus
                    total_streak += reward.streak_penalty

                    # RL agents might have learn()
                    if hasattr(_agent, "learn"):
                        _agent.learn(reward, info)

                    if done:
                        break

                stats = env.episode_stats()
                elapsed = time.time() - start

                var_results.append(
                    {
                        "episode": ep + 1,
                        "score": env.get_score(),
                        "accuracy": stats.accuracy,
                        "correct": stats.correct_count,
                        "total": stats.steps,
                        "memory_bonus_total": total_memory,
                        "streak_penalty_total": total_streak,
                        "time_seconds": elapsed,
                    }
                )

                if verbose:
                    print(
                        f"  [{var_name:15s}] ep={ep + 1} "
                        f"acc={stats.accuracy:.0%} "
                        f"score={env.get_score():.4f}"
                    )

            all_results[var_name] = var_results

        return ExperimentResults(self.name, all_results)


class ExperimentResults:
    """Results from an A/B experiment."""

    def __init__(self, name: str, results: Dict[str, List[Dict]]):
        self.name = name
        self.results = results

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Compute summary stats per variant."""
        summary = {}
        for var, runs in self.results.items():
            scores = [r["score"] for r in runs]
            accs = [r["accuracy"] for r in runs]
            mem = [r["memory_bonus_total"] for r in runs]
            summary[var] = {
                "avg_score": sum(scores) / len(scores),
                "best_score": max(scores),
                "avg_accuracy": sum(accs) / len(accs),
                "best_accuracy": max(accs),
                "avg_memory_bonus": sum(mem) / len(mem),
                "episodes": len(runs),
            }
        return summary

    def winner(self) -> str:
        """Return the variant with the highest average score."""
        s = self.summary()
        return max(s, key=lambda k: s[k]["avg_score"])

    def improvement(self) -> Optional[Dict[str, float]]:
        """Calculate improvement of winner over worst."""
        s = self.summary()
        if len(s) < 2:
            return None
        best_name = max(s, key=lambda k: s[k]["avg_score"])
        worst_name = min(s, key=lambda k: s[k]["avg_score"])
        best = s[best_name]
        worst = s[worst_name]

        return {
            "winner": best_name,
            "loser": worst_name,
            "score_improvement": best["avg_score"] - worst["avg_score"],
            "accuracy_improvement": best["avg_accuracy"] - worst["avg_accuracy"],
        }

    def print_comparison(self):
        """Print formatted comparison."""
        s = self.summary()
        w = self.winner()

        logger.info(f"\n{'=' * 65}")
        logger.info(f"  Experiment Results: {self.name}")
        logger.info(f"{'=' * 65}")
        print(
            f"  {'Variant':<20s} {'Avg Score':<12s} {'Best Score':<12s} {'Avg Acc':<10s} {'Memory+':<10s}"
        )
        logger.info(f"  {'-' * 62}")

        for var in sorted(s, key=lambda k: -s[k]["avg_score"]):
            v = s[var]
            marker = " <-- WINNER" if var == w else ""
            print(
                f"  {var:<20s} "
                f"{v['avg_score']:<12.4f} "
                f"{v['best_score']:<12.4f} "
                f"{v['avg_accuracy'] * 100:<9.0f}% "
                f"{v['avg_memory_bonus']:<+10.2f}"
                f"{marker}"
            )

        imp = self.improvement()
        if imp:
            logger.info(f"\n  {imp['winner']} beats {imp['loser']} by:")
            logger.info(f"    Score: +{imp['score_improvement']:.4f}")
            logger.info(f"    Accuracy: +{imp['accuracy_improvement'] * 100:.0f}%")

        logger.info(f"{'=' * 65}\n")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary(),
            "winner": self.winner(),
            "improvement": self.improvement(),
            "raw_results": self.results,
        }
