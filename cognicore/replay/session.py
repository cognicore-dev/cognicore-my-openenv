"""
CogniCore Session Replay — Record and replay full episodes.

Records every step of an episode for later replay, analysis, or export.

Usage::

    from cognicore.replay import SessionRecorder, replay

    rec = SessionRecorder("SafetyClassification-v1")
    rec.record(agent, episodes=3)
    rec.save("session.json")

    # Later
    replay("session.json")
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import cognicore
from cognicore.agents.base_agent import RandomAgent
import logging

logger = logging.getLogger("cognicore.replay")


class SessionRecorder:
    """Record full episodes step-by-step for replay."""

    def __init__(
        self, env_id: str = "SafetyClassification-v1", difficulty: str = "easy"
    ):
        self.env_id = env_id
        self.difficulty = difficulty
        self.recordings: List[Dict] = []

    def record(self, agent=None, episodes: int = 1) -> List[Dict]:
        """Record episodes."""
        for ep in range(episodes):
            env = cognicore.make(self.env_id, difficulty=self.difficulty)
            if agent is None:
                _agent = RandomAgent(env.action_space)
            else:
                _agent = agent

            obs = env.reset()
            steps = []
            step_num = 0

            while True:
                step_num += 1
                t0 = time.time()
                action = _agent.act(obs)
                latency = (time.time() - t0) * 1000

                obs_before = {
                    k: str(v)[:200] for k, v in obs.items() if not k.startswith("_")
                }
                obs, reward, done, _, info = env.step(action)
                er = info.get("eval_result", {})

                if hasattr(_agent, "learn"):
                    _agent.learn(reward, info)

                steps.append(
                    {
                        "step": step_num,
                        "observation": obs_before,
                        "action": {k: str(v) for k, v in action.items()},
                        "category": er.get("category", "?"),
                        "correct": er.get("correct", False),
                        "predicted": str(er.get("predicted", "")),
                        "ground_truth": str(er.get("ground_truth", "")),
                        "reward_total": reward.total,
                        "reward_base": reward.base_score,
                        "memory_bonus": reward.memory_bonus,
                        "streak_penalty": reward.streak_penalty,
                        "novelty_bonus": reward.novelty_bonus,
                        "latency_ms": round(latency, 2),
                        "timestamp": time.time(),
                    }
                )

                if done:
                    break

            stats = env.episode_stats()
            recording = {
                "episode": ep + 1,
                "env_id": self.env_id,
                "difficulty": self.difficulty,
                "agent": getattr(_agent, "name", type(_agent).__name__),
                "accuracy": stats.accuracy,
                "score": env.get_score(),
                "total_steps": step_num,
                "correct_count": stats.correct_count,
                "steps": steps,
                "recorded_at": time.time(),
            }
            self.recordings.append(recording)

        return self.recordings

    def save(self, path: str = "session.json"):
        """Save recordings to JSON."""
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "_cognicore_session": True,
                    "_version": "1.0",
                    "recordings": self.recordings,
                },
                f,
                indent=2,
                default=str,
            )
        return path

    @staticmethod
    def load(path: str) -> List[Dict]:
        """Load recordings from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("recordings", [])


def replay(path: str, speed: float = 1.0, verbose: bool = True):
    """Replay a recorded session."""
    recordings = SessionRecorder.load(path)

    for rec in recordings:
        if verbose:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"  Replay: {rec['env_id']} ({rec['difficulty']})")
            logger.info(f"  Agent: {rec['agent']}")
            logger.info(f"  Episode {rec['episode']} | {rec['total_steps']} steps")
            logger.info(f"{'=' * 60}")

        for step in rec["steps"]:
            if verbose:
                icon = " OK" if step["correct"] else "ERR"
                print(
                    f"  {step['step']:3d} [{icon}] {step['category']:20s} "
                    f"predicted={step['predicted']:15s} "
                    f"truth={step['ground_truth']:15s} "
                    f"reward={step['reward_total']:+.2f}"
                )

        if verbose:
            print(
                f"\n  Result: {rec['accuracy']:.0%} accuracy, score={rec['score']:.4f}"
            )
            logger.info(f"{'=' * 60}")

    return recordings
