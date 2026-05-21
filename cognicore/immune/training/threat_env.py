"""
Threat Environment — Gymnasium-compatible RL environment for training the defender.
Episodes are sequences of safe + malicious inputs. Curriculum levels 1-5.
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any

from cognicore.rl.dqn import FeatureExtractor
from cognicore.immune.training.threat_data import (
    get_all_samples, get_by_difficulty, get_training_split)
from cognicore.immune.rl_defender import DefenseAction


@dataclass
class EnvConfig:
    episode_length: int = 50
    difficulty: int = 1       # 1-5 curriculum level
    threat_ratio: float = 0.3  # fraction of threats per episode


class ThreatEnvironment:
    """
    RL environment for training the immune defender.

    Observation: 128-dim feature vector
    Action:      6 defense actions (int 0-5)
    Reward:      based on correct/incorrect decisions

    Curriculum levels:
      1: obvious threats only
      2: subtle threats mixed in
      3: adversarial inputs
      4: novel patterns
      5: adaptive adversary
    """

    def __init__(self, config: EnvConfig = None):
        self.config = config or EnvConfig()
        self.extractor = FeatureExtractor()
        self.observation_space_dim = 128
        self.action_space_n = len(DefenseAction)

        # Load data by difficulty
        self._all_samples = get_by_difficulty(self.config.difficulty)
        self._safe = [s for s in self._all_samples if not s["is_threat"]]
        self._threats = [s for s in self._all_samples if s["is_threat"]]

        # Episode state
        self._episode_samples = []
        self._step = 0
        self._current_sample = None
        self._done = False

        # Metrics
        self.episode_rewards = []
        self.total_episodes = 0
        self.total_steps = 0
        self.correct_decisions = 0
        self.total_decisions = 0

    def reset(self) -> np.ndarray:
        """Reset environment for new episode. Returns first observation."""
        self._step = 0
        self._done = False

        # Build episode: mix of safe and threat inputs
        n = self.config.episode_length
        n_threats = int(n * self.config.threat_ratio)
        n_safe = n - n_threats

        safe_picks = [self._safe[i % len(self._safe)]
                     for i in np.random.permutation(len(self._safe))[:n_safe]]
        threat_picks = [self._threats[i % len(self._threats)]
                       for i in np.random.permutation(len(self._threats))[:n_threats]]

        self._episode_samples = safe_picks + threat_picks
        np.random.shuffle(self._episode_samples)

        self._current_sample = self._episode_samples[0]
        obs = self.extractor.extract(self._current_sample["text"])

        self.total_episodes += 1
        self._episode_reward = 0.0

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take action on current input.
        Returns: (next_obs, reward, done, info)
        """
        if self._done:
            return np.zeros(128), 0.0, True, {}

        sample = self._current_sample
        is_threat = sample["is_threat"]
        category = sample["category"]

        # Compute reward
        reward = self._compute_reward(action, is_threat)
        self._episode_reward += reward

        # Track accuracy
        self.total_decisions += 1
        self.total_steps += 1
        correct = False
        if is_threat and action in (1, 2, 3, 4, 5):  # any non-allow
            correct = True
        elif not is_threat and action == 0:  # allow safe
            correct = True
        if correct:
            self.correct_decisions += 1

        # Advance
        self._step += 1
        if self._step >= len(self._episode_samples):
            self._done = True
            self.episode_rewards.append(self._episode_reward)
            return np.zeros(128), reward, True, {
                "episode_reward": self._episode_reward,
                "accuracy": self.correct_decisions / max(self.total_decisions, 1),
            }

        self._current_sample = self._episode_samples[self._step]
        next_obs = self.extractor.extract(self._current_sample["text"])

        info = {
            "was_threat": is_threat,
            "category": category,
            "correct": correct,
            "step": self._step,
        }

        return next_obs, reward, False, info

    def _compute_reward(self, action: int, is_threat: bool) -> float:
        """Reward function matching the RL defender spec."""
        if is_threat:
            if action == DefenseAction.BLOCK:
                return 1.0
            elif action == DefenseAction.QUARANTINE:
                return 0.5
            elif action == DefenseAction.SANITIZE:
                return 0.4
            elif action == DefenseAction.RATE_LIMIT:
                return 0.3
            elif action == DefenseAction.ALERT_HUMAN:
                return 0.3
            elif action == DefenseAction.ALLOW:
                return -2.0  # false negative
        else:
            if action == DefenseAction.ALLOW:
                return 0.5
            elif action == DefenseAction.BLOCK:
                return -1.0  # false positive
            elif action == DefenseAction.QUARANTINE:
                return -0.1
            elif action == DefenseAction.SANITIZE:
                return -0.2
            else:
                return -0.1
        return 0.0

    def set_difficulty(self, level: int):
        """Update curriculum difficulty (1-5)."""
        self.config.difficulty = min(5, max(1, level))
        self._all_samples = get_by_difficulty(self.config.difficulty)
        self._safe = [s for s in self._all_samples if not s["is_threat"]]
        self._threats = [s for s in self._all_samples if s["is_threat"]]

    def get_metrics(self) -> dict:
        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "accuracy": self.correct_decisions / max(self.total_decisions, 1),
            "avg_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            "difficulty": self.config.difficulty,
        }


def train_defender(defender, episodes=100, difficulty=1):
    """Convenience function to train a defender on the threat env."""
    env = ThreatEnvironment(EnvConfig(
        episode_length=30, difficulty=difficulty, threat_ratio=0.4))

    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            decision = defender.decide(obs)
            next_obs, reward, done, info = env.step(int(decision.action))
            defender.update(obs, int(decision.action), reward, next_obs, done)
            obs = next_obs

        # Curriculum: increase difficulty as accuracy improves
        metrics = env.get_metrics()
        if metrics["accuracy"] > 0.85 and env.config.difficulty < 5:
            env.set_difficulty(env.config.difficulty + 1)

    defender.save()
    return env.get_metrics()
