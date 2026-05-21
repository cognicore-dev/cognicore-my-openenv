"""
RL Defender — DQN agent that learns defense policies.
Uses the shared DQN from cognicore.rl.dqn.
"""
import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from pathlib import Path
from cognicore.rl.dqn import DQN, ReplayBuffer


class DefenseAction(IntEnum):
    ALLOW = 0
    BLOCK = 1
    QUARANTINE = 2
    SANITIZE = 3
    RATE_LIMIT = 4
    ALERT_HUMAN = 5


ACTION_NAMES = {
    DefenseAction.ALLOW: "allow",
    DefenseAction.BLOCK: "block",
    DefenseAction.QUARANTINE: "quarantine",
    DefenseAction.SANITIZE: "sanitize",
    DefenseAction.RATE_LIMIT: "rate_limit",
    DefenseAction.ALERT_HUMAN: "alert_human",
}

# Reward structure
REWARDS = {
    "correct_block": 1.0,
    "correct_allow": 0.5,
    "false_positive": -1.0,   # blocked safe input
    "false_negative": -2.0,   # allowed real threat
    "unnecessary_quarantine": -0.1,
    "good_sanitize": 0.7,
    "good_rate_limit": 0.6,
    "good_escalate": 0.3,
}

MODEL_DIR = Path.home() / ".cognicore" / "immune"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DefenseDecision:
    action: DefenseAction
    confidence: float
    q_values: np.ndarray = None
    action_name: str = ""

    def __post_init__(self):
        self.action_name = ACTION_NAMES.get(self.action, "unknown")


class RLDefender:
    """DQN-based defense agent. Learns which action to take for each threat."""

    def __init__(self, input_dim=128, epsilon=0.1, lr=0.001,
                 model_path=None):
        self.input_dim = input_dim
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.train_step_count = 0
        self.batch_size = 32

        self.q_network = DQN(
            input_dim=input_dim,
            hidden_dims=[256, 128, 64],
            output_dim=len(DefenseAction),
            lr=lr)

        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.model_path = model_path or str(MODEL_DIR / "defender.json")

        # Try to load existing model
        if Path(self.model_path).exists():
            try:
                self.q_network.load(self.model_path)
            except Exception:
                pass

    def decide(self, features: np.ndarray) -> DefenseDecision:
        """Epsilon-greedy action selection."""
        features = np.array(features, dtype=np.float32).reshape(-1)
        if len(features) != self.input_dim:
            features = np.resize(features, self.input_dim)

        q_values = self.q_network.predict(features)

        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            action = np.random.randint(len(DefenseAction))
        else:
            action = int(np.argmax(q_values))

        # Confidence via softmax
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / (exp_q.sum() + 1e-8)
        confidence = float(probs[action])

        return DefenseDecision(
            action=DefenseAction(action),
            confidence=confidence,
            q_values=q_values)

    def compute_reward(self, action: DefenseAction, was_threat: bool,
                      threat_score: float = 0.0) -> float:
        """Compute reward for a defense decision."""
        if was_threat:
            if action == DefenseAction.BLOCK:
                return REWARDS["correct_block"]
            elif action == DefenseAction.QUARANTINE:
                return 0.5  # partial credit
            elif action == DefenseAction.SANITIZE:
                return REWARDS["good_sanitize"] * threat_score
            elif action == DefenseAction.ALLOW:
                return REWARDS["false_negative"]
            elif action == DefenseAction.RATE_LIMIT:
                return REWARDS["good_rate_limit"]
            elif action == DefenseAction.ALERT_HUMAN:
                return REWARDS["good_escalate"]
        else:
            if action == DefenseAction.ALLOW:
                return REWARDS["correct_allow"]
            elif action == DefenseAction.BLOCK:
                return REWARDS["false_positive"]
            elif action == DefenseAction.QUARANTINE:
                return REWARDS["unnecessary_quarantine"]
            elif action == DefenseAction.SANITIZE:
                return -0.2  # unnecessary sanitization
            else:
                return -0.1
        return 0.0

    def update(self, features, action, reward, next_features, done=False):
        """Store transition and train if enough data."""
        features = np.array(features, dtype=np.float32).reshape(-1)
        next_features = np.array(next_features, dtype=np.float32).reshape(-1)

        if len(features) != self.input_dim:
            features = np.resize(features, self.input_dim)
        if len(next_features) != self.input_dim:
            next_features = np.resize(next_features, self.input_dim)

        self.replay_buffer.add(
            features, int(action), float(reward), next_features, done)

        loss = 0.0
        if len(self.replay_buffer) >= self.batch_size:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            loss = self.q_network.update(
                states, actions, rewards, next_states, dones)
            self.train_step_count += 1

            # Decay epsilon
            self.epsilon = max(self.epsilon_min,
                             self.epsilon * self.epsilon_decay)

        return loss

    def save(self):
        """Persist model to disk."""
        self.q_network.save(self.model_path)

    def load(self):
        """Load model from disk."""
        if Path(self.model_path).exists():
            self.q_network.load(self.model_path)

    def get_stats(self) -> dict:
        return {
            "epsilon": round(self.epsilon, 4),
            "train_steps": self.train_step_count,
            "buffer_size": len(self.replay_buffer),
            "model_path": self.model_path,
        }
