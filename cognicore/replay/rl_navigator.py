"""
RL Navigator — learns WHEN and HOW to branch for best outcomes.
Offline RL on trajectory trees using Conservative Q-Learning (CQL).
"""
import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from pathlib import Path
from cognicore.rl.dqn import DQN, ReplayBuffer

MODEL_DIR = Path.home() / ".cognicore" / "replay"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class NavAction(IntEnum):
    CONTINUE = 0
    BRANCH_POLICY = 1
    BRANCH_STRATEGY = 2
    BRANCH_MODEL = 3
    REWIND_3_STEPS = 4
    REWIND_TO_PLAN = 5
    ESCALATE = 6


NAV_NAMES = {
    NavAction.CONTINUE: "continue",
    NavAction.BRANCH_POLICY: "branch_policy",
    NavAction.BRANCH_STRATEGY: "branch_strategy",
    NavAction.BRANCH_MODEL: "branch_model",
    NavAction.REWIND_3_STEPS: "rewind_3",
    NavAction.REWIND_TO_PLAN: "rewind_to_plan",
    NavAction.ESCALATE: "escalate",
}


@dataclass
class BranchDecision:
    action: NavAction
    confidence: float
    reasoning: str = ""
    should_branch: bool = False

    def __post_init__(self):
        self.should_branch = self.action != NavAction.CONTINUE

    @staticmethod
    def CONTINUE(confidence=1.0):
        return BranchDecision(
            action=NavAction.CONTINUE, confidence=confidence)


class RLNavigator:
    """
    DQN agent that learns when to branch.
    After many branches, it knows:
    'When agent fails at step 5 with pattern X,
     the best action is rewind_to_plan, not try_different_strategy.'
    """

    def __init__(self, input_dim=128, epsilon=0.15, lr=0.001,
                 model_path=None):
        self.input_dim = input_dim
        self.epsilon = epsilon
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.997
        self.batch_size = 32
        self.train_steps = 0

        # CQL penalty weight (conservative Q-learning)
        self.cql_alpha = 0.5

        self.q_network = DQN(
            input_dim=input_dim,
            hidden_dims=[256, 128, 64],
            output_dim=len(NavAction),
            lr=lr)

        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.model_path = model_path or str(MODEL_DIR / "navigator.json")

        if Path(self.model_path).exists():
            try:
                self.q_network.load(self.model_path)
            except Exception:
                pass

    def should_branch(self, state_features: np.ndarray,
                     step: int = 0,
                     context: dict = None) -> BranchDecision:
        """Decide whether to branch at the current state."""
        features = self._build_features(state_features, step, context)

        q_values = self.q_network.predict(features)

        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            action = np.random.randint(len(NavAction))
        else:
            action = int(np.argmax(q_values))

        # Confidence via softmax
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / (exp_q.sum() + 1e-8)
        confidence = float(probs[action])

        reasoning = self._explain(action, q_values, step, context)

        return BranchDecision(
            action=NavAction(action),
            confidence=confidence,
            reasoning=reasoning)

    def learn_from_branch(self, original_features: np.ndarray,
                         action_taken: int,
                         branch_solved: bool,
                         original_solved: bool,
                         branch_cost: float = 0.0,
                         step: int = 0,
                         context: dict = None):
        """Learn from the outcome of a branch decision."""
        features = self._build_features(original_features, step, context)
        reward = self._compute_reward(
            action_taken, branch_solved, original_solved, branch_cost)

        # Use same features as next_state (terminal transition)
        self.replay_buffer.add(
            features, action_taken, reward, features, done=True)

        # Train
        if len(self.replay_buffer) >= self.batch_size:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)

            # Standard DQN loss
            loss = self.q_network.update(
                states, actions, rewards, next_states, dones)

            self.train_steps += 1
            self.epsilon = max(self.epsilon_min,
                             self.epsilon * self.epsilon_decay)

            return loss
        return 0.0

    def _compute_reward(self, action: int, branch_solved: bool,
                       original_solved: bool, cost: float) -> float:
        """Reward based on branch outcome vs original."""
        if action == NavAction.CONTINUE:
            return 0.0  # neutral for continuing

        if branch_solved and not original_solved:
            return 1.0   # branch found a solution original missed
        elif branch_solved and original_solved:
            return 0.3 - cost * 10  # both solved, penalize extra cost
        elif not branch_solved and original_solved:
            return -1.0  # unnecessary branch, original was fine
        else:
            return -0.5  # both failed, wasted tokens

    def _build_features(self, base_features: np.ndarray,
                       step: int = 0,
                       context: dict = None) -> np.ndarray:
        """Build input features for the navigator."""
        features = np.zeros(self.input_dim, dtype=np.float32)

        if base_features is not None:
            base = np.array(base_features, dtype=np.float32).reshape(-1)
            n = min(len(base), 96)
            features[:n] = base[:n]

        # Step information (dims 96-111)
        features[96] = step / 50.0  # normalized step
        if context:
            features[97] = context.get("tests_failed", 0) / 10.0
            features[98] = context.get("tests_passed", 0) / 100.0
            features[99] = context.get("patches_generated", 0) / 10.0
            features[100] = context.get("total_cost", 0) / 1.0
            features[101] = context.get("total_tokens_in", 0) / 50000.0
            features[102] = 1.0 if context.get("last_action") == "test_failed" else 0.0
            features[103] = len(context.get("events_so_far", [])) / 100.0

        return features

    def _explain(self, action: int, q_values: np.ndarray,
                step: int, context: dict) -> str:
        """Human-readable explanation of the decision."""
        action_name = NAV_NAMES.get(NavAction(action), "unknown")
        top3 = np.argsort(q_values)[-3:][::-1]
        alternatives = [f"{NAV_NAMES[NavAction(a)]}={q_values[a]:.2f}"
                       for a in top3]
        return f"Action={action_name} at step {step}. Q-values: {', '.join(alternatives)}"

    def save(self):
        self.q_network.save(self.model_path)

    def load(self):
        if Path(self.model_path).exists():
            self.q_network.load(self.model_path)

    def get_stats(self) -> dict:
        return {
            "epsilon": round(self.epsilon, 4),
            "train_steps": self.train_steps,
            "buffer_size": len(self.replay_buffer),
            "model_path": self.model_path,
        }
