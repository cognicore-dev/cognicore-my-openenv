"""
NEXUS RL Policy Learner — learns orchestration policies from trajectory data.
Uses offline RL (Conservative Q-Learning style) on 220+ stored trajectories.
"""
import json, math, random, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from collections import defaultdict
from cognicore.nexus.trajectory_store import TrajectoryStore


class PolicyFeatures:
    """Extract features from task state for policy decisions."""

    @staticmethod
    def extract(task_desc, category, attempt, prev_failures):
        feats = {
            "len_short": len(task_desc) < 100,
            "len_medium": 100 <= len(task_desc) < 300,
            "len_long": len(task_desc) >= 300,
            "attempt_1": attempt == 1,
            "attempt_2": attempt == 2,
            "attempt_3plus": attempt >= 3,
            "has_failures": len(prev_failures) > 0,
            "many_failures": len(prev_failures) >= 3,
        }
        # Category features
        cats = ["arithmetic", "encoding", "off_by_one", "none_handling",
                "parsing", "error_handling", "type_conversion", "validation",
                "concurrency", "caching", "serialization", "boundary",
                "state_mgmt", "string_ops", "collection_ops", "io_handling"]
        for c in cats:
            feats[f"cat_{c}"] = (category == c)
        return feats

    @staticmethod
    def to_vector(feats):
        keys = sorted(feats.keys())
        return tuple(int(feats[k]) for k in keys)


class QLearningPolicyAgent:
    """Offline Q-Learning agent that selects routing policies."""

    POLICIES = ["minimal", "standard", "test_first", "review_first"]

    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: {p: 0.0 for p in self.POLICIES})
        self.training_steps = 0

    def select_policy(self, state_vec, explore=False):
        if explore and random.random() < self.epsilon:
            return random.choice(self.POLICIES)
        q_vals = self.q_table[state_vec]
        return max(q_vals, key=q_vals.get)

    def update(self, state_vec, action, reward, next_state_vec=None):
        old_q = self.q_table[state_vec][action]
        if next_state_vec is not None:
            max_next = max(self.q_table[next_state_vec].values())
        else:
            max_next = 0.0
        self.q_table[state_vec][action] = old_q + self.alpha * (
            reward + self.gamma * max_next - old_q)
        self.training_steps += 1

    def train_from_trajectories(self, trajectories, epochs=10):
        """Train on trajectory data (offline RL)."""
        for epoch in range(epochs):
            random.shuffle(trajectories)
            total_reward = 0
            for traj in trajectories:
                feats = PolicyFeatures.extract(
                    traj.get("task_description", ""),
                    traj.get("category", "unknown"),
                    1,  # first attempt
                    []
                )
                state = PolicyFeatures.to_vector(feats)
                policy = traj.get("policy", "standard")
                if policy not in self.POLICIES:
                    continue
                reward = traj.get("total_reward", 0)
                self.update(state, policy, reward)
                total_reward += reward

            if (epoch + 1) % 5 == 0:
                avg = total_reward / max(len(trajectories), 1)
                print(f"  Epoch {epoch+1}/{epochs}: avg_reward={avg:+.3f}, "
                      f"states={len(self.q_table)}, steps={self.training_steps}")

    def get_learned_policy_map(self):
        """Return the best policy for each known state."""
        policy_map = {}
        for state, q_vals in self.q_table.items():
            best = max(q_vals, key=q_vals.get)
            policy_map[state] = {
                "best_policy": best,
                "q_values": dict(q_vals),
                "confidence": q_vals[best] - min(q_vals.values())
            }
        return policy_map

    def summary(self):
        policy_counts = defaultdict(int)
        for state, q_vals in self.q_table.items():
            best = max(q_vals, key=q_vals.get)
            policy_counts[best] += 1
        return dict(policy_counts)

    def save(self, path):
        data = {
            "q_table": {str(k): v for k, v in self.q_table.items()},
            "training_steps": self.training_steps,
            "alpha": self.alpha, "gamma": self.gamma, "epsilon": self.epsilon
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.training_steps = data["training_steps"]
        for k, v in data["q_table"].items():
            self.q_table[eval(k)] = v


def train_policy():
    """Train RL policy from stored trajectories."""
    print("=" * 60)
    print("  NEXUS RL Policy Training")
    print("=" * 60)

    ts = TrajectoryStore()
    stats = ts.get_stats()
    print(f"\n  Trajectory store: {stats['total_trajectories']} trajectories")
    print(f"  Solved: {stats['solved']}")
    print(f"  Policies: {stats['policies']}")

    # Export and load
    export_path = ts.export_for_training()
    with open(export_path) as f:
        trajectories = [json.loads(line) for line in f]

    print(f"  Loaded {len(trajectories)} trajectories for training\n")

    # Train
    agent = QLearningPolicyAgent(alpha=0.1, gamma=0.95, epsilon=0.05)
    agent.train_from_trajectories(trajectories, epochs=50)

    # Results
    print(f"\n  Training complete: {agent.training_steps} updates")
    print(f"  Unique states: {len(agent.q_table)}")
    print(f"\n  Learned policy distribution:")
    for policy, count in sorted(agent.summary().items(), key=lambda x: -x[1]):
        pct = 100 * count / max(len(agent.q_table), 1)
        print(f"    {policy:<15} {count:>3} states ({pct:.0f}%)")

    # Save model
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'nexus_policy_model.json')
    agent.save(model_path)
    print(f"\n  Model saved: {model_path}")

    # Show Q-values for sample states
    print(f"\n  Sample Q-values:")
    for i, (state, q_vals) in enumerate(list(agent.q_table.items())[:5]):
        best = max(q_vals, key=q_vals.get)
        print(f"    State {i}: best={best} Q={q_vals[best]:+.3f}")

    print("=" * 60)
    return agent


if __name__ == "__main__":
    train_policy()
