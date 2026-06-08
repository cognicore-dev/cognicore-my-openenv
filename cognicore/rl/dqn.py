"""
DQN (Deep Q-Network) — numpy-only implementation.
Used by both the Immune Defender and the Replay Navigator.
"""
import numpy as np
import json, os
from pathlib import Path


class DQN:
    """Pure-numpy DQN with experience replay and target network."""

    def __init__(self, input_dim, hidden_dims, output_dim, lr=0.001, gamma=0.99):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.gamma = gamma
        self.layers = []
        self.target_layers = []

        # Build network weights
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self.layers.append((w.copy(), b.copy()))
            self.target_layers.append((w.copy(), b.copy()))

        self._update_count = 0
        self._target_update_freq = 100

    def forward(self, x, use_target=False):
        """Forward pass through the network."""
        layers = self.target_layers if use_target else self.layers
        h = np.array(x, dtype=np.float32)
        activations = [h]
        for i, (w, b) in enumerate(layers):
            h = h @ w + b
            if i < len(layers) - 1:
                h = np.maximum(0, h)  # ReLU
            activations.append(h)
        return h, activations

    def predict(self, state):
        """Get Q-values for a state."""
        q_values, _ = self.forward(state.reshape(1, -1))
        return q_values[0]

    def predict_batch(self, states):
        """Get Q-values for a batch of states."""
        q_values, _ = self.forward(states)
        return q_values

    def update(self, states, actions, rewards, next_states, dones):
        """Single gradient update step (batch)."""
        batch_size = len(states)

        # Current Q-values
        q_current, activations = self.forward(states)

        # Target Q-values (from target network)
        q_next, _ = self.forward(next_states, use_target=True)
        targets = q_current.copy()

        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])

        # Backprop through layers
        loss = np.mean((q_current - targets) ** 2)
        grad = 2.0 * (q_current - targets) / batch_size

        # Backward pass
        for i in range(len(self.layers) - 1, -1, -1):
            w, b = self.layers[i]
            h_in = activations[i]

            dw = h_in.T @ grad
            db = grad.sum(axis=0)

            if i > 0:
                grad = grad @ w.T
                # ReLU backward
                grad = grad * (activations[i] > 0)

            # SGD update
            self.layers[i] = (w - self.lr * np.clip(dw, -1, 1),
                             b - self.lr * np.clip(db, -1, 1))

        # Periodically update target network
        self._update_count += 1
        if self._update_count % self._target_update_freq == 0:
            self.sync_target()

        return loss

    def sync_target(self):
        """Copy weights to target network."""
        self.target_layers = [(w.copy(), b.copy()) for w, b in self.layers]

    def save(self, path):
        """Save weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "lr": self.lr,
            "gamma": self.gamma,
            "layers": [(w.tolist(), b.tolist()) for w, b in self.layers],
        }
        path.write_text(json.dumps(data))

    def load(self, path):
        """Load weights from disk."""
        data = json.loads(Path(path).read_text())
        self.layers = [(np.array(w), np.array(b)) for w, b in data["layers"]]
        self.sync_target()


class ReplayBuffer:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        transition = (
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class FeatureExtractor:
    """128-dim feature vector from text input. Shared by immune + replay."""

    def extract(self, text: str) -> np.ndarray:
        features = np.zeros(128, dtype=np.float32)

        # === Lexical features (dims 0-31) ===
        tokens = text.split()
        features[0] = min(len(tokens) / 500.0, 1.0)  # token count (normalized)
        features[1] = np.mean([len(t) for t in tokens]) / 20.0 if tokens else 0
        features[2] = sum(1 for c in text if c in '!@#$%^&*()') / max(len(text), 1)
        features[3] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features[4] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        features[5] = self._entropy(text)
        features[6] = 1.0 if any(c > '\u007f' for c in text) else 0.0  # non-ASCII
        features[7] = min(len(text) / 10000.0, 1.0)  # raw length

        # N-gram features
        bigrams = [text[i:i+2] for i in range(len(text)-1)]
        features[8] = len(set(bigrams)) / max(len(bigrams), 1)  # bigram diversity

        # Word length distribution
        if tokens:
            lengths = [len(t) for t in tokens]
            features[9] = np.std(lengths) / 10.0 if len(lengths) > 1 else 0
            features[10] = max(lengths) / 50.0

        # === Threat signature features (dims 32-63) ===
        threat_keywords = [
            "ignore previous", "ignore all", "disregard",
            "forget your", "new instructions", "system prompt",
            "act as", "you are now", "pretend to be",
            "DAN", "jailbreak", "bypass", "override",
            "base64", "eval(", "exec(", "__import__",
            "as your developer", "I am the admin",
            "hypothetically", "in theory", "fictional",
            "my life depends", "emergency", "urgent override"
        ]
        text_lower = text.lower()
        for i, kw in enumerate(threat_keywords[:24]):
            features[32 + i] = 1.0 if kw.lower() in text_lower else 0.0

        # Instruction-like patterns
        features[56] = 1.0 if any(p in text_lower for p in
            ["you must", "you should", "you will", "do not"]) else 0.0
        features[57] = 1.0 if "```" in text else 0.0  # code blocks
        features[58] = text_lower.count("http") / 10.0  # URL count
        features[59] = 1.0 if self._has_base64(text) else 0.0

        # === Structural features (dims 64-95) ===
        lines = text.split("\n")
        features[64] = min(len(lines) / 100.0, 1.0)  # line count
        features[65] = 1.0 if any(l.startswith("#") for l in lines) else 0.0
        features[66] = text.count("{") / max(len(text), 1) * 100  # brace density
        features[67] = 1.0 if "```" in text else 0.0

        # Role-play indicators
        rp_patterns = ["act as", "you are", "pretend", "roleplay",
                      "character", "persona", "imagine you"]
        features[68] = sum(1 for p in rp_patterns if p in text_lower) / len(rp_patterns)

        # Negation density
        negations = ["not", "don't", "never", "no", "shouldn't", "can't"]
        features[69] = sum(text_lower.count(n) for n in negations) / max(len(tokens), 1)

        # Question marks, exclamation
        features[70] = text.count("?") / max(len(text), 1) * 100
        features[71] = text.count("!") / max(len(text), 1) * 100

        # Repeated characters (potential bombing)
        features[72] = max(len(list(g)) for _, g in
            __import__('itertools').groupby(text)) / max(len(text), 1) if text else 0

        # === Historical / contextual features (dims 96-127) ===
        # These are filled in by the caller with session context
        # Leaving as zeros for standalone extraction

        return features

    def _entropy(self, text: str) -> float:
        if not text:
            return 0.0
        freq = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        probs = np.array(list(freq.values())) / len(text)
        return float(-np.sum(probs * np.log2(probs + 1e-10)) / 8.0)

    def _has_base64(self, text: str) -> bool:
        import re
        return bool(re.search(r'[A-Za-z0-9+/]{20,}={0,2}', text))
