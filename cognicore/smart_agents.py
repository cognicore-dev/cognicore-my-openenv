"""
CogniCore Smart Agents — Prebuilt agents that use CogniCore's cognitive features.

Developers don't need to build from scratch — just use these:
  - AutoLearner: uses memory + reflection to improve each step
  - SafeAgent: conservative, flags uncertain cases
  - AdaptiveAgent: adjusts strategy based on performance

Usage::

    from cognicore.smart_agents import AutoLearner

    agent = AutoLearner()
    env = cognicore.make("SafetyClassification-v1")
    obs = env.reset()
    while True:
        action = agent.act(obs)
        obs, reward, done, _, info = env.step(action)
        agent.learn(reward, info)
        if done: break
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional


class AutoLearner:
    """Agent that actually learns from CogniCore's memory and reflection.

    Maintains a knowledge base of category→action mappings based on
    past rewards. Uses reflection hints to adjust strategy.
    """

    def __init__(self, actions: Optional[List[str]] = None):
        self.name = "AutoLearner"
        self.knowledge: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.actions = actions or ["SAFE", "UNSAFE", "NEEDS_REVIEW"]
        self.history: List[Dict] = []
        self.epsilon = 0.15  # exploration rate

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Choose action using knowledge + memory + reflection."""
        category = obs.get("category", "unknown")

        # 1. Check reflection hints
        hints = str(obs.get("reflection_hints", "")).lower()
        if "unsafe" in hints and "safe" not in hints[:4]:
            return {"classification": "UNSAFE"}
        if "needs_review" in hints:
            return {"classification": "NEEDS_REVIEW"}

        # 2. Check memory context for direct matches
        for mem in obs.get("memory_context", []):
            if mem.get("category") == category and mem.get("correct"):
                return {"classification": mem.get("predicted", self.actions[0])}

        # 3. Use knowledge base
        if category in self.knowledge and self.knowledge[category]:
            if random.random() > self.epsilon:
                best = max(self.knowledge[category], key=self.knowledge[category].get)
                return {"classification": best}

        # 4. Heuristic: check prompt keywords
        prompt = obs.get("prompt", "").lower()
        unsafe_signals = [
            "malware",
            "hack",
            "exploit",
            "phish",
            "weapon",
            "bomb",
            "attack",
            "kill",
            "drug",
            "password",
            "credential",
        ]
        if sum(1 for w in unsafe_signals if w in prompt) >= 1:
            return {"classification": "UNSAFE"}

        # 5. Default
        return {"classification": random.choice(self.actions)}

    def learn(self, reward, info: Optional[Dict[str, Any]] = None):
        """Update knowledge from reward and eval result."""
        if info is None: info = {}
        er = info.get("eval_result", {})
        category = er.get("category", "unknown")
        action = er.get("predicted", "")
        correct = er.get("correct", False)

        # Update knowledge: increase weight for correct, decrease for wrong
        if correct:
            self.knowledge[category][action] += reward.total + 0.5
        else:
            self.knowledge[category][action] -= 0.3
            # Learn the correct answer
            truth = er.get("ground_truth", "")
            if truth:
                self.knowledge[category][truth] += 0.8

        self.history.append(
            {
                "category": category,
                "action": action,
                "correct": correct,
                "reward": reward.total,
            }
        )

        # Decay epsilon
        self.epsilon = max(0.02, self.epsilon * 0.98)


class SafeAgent:
    """Conservative agent that flags uncertain cases.

    Prioritizes safety — when unsure, says NEEDS_REVIEW or UNSAFE.
    Good for production safety systems where false negatives are costly.
    """

    def __init__(self, conservative_threshold: float = 0.3):
        self.name = "SafeAgent"
        self.threshold = conservative_threshold
        self.knowledge: Dict[str, str] = {}

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        category = obs.get("category", "unknown")
        prompt = obs.get("prompt", "").lower()

        # Strong unsafe signals
        strong_unsafe = [
            "malware",
            "hack",
            "exploit",
            "weapon",
            "bomb",
            "attack",
            "kill",
            "drug",
            "illegal",
        ]
        if any(w in prompt for w in strong_unsafe):
            return {"classification": "UNSAFE"}

        # Known categories from memory
        for mem in obs.get("memory_context", []):
            if mem.get("category") == category and mem.get("correct"):
                return {"classification": mem.get("predicted", "NEEDS_REVIEW")}

        # Known from past learning
        if category in self.knowledge:
            return {"classification": self.knowledge[category]}

        # When uncertain: flag for review (conservative)
        return {"classification": "NEEDS_REVIEW"}

    def learn(self, reward, info: Optional[Dict[str, Any]] = None):
        if info is None: info = {}
        er = info.get("eval_result", {})
        if er.get("correct"):
            self.knowledge[er.get("category", "?")] = er.get("predicted", "SAFE")


class AdaptiveAgent:
    """Agent that switches strategy based on its performance.

    - Performing well → exploit known patterns
    - Performing poorly → explore more aggressively
    - Memory-rich → rely on memory
    - Memory-poor → use heuristics
    """

    def __init__(self, actions: Optional[List[str]] = None):
        self.name = "AdaptiveAgent"
        self.actions = actions or ["SAFE", "UNSAFE", "NEEDS_REVIEW"]
        self.knowledge: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._correct = 0
        self._total = 0

    @property
    def accuracy(self) -> float:
        return self._correct / self._total if self._total > 0 else 0.5

    @property
    def strategy(self) -> str:
        if self._total < 3:
            return "exploring"
        if self.accuracy > 0.7:
            return "exploiting"
        if self.accuracy < 0.3:
            return "aggressive_explore"
        return "balanced"

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        category = obs.get("category", "unknown")
        prompt = obs.get("prompt", "").lower()
        strat = self.strategy

        if strat == "exploiting":
            # Trust knowledge base
            if category in self.knowledge:
                best = max(self.knowledge[category], key=self.knowledge[category].get)
                return {"classification": best}

        if strat == "aggressive_explore":
            # Try random to find what works
            return {"classification": random.choice(self.actions)}

        # Balanced / exploring: use hints + heuristics
        hints = str(obs.get("reflection_hints", "")).lower()
        if "unsafe" in hints:
            return {"classification": "UNSAFE"}

        unsafe_words = ["malware", "hack", "weapon", "phish", "exploit"]
        if any(w in prompt for w in unsafe_words):
            return {"classification": "UNSAFE"}

        if category in self.knowledge:
            best = max(self.knowledge[category], key=self.knowledge[category].get)
            if random.random() > 0.2:
                return {"classification": best}

        return {"classification": random.choice(self.actions)}

    def learn(self, reward, info: Optional[Dict[str, Any]] = None):
        if info is None: info = {}
        er = info.get("eval_result", {})
        category = er.get("category", "unknown")
        predicted = er.get("predicted", "")
        correct = er.get("correct", False)

        self._total += 1
        if correct:
            self._correct += 1
            self.knowledge[category][predicted] += reward.total + 0.5
        else:
            self.knowledge[category][predicted] -= 0.3
            truth = er.get("ground_truth", "")
            if truth:
                self.knowledge[category][truth] += 0.8
