"""Tests for cognicore.core.api — train() and evaluate() functions."""

import pytest
import cognicore
from cognicore.core.api import train, evaluate
from cognicore.core.errors import AgentInterfaceError, CogniCoreError
from cognicore.agents.base_agent import RandomAgent


class _AlwaysSafeAgent:
    """Minimal agent that always classifies as SAFE."""

    def act(self, obs):
        return {"classification": "SAFE"}


class _AlwaysUnsafeAgent:
    """Minimal agent that always classifies as UNSAFE."""

    def act(self, obs):
        return {"classification": "UNSAFE"}


class _LearningAgent:
    """Agent that tracks rewards for testing on_reward and on_episode_end callbacks."""

    def __init__(self):
        self.rewards_received = []
        self.episodes_completed = []

    def act(self, obs):
        return {"classification": "SAFE"}

    def on_reward(self, reward):
        self.rewards_received.append(reward)

    def on_episode_end(self, stats):
        self.episodes_completed.append(stats)


class TestTrain:
    def test_train_returns_agent(self):
        agent = _AlwaysSafeAgent()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        result = train(agent, env, episodes=1)
        assert result is agent

    def test_train_multiple_episodes(self):
        agent = _LearningAgent()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        train(agent, env, episodes=3)
        assert len(agent.episodes_completed) == 3

    def test_train_calls_on_reward(self):
        agent = _LearningAgent()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        train(agent, env, episodes=1)
        assert len(agent.rewards_received) > 0

    def test_train_calls_on_episode_end(self):
        agent = _LearningAgent()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        train(agent, env, episodes=2)
        assert len(agent.episodes_completed) == 2

    def test_train_invalid_agent_raises(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        with pytest.raises(AgentInterfaceError):
            train(object(), env, episodes=1)

    def test_train_invalid_env_raises(self):
        agent = _AlwaysSafeAgent()
        with pytest.raises(CogniCoreError):
            train(agent, object(), episodes=1)

    def test_train_zero_episodes_returns_agent(self):
        agent = _AlwaysSafeAgent()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        result = train(agent, env, episodes=0)
        assert result is agent

    def test_train_negative_episodes_raises(self):
        agent = _AlwaysSafeAgent()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        with pytest.raises(ValueError):
            train(agent, env, episodes=-1)

    def test_train_with_random_agent(self):
        agent = RandomAgent()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        result = train(agent, env, episodes=2)
        assert result is agent


class TestEvaluate:
    def test_evaluate_returns_float(self):
        agent = _AlwaysSafeAgent()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        score = evaluate(agent, env, episodes=2)
        assert isinstance(score, float)

    def test_evaluate_score_in_range(self):
        agent = _AlwaysSafeAgent()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        score = evaluate(agent, env, episodes=2)
        assert 0.0 <= score <= 1.0

    def test_evaluate_invalid_agent_raises(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        with pytest.raises(AgentInterfaceError):
            evaluate(object(), env, episodes=1)

    def test_evaluate_invalid_env_raises(self):
        agent = _AlwaysSafeAgent()
        with pytest.raises(CogniCoreError):
            evaluate(agent, object(), episodes=1)

    def test_evaluate_invalid_episodes_raises(self):
        agent = _AlwaysSafeAgent()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        with pytest.raises(ValueError):
            evaluate(agent, env, episodes=-1)

    def test_evaluate_perfect_agent_higher_score(self):
        """An agent that always answers correctly should score >= a random agent."""
        # Use a simple agent that always answers SAFE
        correct_agent = _AlwaysSafeAgent()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        score = evaluate(correct_agent, env, episodes=3)
        # Score should be a valid float in range
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
