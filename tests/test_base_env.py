"""Tests for CogniCoreEnv base class with a minimal concrete subclass."""

import pytest

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import CogniCoreConfig, EvalResult
from cognicore.core.spaces import DiscreteSpace


class MathEnv(CogniCoreEnv):
    """Minimal test environment — answer math questions."""

    def _setup(self, **kwargs):
        self.action_space = DiscreteSpace(10)
        self.problems = kwargs.get(
            "problems",
            [
                {"q": "2+2", "a": 4, "category": "addition"},
                {"q": "3*3", "a": 9, "category": "multiplication"},
                {"q": "10-7", "a": 3, "category": "subtraction"},
            ],
        )

    def _generate_tasks(self):
        return self.problems

    def _evaluate(self, action):
        task = self._tasks[self._current_step]
        answer = action.get("answer")
        correct = answer == task["a"]
        return EvalResult(
            base_score=1.0 if correct else 0.0,
            correct=correct,
            ground_truth=task["a"],
            predicted=answer,
            category=task["category"],
        )

    def _get_obs(self):
        task = self._tasks[self._current_step]
        return {"question": task["q"], "category": task["category"]}


class TestBaseEnvLifecycle:
    """Test the reset/step/done lifecycle."""

    def test_reset(self):
        env = MathEnv()
        obs = env.reset()
        assert obs["question"] == "2+2"
        assert obs["step"] == 0
        assert obs["max_steps"] == 3

    def test_step_correct(self):
        env = MathEnv()
        env.reset()
        obs, reward, done, truncated, info = env.step({"answer": 4})

        assert reward.base_score == 1.0
        assert info["eval_result"]["correct"] is True
        assert not done

    def test_step_wrong(self):
        env = MathEnv()
        env.reset()
        obs, reward, done, truncated, info = env.step({"answer": 99})

        assert reward.base_score == 0.0
        assert info["eval_result"]["correct"] is False

    def test_episode_completes(self):
        env = MathEnv()
        env.reset()

        env.step({"answer": 4})
        env.step({"answer": 9})
        obs, reward, done, truncated, info = env.step({"answer": 3})

        assert done is True

    def test_step_after_done(self):
        env = MathEnv()
        env.reset()
        for a in [4, 9, 3]:
            env.step({"answer": a})

        obs, reward, done, _, info = env.step({"answer": 1})
        assert done is True
        assert "error" in info


class TestBaseEnvMiddleware:
    """Test cognitive middleware auto-wiring."""

    def test_memory_stores_entries(self):
        env = MathEnv()
        env.reset()
        env.step({"answer": 4})

        assert len(env.memory.entries) == 1
        assert env.memory.entries[0].category == "addition"
        assert env.memory.entries[0].correct is True

    def test_safety_monitor_tracks_streak(self):
        env = MathEnv()
        env.reset()
        env.step({"answer": 99})  # wrong
        env.step({"answer": 99})  # wrong

        assert env.safety_monitor.get_wrong_streak() == 2

    def test_safety_monitor_resets_on_correct(self):
        env = MathEnv()
        env.reset()
        env.step({"answer": 99})  # wrong
        env.step({"answer": 9})  # correct (3*3=9)

        assert env.safety_monitor.get_wrong_streak() == 0

    def test_episode_stats(self):
        env = MathEnv()
        env.reset()
        env.step({"answer": 4})
        env.step({"answer": 9})
        env.step({"answer": 99})

        stats = env.episode_stats()
        assert stats.steps == 3
        assert stats.correct_count == 2
        assert abs(stats.accuracy - 2 / 3) < 0.01

    def test_state_dict(self):
        env = MathEnv()
        env.reset()
        env.step({"answer": 4})

        state = env.state()
        assert state["current_step"] == 1
        assert "memory_stats" in state
        assert "reflection_stats" in state
        assert "safety_stats" in state

    def test_get_score(self):
        env = MathEnv()
        env.reset()
        env.step({"answer": 4})  # correct
        env.step({"answer": 9})  # correct
        env.step({"answer": 3})  # correct

        score = env.get_score()
        assert score > 0.9


class TestBaseEnvPropose:
    """Test PROPOSE → Revise on the base env."""

    def test_propose_returns_feedback(self):
        env = MathEnv()
        env.reset()

        feedback = env.propose({"answer": 4})
        assert hasattr(feedback, "confidence_estimate")
        assert hasattr(feedback, "memory_context")

    def test_revise_is_actual_step(self):
        env = MathEnv()
        env.reset()

        env.propose({"answer": 99})
        obs, reward, done, truncated, info = env.revise({"answer": 4})

        assert info["eval_result"]["correct"] is True

    def test_propose_disabled(self):
        config = CogniCoreConfig(enable_propose_revise=False)
        env = MathEnv(config=config)
        env.reset()

        with pytest.raises(RuntimeError, match="disabled"):
            env.propose({"answer": 4})


class TestBaseEnvMultiEpisode:
    """Test cross-episode memory persistence."""

    def test_memory_persists(self):
        env = MathEnv()

        env.reset()
        env.step({"answer": 4})
        env.step({"answer": 9})
        env.step({"answer": 3})

        assert len(env.memory.entries) == 3

        env.reset()
        env.step({"answer": 4})

        assert len(env.memory.entries) == 4  # memory grows

    def test_episode_counter(self):
        env = MathEnv()
        env.reset()
        assert env._episode_count == 1
        env.reset()
        assert env._episode_count == 2

    def test_repr(self):
        env = MathEnv()
        env.reset()
        s = repr(env)
        assert "MathEnv" in s
        assert "episode=1" in s
