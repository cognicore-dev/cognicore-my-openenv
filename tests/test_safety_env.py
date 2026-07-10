"""Tests for SafetyClassificationEnv — the first built-in CogniCore environment."""

import cognicore
from cognicore.envs.safety_classification import SafetyClassificationEnv
from cognicore.core.types import StructuredReward


class TestSafetyEnvBasics:
    """Basic environment lifecycle."""

    def test_create_via_make(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        assert isinstance(env, SafetyClassificationEnv)
        assert env.difficulty == "easy"

    def test_create_presets(self):
        env_easy = cognicore.make("SafetyClassification-Easy-v1")
        env_med = cognicore.make("SafetyClassification-Medium-v1")
        env_hard = cognicore.make("SafetyClassification-Hard-v1")

        assert env_easy.difficulty == "easy"
        assert env_med.difficulty == "medium"
        assert env_hard.difficulty == "hard"

    def test_reset_returns_observation(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        obs = env.reset()

        assert isinstance(obs, dict)
        assert "case_id" in obs
        assert "prompt" in obs
        assert "response" in obs
        assert "step" in obs
        assert obs["step"] == 0

    def test_step_returns_5_tuple(self):
        env = cognicore.make("SafetyClassification-v1")
        env.reset()
        result = env.step({"classification": "SAFE"})

        assert len(result) == 5
        obs, reward, done, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, StructuredReward)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_reward_is_structured(self):
        env = cognicore.make("SafetyClassification-v1")
        env.reset()
        _, reward, _, _, _ = env.step({"classification": "SAFE"})

        assert hasattr(reward, "base_score")
        assert hasattr(reward, "memory_bonus")
        assert hasattr(reward, "total")
        assert hasattr(reward, "to_dict")

    def test_step_after_done_returns_error(self):
        env = cognicore.make("SafetyClassification-v1")
        env.reset()

        # Burn through all steps
        for _ in range(10):
            env.step({"classification": "SAFE"})

        obs, reward, done, _, info = env.step({"classification": "SAFE"})
        assert done is True
        assert "error" in info


class TestSafetyEnvScoring:
    """Verify scoring matches the original environment."""

    def test_perfect_easy_score(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        env.reset()

        # The correct answers for easy cases
        ground_truths = [
            "SAFE",
            "UNSAFE",
            "SAFE",
            "UNSAFE",
            "SAFE",
            "UNSAFE",
            "SAFE",
            "UNSAFE",
            "SAFE",
            "UNSAFE",
        ]

        for gt in ground_truths:
            obs, reward, done, _, info = env.step({"classification": gt})
            assert info["eval_result"]["correct"] is True

        assert done is True
        stats = env.episode_stats()
        assert stats.correct_count == 10
        assert stats.accuracy == 1.0

    def test_all_wrong_easy(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        env.reset()

        # All wrong (opposite of correct)
        wrong_answers = [
            "UNSAFE",
            "SAFE",
            "UNSAFE",
            "SAFE",
            "UNSAFE",
            "SAFE",
            "UNSAFE",
            "SAFE",
            "UNSAFE",
            "SAFE",
        ]

        for answer in wrong_answers:
            env.step({"classification": answer})

        stats = env.episode_stats()
        assert stats.correct_count == 0
        assert stats.accuracy == 0.0


class TestSafetyEnvCognitiveFeatures:
    """Test cognitive middleware integration."""

    def test_memory_grows_across_steps(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        env.reset()

        assert len(env.memory.entries) == 0
        env.step({"classification": "SAFE"})
        assert len(env.memory.entries) == 1
        env.step({"classification": "UNSAFE"})
        assert len(env.memory.entries) == 2

    def test_memory_persists_across_episodes(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")

        # Episode 1
        env.reset()
        for _ in range(10):
            env.step({"classification": "SAFE"})

        entries_after_ep1 = len(env.memory.entries)
        assert entries_after_ep1 == 10

        # Episode 2
        env.reset()
        env.step({"classification": "SAFE"})

        # Memory persists
        assert len(env.memory.entries) == entries_after_ep1 + 1

    def test_observation_includes_memory_context(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")

        # Run one episode to build memory
        env.reset()
        for _ in range(10):
            env.step({"classification": "SAFE"})

        # Start second episode — observations should have memory
        obs = env.reset()
        if obs.get("category") and len(env.memory.get_by_category(obs.get("category", ""), top_k=1)) > 0:
            assert "memory_context" in obs

    def test_safety_monitor_streak(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        env.reset()

        # Make 3 wrong answers to trigger streak
        for _ in range(3):
            _, reward, _, _, info = env.step({"classification": "NEEDS_REVIEW"})

        assert info["wrong_streak"] >= 3

    def test_episode_stats(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        env.reset()
        env.step({"classification": "SAFE"})
        env.step({"classification": "UNSAFE"})

        stats = env.episode_stats()
        assert stats.steps == 2
        assert stats.episode_number == 1

    def test_state_dict(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        env.reset()
        env.step({"classification": "SAFE"})

        state = env.state()
        assert "current_step" in state
        assert "memory_stats" in state
        assert "reflection_stats" in state
        assert "safety_stats" in state
        assert "agent_status" in state

    def test_get_score(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        env.reset()

        ground_truths = [
            "SAFE",
            "UNSAFE",
            "SAFE",
            "UNSAFE",
            "SAFE",
            "UNSAFE",
            "SAFE",
            "UNSAFE",
            "SAFE",
            "UNSAFE",
        ]
        for gt in ground_truths:
            env.step({"classification": gt})

        score = env.get_score()
        assert 0.0 < score <= 0.99
