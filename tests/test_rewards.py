"""Tests for CogniCore StructuredReward and RewardBuilder."""

from cognicore.core.types import CogniCoreConfig, EvalResult, StructuredReward
from cognicore.memory.tfidf_backend import TFIDFMemoryBackend
from cognicore.memory.base import MemoryEntry
from cognicore.middleware.rewards import RewardBuilder


class TestStructuredReward:
    """Test the 8-component StructuredReward dataclass."""

    def test_total_sum(self):
        r = StructuredReward(
            base_score=1.0,
            memory_bonus=0.05,
            reflection_bonus=0.03,
            streak_penalty=-0.1,
            propose_bonus=0.05,
            novelty_bonus=0.04,
            confidence_cal=0.02,
            time_decay=-0.01,
        )
        expected = 1.0 + 0.05 + 0.03 - 0.1 + 0.05 + 0.04 + 0.02 - 0.01
        assert abs(r.total - expected) < 1e-10

    def test_as_float(self):
        r = StructuredReward(base_score=0.8)
        assert r.as_float() == r.total

    def test_to_dict(self):
        r = StructuredReward(base_score=1.0)
        d = r.to_dict()
        assert "total" in d
        assert d["base_score"] == 1.0
        assert d["total"] == 1.0

    def test_default_values(self):
        r = StructuredReward()
        assert r.total == 0.0

    def test_repr(self):
        r = StructuredReward(base_score=0.5)
        s = repr(r)
        assert "StructuredReward" in s
        assert "base=0.50" in s


class TestRewardBuilder:
    """Test the RewardBuilder middleware."""

    def test_base_score_only(self):
        config = CogniCoreConfig(enable_memory=False, enable_reflection=False)
        mem = TFIDFMemoryBackend()
        builder = RewardBuilder(config, mem)

        result = EvalResult(base_score=1.0, correct=True, category="math")
        reward = builder.build(result)

        assert reward.base_score == 1.0
        assert reward.memory_bonus == 0.0
        assert reward.total == 1.0

    def test_memory_bonus(self):
        config = CogniCoreConfig(enable_memory=True, memory_bonus_value=0.05)
        mem = TFIDFMemoryBackend()
        # Store a success for this category
        mem.store(MemoryEntry(text="", category="math", correct=True, action="42"))

        builder = RewardBuilder(config, mem)
        result = EvalResult(
            base_score=1.0, correct=True, category="math", predicted="42"
        )
        reward = builder.build(result)

        assert reward.memory_bonus == 0.05

    def test_no_memory_bonus_when_wrong(self):
        config = CogniCoreConfig(enable_memory=True)
        mem = TFIDFMemoryBackend()
        mem.store(MemoryEntry(text="", category="math", correct=True, action="42"))

        builder = RewardBuilder(config, mem)
        result = EvalResult(
            base_score=0.0, correct=False, category="math", predicted="99"
        )
        reward = builder.build(result)

        assert reward.memory_bonus == 0.0

    def test_streak_penalty_passthrough(self):
        config = CogniCoreConfig()
        mem = TFIDFMemoryBackend()
        builder = RewardBuilder(config, mem)

        result = EvalResult(base_score=0.0, correct=False, category="x")
        reward = builder.build(result, streak_penalty=-0.1)

        assert reward.streak_penalty == -0.1

    def test_novelty_bonus(self):
        config = CogniCoreConfig(novelty_bonus_value=0.04)
        mem = TFIDFMemoryBackend()
        builder = RewardBuilder(config, mem)

        result = EvalResult(base_score=1.0, correct=True, category="new")
        reward = builder.build(result, is_novel_group=True)

        assert reward.novelty_bonus == 0.04

    def test_confidence_calibration_correct(self):
        config = CogniCoreConfig(confidence_bonus_scale=0.02)
        mem = TFIDFMemoryBackend()
        builder = RewardBuilder(config, mem)

        result = EvalResult(base_score=1.0, correct=True, category="x")
        reward = builder.build(result, confidence=0.9)

        assert reward.confidence_cal > 0

    def test_confidence_calibration_wrong(self):
        config = CogniCoreConfig(confidence_bonus_scale=0.02)
        mem = TFIDFMemoryBackend()
        builder = RewardBuilder(config, mem)

        result = EvalResult(base_score=0.0, correct=False, category="x")
        reward = builder.build(result, confidence=0.9)

        assert reward.confidence_cal < 0
