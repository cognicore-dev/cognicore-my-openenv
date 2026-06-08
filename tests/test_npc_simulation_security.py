import pytest

from cognicore.envs.npc_simulation import _safe_eval_goal


def test_safe_eval_goal_with_valid_expression():
    assert _safe_eval_goal("trust >= 70 and aggression < 30", {"trust": 75, "aggression": 20, "mood": 50}) is True


def test_safe_eval_goal_rejects_function_calls():
    with pytest.raises(ValueError):
        _safe_eval_goal("__import__('os').system('echo hack')", {"trust": 75, "aggression": 20, "mood": 50})
