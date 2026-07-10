"""Tests for CodeDebuggingEnv."""

import cognicore
from cognicore.envs.code_debugging import CodeDebuggingEnv


class TestCodeEnvBasics:
    def test_create_via_make(self):
        env = cognicore.make("CodeDebugging-v1", difficulty="easy")
        assert isinstance(env, CodeDebuggingEnv)
        assert env.difficulty == "easy"

    def test_create_all_difficulties(self):
        for d in ("easy", "medium", "hard"):
            env = cognicore.make(f"CodeDebugging-{d.capitalize()}-v1")
            assert env.difficulty == d

    def test_reset_returns_observation(self):
        env = cognicore.make("CodeDebugging-v1")
        obs = env.reset()
        assert "buggy_code" in obs
        assert "bug_description" in obs
        assert "language" in obs
        assert obs["step"] == 0

    def test_step_correct_answer(self):
        env = cognicore.make("CodeDebugging-v1", difficulty="easy")
        env.reset()
        # First easy case: wrong operator on line 2
        obs, reward, done, truncated, info = env.step(
            {
                "bug_line": 2,
                "fix_type": "wrong_operator",
            }
        )
        assert info["eval_result"]["correct"] is True
        assert reward.base_score == 1.0

    def test_step_correct_line_wrong_fix(self):
        env = cognicore.make("CodeDebugging-v1", difficulty="easy")
        env.reset()
        obs, reward, done, truncated, info = env.step(
            {
                "bug_line": 2,
                "fix_type": "syntax_error",  # wrong fix type
            }
        )
        assert reward.base_score == 0.6  # correct line, wrong fix

    def test_step_adjacent_line(self):
        env = cognicore.make("CodeDebugging-v1", difficulty="easy")
        env.reset()
        obs, reward, done, truncated, info = env.step(
            {
                "bug_line": 1,  # off by one
                "fix_type": "wrong_operator",
            }
        )
        assert reward.base_score == 0.3  # within 1 line

    def test_step_wrong_answer(self):
        env = cognicore.make("CodeDebugging-v1", difficulty="easy")
        env.reset()
        obs, reward, done, truncated, info = env.step(
            {
                "bug_line": 99,
                "fix_type": "unknown",
            }
        )
        assert reward.base_score == 0.0

    def test_episode_completes(self):
        env = cognicore.make("CodeDebugging-v1", difficulty="easy")
        env.reset()
        for _ in range(10):
            env.step({"bug_line": 1, "fix_type": "x"})
        assert env._done is True

    def test_memory_tracks_categories(self):
        env = cognicore.make("CodeDebugging-v1", difficulty="easy")
        env.reset()
        env.step({"bug_line": 2, "fix_type": "wrong_operator"})
        env.step({"bug_line": 2, "fix_type": "syntax_error"})
        assert len(env.memory.entries) == 2
        categories = {e.category for e in env.memory.entries}
        assert "operator_error" in categories or "syntax" in categories


class TestCodeGrading:
    def test_grading_function(self):
        from cognicore.envs.data.code_cases import grade_code_answer

        assert grade_code_answer(2, 2, "wrong_operator", "wrong_operator") == 1.0
        assert grade_code_answer(2, 2, "syntax", "wrong_operator") == 0.6
        assert grade_code_answer(3, 2, "wrong_operator", "wrong_operator") == 0.3
        assert grade_code_answer(99, 2, "syntax", "wrong_operator") == 0.0
