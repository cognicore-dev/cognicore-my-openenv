"""Tests for MemoryManager, Leaderboard, EpisodeRecorder, and MultiAgentEnv."""

import os
import json
import shutil

import cognicore
from cognicore.memory_manager import MemoryManager
from cognicore.leaderboard import Leaderboard
from cognicore.finetuning import EpisodeRecorder, export_jsonl, export_reward_dataset
from cognicore.multi_agent import DebateEnv


worker_id = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
TEST_DIR = os.path.join(os.path.dirname(__file__), "..", f"_test_data_new_features_{worker_id}")

def setup_module():
    os.makedirs(TEST_DIR, exist_ok=True)

def teardown_module():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR, ignore_errors=True)


# ---- MemoryManager ----


class TestMemoryManager:
    def test_save_and_load(self):
        mgr = MemoryManager(storage_dir=os.path.join(TEST_DIR, "mem"))
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        env.reset()
        env.step({"classification": "SAFE"})
        env.step({"classification": "UNSAFE"})

        path = mgr.save_session("test-agent", env)
        assert os.path.exists(path)

        # Load into a fresh env
        env2 = cognicore.make("SafetyClassification-v1", difficulty="easy")
        env2.reset()
        assert mgr.load_session("test-agent", env2) is True
        assert len(env2.memory.entries) == 2

    def test_list_sessions(self):
        mgr = MemoryManager(storage_dir=os.path.join(TEST_DIR, "mem2"))
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        env.reset()
        env.step({"classification": "SAFE"})

        mgr.save_session("agent-a", env)
        mgr.save_session("agent-b", env)

        sessions = mgr.list_sessions()
        ids = [s["agent_id"] for s in sessions]
        assert "agent-a" in ids
        assert "agent-b" in ids

    def test_get_history(self):
        mgr = MemoryManager(storage_dir=os.path.join(TEST_DIR, "mem3"))
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        env.reset()
        env.step({"classification": "SAFE"})

        mgr.save_session("hist-agent", env)
        history = mgr.get_history("hist-agent")
        assert len(history) == 1
        assert "accuracy" in history[0]

    def test_delete_session(self):
        mgr = MemoryManager(storage_dir=os.path.join(TEST_DIR, "mem4"))
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        env.reset()
        env.step({"classification": "SAFE"})
        mgr.save_session("del-agent", env)
        assert mgr.delete_session("del-agent") is True
        assert mgr.load_session("del-agent", env) is False

    def test_load_nonexistent(self):
        mgr = MemoryManager(storage_dir=os.path.join(TEST_DIR, "mem5"))
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        env.reset()
        assert mgr.load_session("nobody", env) is False


# ---- Leaderboard ----


class TestLeaderboard:
    def test_submit_and_rank(self):
        lb = Leaderboard(storage_dir=os.path.join(TEST_DIR, "lb1"))
        lb.clear()

        lb.submit("agent-x", "SafetyClassification-v1", score=0.8, accuracy=0.7)
        lb.submit("agent-y", "SafetyClassification-v1", score=0.9, accuracy=0.8)
        lb.submit("agent-z", "SafetyClassification-v1", score=0.7, accuracy=0.6)

        rankings = lb.get_rankings("SafetyClassification-v1")
        assert rankings[0]["agent_id"] == "agent-y"
        assert rankings[0]["rank"] == 1
        assert rankings[1]["agent_id"] == "agent-x"
        assert rankings[2]["agent_id"] == "agent-z"

    def test_best_score_per_agent(self):
        lb = Leaderboard(storage_dir=os.path.join(TEST_DIR, "lb2"))
        lb.clear()

        lb.submit("agent-a", "EnvA", score=0.5, accuracy=0.4)
        lb.submit("agent-a", "EnvA", score=0.9, accuracy=0.8)
        lb.submit("agent-a", "EnvA", score=0.6, accuracy=0.5)

        rankings = lb.get_rankings("EnvA")
        assert len(rankings) == 1  # only best score kept
        assert rankings[0]["score"] == 0.9

    def test_stats(self):
        lb = Leaderboard(storage_dir=os.path.join(TEST_DIR, "lb3"))
        lb.clear()
        lb.submit("a1", "E1", score=0.5, accuracy=0.5)
        lb.submit("a2", "E2", score=0.6, accuracy=0.6)

        stats = lb.get_stats()
        assert stats["total_submissions"] == 2
        assert stats["unique_agents"] == 2
        assert stats["unique_environments"] == 2

    def test_filter_by_difficulty(self):
        lb = Leaderboard(storage_dir=os.path.join(TEST_DIR, "lb4"))
        lb.clear()
        lb.submit("a1", "E1", score=0.5, accuracy=0.5, difficulty="easy")
        lb.submit("a2", "E1", score=0.9, accuracy=0.9, difficulty="hard")

        easy_rankings = lb.get_rankings("E1", difficulty="easy")
        assert len(easy_rankings) == 1
        assert easy_rankings[0]["agent_id"] == "a1"


# ---- EpisodeRecorder & Fine-tuning ----


class TestFineTuning:
    def test_record_episode(self):
        recorder = EpisodeRecorder()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        obs = env.reset()
        recorder.start_episode(env_id="SafetyClassification-v1")

        for _ in range(10):
            action = {"classification": "SAFE"}
            obs, reward, done, _, info = env.step(action)
            recorder.record_step(obs, action, reward, info)
            if done:
                break

        recorder.end_episode(
            score=env.get_score(),
            accuracy=env.episode_stats().accuracy,
        )

        assert len(recorder.episodes) == 1
        assert recorder.total_steps == 10

    def test_export_jsonl(self):
        recorder = EpisodeRecorder()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        obs = env.reset()
        recorder.start_episode()

        for _ in range(10):
            obs, reward, done, _, info = env.step({"classification": "SAFE"})
            recorder.record_step(obs, {"classification": "SAFE"}, reward, info)
            if done:
                break

        recorder.end_episode()

        out = os.path.join(TEST_DIR, "train.jsonl")
        count = export_jsonl(recorder.episodes, out, system_prompt="Test")
        assert count > 0
        assert os.path.exists(out)

        # Verify JSONL format
        with open(out) as f:
            for line in f:
                data = json.loads(line)
                assert "messages" in data

    def test_export_reward_dataset(self):
        recorder = EpisodeRecorder()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        obs = env.reset()
        recorder.start_episode()

        for _ in range(10):
            obs, reward, done, _, info = env.step({"classification": "SAFE"})
            recorder.record_step(obs, {"classification": "SAFE"}, reward, info)
            if done:
                break
        recorder.end_episode()

        out = os.path.join(TEST_DIR, "reward.jsonl")
        count = export_reward_dataset(recorder.episodes, out)
        assert count == 10


# ---- Multi-Agent ----


class TestMultiAgent:
    def test_debate_env_create(self):
        env = DebateEnv()
        assert env.num_agents == 2
        assert env.agent_ids == ["pro", "con"]

    def test_debate_env_reset(self):
        env = DebateEnv()
        multi_obs = env.reset()
        assert "pro" in multi_obs
        assert "con" in multi_obs
        assert multi_obs["pro"]["your_side"] == "pro"
        assert multi_obs["con"]["your_side"] == "con"
        assert "topic" in multi_obs["pro"]

    def test_debate_env_step_agents(self):
        env = DebateEnv()
        env.reset()

        # Pro submits
        result1 = env.step_agent(
            "pro", {"argument": "AI safety and alignment are critical risks"}
        )
        assert result1["status"] == "waiting"

        # Con submits
        result2 = env.step_agent(
            "con", {"argument": "Progress and innovation require continued development"}
        )
        # Both submitted -> resolved
        assert "_done" in result2
        assert "pro" in result2
        assert "con" in result2

    def test_debate_env_scoring(self):
        env = DebateEnv()
        env.reset()

        # Pro hits all key points
        env.step_agent(
            "pro",
            {
                "argument": "AI poses safety risks, alignment is unsolved, regulation needed to manage risk"
            },
        )
        result = env.step_agent(
            "con", {"argument": "generic argument with no key points"}
        )

        pro_score = result["pro"]["eval_result"]["base_score"]
        con_score = result["con"]["eval_result"]["base_score"]
        assert pro_score > con_score

    def test_debate_env_multi_rounds(self):
        env = DebateEnv()
        env.reset()

        for _ in range(5):  # 5 topics
            env.step_agent("pro", {"argument": "test pro"})
            result = env.step_agent("con", {"argument": "test con"})

        assert result["_done"] is True
