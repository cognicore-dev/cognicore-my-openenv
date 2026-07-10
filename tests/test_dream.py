import unittest
import random
from cognicore.dream import DreamEngine, DreamEvent

class DummyEnvModel:
    def simulate(self, state, action):
        if action == "good":
            return ("state_good", 1.0, False)
        elif action == "bad":
            return ("state_bad", -1.0, False)
        elif action == "terminal":
            return ("state_end", 0.0, True)
        return (state, 0.0, False)

class DummyInvalidEnvModel:
    pass

class DummyAgent:
    def act(self, obs):
        return "good"

class DummyAgentSB3:
    def predict(self, obs):
        return ("good", None)

class TestDreamEngine(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.env_model = DummyEnvModel()
        self.action_space = ["good", "bad", "terminal", "neutral"]
        self.engine = DreamEngine(self.env_model, self.action_space)
        self.agent = DummyAgent()

    def test_dream_event_to_dict(self):
        # 1. Test DreamEvent serialization
        event = DreamEvent("s1", "act", 1.0, "s2", False)
        d = event.to_dict()
        self.assertEqual(d["reward"], 1.0)
        self.assertEqual(d["state"], "s1")

    def test_dream_basic(self):
        # 2. Test basic dream rollout
        events = self.engine.dream("start", self.agent, steps=3, random_exploration=0.0)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].action, "good")
        self.assertEqual(events[0].reward, 1.0)

    def test_dream_exploration(self):
        # 3. Test dream incorporates random exploration
        random.seed(42)
        # With high exploration, it should pick actions other than "good"
        events = self.engine.dream("start", self.agent, steps=20, random_exploration=1.0)
        actions = set(e.action for e in events)
        self.assertTrue(len(actions) > 1)

    def test_dream_terminal(self):
        # 4. Test dream stops at terminal state
        class TerminalAgent:
            def act(self, obs): return "terminal"
        events = self.engine.dream("start", TerminalAgent(), steps=10, random_exploration=0.0)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].done)

    def test_dream_sb3_interface(self):
        # 5. Test dream supports SB3 predict interface
        events = self.engine.dream("start", DummyAgentSB3(), steps=2, random_exploration=0.0)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].action, "good")

    def test_dream_invalid_env(self):
        # 6. Test dream handles invalid env model gracefully
        bad_engine = DreamEngine(DummyInvalidEnvModel())
        events = bad_engine.dream("start", self.agent)
        self.assertEqual(len(events), 0)

    def test_nightmare_basic(self):
        # 7. Test nightmare always picks worst action
        events = self.engine.nightmare("start", steps=3)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].action, "bad")
        self.assertEqual(events[0].reward, -1.0)

    def test_nightmare_terminal(self):
        # 8. Test nightmare stops at terminal state
        # Modify env model to make terminal the worst
        class WorseTerminalEnv:
            def simulate(self, state, action):
                if action == "terminal": return ("end", -10.0, True)
                return (state, 0.0, False)
        
        night_engine = DreamEngine(WorseTerminalEnv(), ["neutral", "terminal"])
        events = night_engine.nightmare("start", steps=5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].action, "terminal")

    def test_nightmare_no_action_space(self):
        # 9. Test nightmare requires action space
        no_act_engine = DreamEngine(self.env_model)
        events = no_act_engine.nightmare("start")
        self.assertEqual(len(events), 0)

    def test_nightmare_invalid_env(self):
        # 10. Test nightmare handles invalid env model gracefully
        bad_engine = DreamEngine(DummyInvalidEnvModel(), self.action_space)
        events = bad_engine.nightmare("start")
        self.assertEqual(len(events), 0)

    def test_hallucinate_goals_basic(self):
        # 11. Test hallucination finds simple goal
        random.seed(42)
        trajectory = self.engine.hallucinate_goals("start", target_reward=2.0, steps=5)
        # Needs 2 "good" actions minimum
        self.assertTrue(len(trajectory) >= 2)
        # Should sum to at least 2.0
        r_sum = sum(1.0 for a in trajectory if a == "good") - sum(1.0 for a in trajectory if a == "bad")
        self.assertGreaterEqual(r_sum, 2.0)

    def test_hallucinate_goals_impossible(self):
        # 12. Test hallucination returns empty for impossible goal
        trajectory = self.engine.hallucinate_goals("start", target_reward=100.0, steps=5, search_budget=10)
        self.assertEqual(len(trajectory), 0)

    def test_hallucinate_goals_terminal_cutoff(self):
        # 13. Test hallucination stops exploring dead ends
        # If it hits terminal it breaks that trajectory early
        trajectory = self.engine.hallucinate_goals("start", target_reward=1.0, steps=2)
        self.assertTrue(len(trajectory) > 0)
        self.assertNotIn("terminal", trajectory[:-1]) # Cannot have terminal before the end

    def test_hallucinate_goals_no_action_space(self):
        # 14. Test hallucination requires action space
        no_act_engine = DreamEngine(self.env_model)
        trajectory = no_act_engine.hallucinate_goals("start", 1.0)
        self.assertEqual(len(trajectory), 0)

    def test_hallucinate_goals_invalid_env(self):
        # 15. Test hallucination handles invalid env gracefully
        bad_engine = DreamEngine(DummyInvalidEnvModel(), self.action_space)
        trajectory = bad_engine.hallucinate_goals("start", 1.0)
        self.assertEqual(len(trajectory), 0)

if __name__ == "__main__":
    unittest.main()
