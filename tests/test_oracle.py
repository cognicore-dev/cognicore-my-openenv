import unittest
from cognicore.oracle import Oracle

class DummyAgent:
    def act(self, obs):
        if "state1" in obs: return "act2"
        if "state2" in obs: return "act3"
        return "default"

class DummyAgentSB3:
    def predict(self, obs):
        return ("sb3_act", None)

class DummyEnvModel:
    def simulate(self, state, action):
        if action == "act1":
            return ("state1", 1.0, False)
        elif action == "act2":
            return ("state2", 2.0, False)
        elif action == "act3":
            return ("state3", -5.0, True)
        elif action == "safe_act":
            return ("state_safe", 0.5, False)
        elif action == "sb3_act":
            return ("state_sb3", 1.5, False)
        return (state, 0.0, False)

class DummyInvalidEnvModel:
    pass

class TestOracle(unittest.TestCase):
    def setUp(self):
        self.agent = DummyAgent()
        self.env_model = DummyEnvModel()
        self.oracle = Oracle(self.agent, self.env_model)

    def test_predict_outcome_basic(self):
        # 1. Test basic prediction formats
        res = self.oracle.predict_outcome("start", "act1", steps=1, rollouts=1)
        self.assertIn("expected_reward", res)
        self.assertIn("success_probability", res)
        self.assertIn("risk_score", res)
        self.assertEqual(res["expected_reward"], 1.0)
        self.assertEqual(res["success_probability"], 1.0)
        self.assertEqual(res["risk_score"], 0.0)

    def test_predict_outcome_multi_step(self):
        # 2. Test multi-step rollout uses agent act
        # step 1: act1 -> state1 (1.0)
        # step 2: act(state1)=act2 -> state2 (2.0)
        res = self.oracle.predict_outcome("start", "act1", steps=2, rollouts=1)
        self.assertEqual(res["expected_reward"], 3.0)

    def test_predict_outcome_terminal(self):
        # 3. Test rollout stops at terminal state
        # act3 -> state3 (-5.0, True)
        res = self.oracle.predict_outcome("start", "act3", steps=5, rollouts=1)
        self.assertEqual(res["expected_reward"], -5.0)

    def test_predict_outcome_risk(self):
        # 4. Test risk score registers failures
        res = self.oracle.predict_outcome("start", "act3", steps=1, rollouts=5)
        self.assertEqual(res["risk_score"], 1.0)
        self.assertEqual(res["success_probability"], 0.0)

    def test_predict_outcome_sb3(self):
        # 5. Test compatibility with SB3 agent predict method
        sb3_oracle = Oracle(DummyAgentSB3(), self.env_model)
        # step 1: act1 -> state1 (1.0)
        # step 2: predict(state1)=sb3_act -> state_sb3 (1.5)
        res = sb3_oracle.predict_outcome("start", "act1", steps=2, rollouts=1)
        self.assertEqual(res["expected_reward"], 2.5)

    def test_best_action(self):
        # 6. Test best_action selects highest reward action
        best = self.oracle.best_action("start", ["act1", "act3", "safe_act"], steps=1, rollouts=1)
        self.assertEqual(best, "act1") # 1.0 vs -5.0 vs 0.5

    def test_best_action_multi_step(self):
        # 7. Test best_action evaluates long-term
        # act1 -> 1.0 + 2.0 (act2) = 3.0
        # safe_act -> 0.5 + 0.0 (default) = 0.5
        best = self.oracle.best_action("start", ["act1", "safe_act"], steps=2, rollouts=1)
        self.assertEqual(best, "act1")

    def test_best_action_empty(self):
        # 8. Test best_action handles empty candidates
        self.assertIsNone(self.oracle.best_action("start", []))

    def test_what_if_sequence(self):
        # 9. Test what_if correctly follows forced sequence
        res = self.oracle.what_if("start", ["act1", "act1", "act2"])
        # step 1: act1 -> state1 (1.0)
        # step 2: act1 -> state1 (1.0)
        # step 3: act2 -> state2 (2.0)
        self.assertEqual(res["total_reward"], 4.0)
        self.assertEqual(res["final_state"], "state2")
        self.assertEqual(res["completed_steps"], 3)

    def test_what_if_terminal_early(self):
        # 10. Test what_if stops on terminal state
        res = self.oracle.what_if("start", ["act1", "act3", "act2"])
        # step 1: act1 -> state1 (1.0)
        # step 2: act3 -> state3 (-5.0, True)
        # step 3 is skipped
        self.assertEqual(res["total_reward"], -4.0)
        self.assertEqual(res["final_state"], "state3")
        self.assertEqual(res["completed_steps"], 2)

    def test_what_if_empty(self):
        # 11. Test what_if with empty sequence
        res = self.oracle.what_if("start", [])
        self.assertEqual(res["total_reward"], 0.0)
        self.assertEqual(res["completed_steps"], 0)
        self.assertEqual(res["final_state"], "start")

    def test_explain_prediction_format(self):
        # 12. Test explain_prediction output structure
        exp = self.oracle.explain_prediction("start", "act1", steps=2)
        self.assertIn("Oracle Simulation for action 'act1'", exp)
        self.assertIn("Step 1:", exp)
        self.assertIn("Step 2:", exp)
        self.assertIn("Total Expected Reward", exp)

    def test_explain_prediction_terminal(self):
        # 13. Test explain_prediction mentions terminal state
        exp = self.oracle.explain_prediction("start", "act3", steps=5)
        self.assertIn("Terminal state reached", exp)
        self.assertNotIn("Step 2:", exp)

    def test_invalid_env_model_predict(self):
        # 14. Test handling of environment models missing simulate method
        bad_oracle = Oracle(self.agent, DummyInvalidEnvModel())
        res = bad_oracle.predict_outcome("start", "act1")
        self.assertEqual(res["expected_reward"], 0.0)

    def test_invalid_env_model_explain(self):
        # 15. Test explain handles bad env model safely
        bad_oracle = Oracle(self.agent, DummyInvalidEnvModel())
        exp = bad_oracle.explain_prediction("start", "act1")
        self.assertIn("Error: Environment model lacks 'simulate' method", exp)

if __name__ == "__main__":
    unittest.main()
