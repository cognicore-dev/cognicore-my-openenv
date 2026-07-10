import unittest
import numpy as np
from cognicore.conscience import Conscience, ConscienceTracker, ConscienceEvent

class DummyEnv:
    def step(self, action):
        return (None, 0.5, False, False, {})

class DummyAgent:
    def __init__(self):
        self.env = DummyEnv()
        self.act_call_count = 0
        
    def act(self, obs):
        self.act_call_count += 1
        return "default_action"

class DummyAgentSB3(DummyAgent):
    def predict(self, obs):
        self.act_call_count += 1
        return ("sb3_action", None)

def dummy_policy_hold(event):
    return "hold"

def dummy_policy_escalate(event):
    return "escalate"
    
def dummy_policy_override(event):
    return "override(safe_action)"

class TestConscience(unittest.TestCase):
    def setUp(self):
        self.agent = DummyAgent()

    def test_wrap_maintains_duck_typing(self):
        # 1. Test wrapping maintains base attributes
        wrapped = Conscience.wrap(self.agent, 0.5, dummy_policy_hold)
        self.assertTrue(hasattr(wrapped, "act"))
        self.assertTrue(hasattr(wrapped, "env"))
        self.assertIsInstance(wrapped.conscience, ConscienceTracker)

class DummyAgentSB3:
    def __init__(self):
        self.env = DummyEnv()
        self.act_call_count = 0
        
    def predict(self, obs):
        self.act_call_count += 1
        return ("sb3_action", None)

    def test_wrap_sb3_interface(self):
        # 2. Test SB3 style predict method
        sb3_agent = DummyAgentSB3()
        wrapped = Conscience.wrap(sb3_agent, 0.5, dummy_policy_hold)
        act, _ = wrapped.predict("obs")
        self.assertEqual(act, "sb3_action")

    def test_threshold_proceed(self):
        # 3. Test action proceeds if threshold is low (0.0)
        wrapped = Conscience.wrap(self.agent, 0.0, dummy_policy_hold)
        action = wrapped.act("obs")
        self.assertEqual(action, "default_action")
        self.assertEqual(wrapped.conscience.events[-1].outcome, "proceed")

    def test_threshold_hold(self):
        # 4. Test action holds if threshold is high (1.0)
        wrapped = Conscience.wrap(self.agent, 1.0, dummy_policy_hold)
        action = wrapped.act("obs")
        self.assertIsNone(action)
        self.assertEqual(wrapped.conscience.events[-1].outcome, "hold")

    def test_escalation_routing_escalate(self):
        # 5. Test routing to escalate
        wrapped = Conscience.wrap(self.agent, 1.0, dummy_policy_escalate)
        action = wrapped.act("obs")
        self.assertIsNone(action)
        self.assertEqual(wrapped.conscience.events[-1].outcome, "escalate")

    def test_escalation_routing_override(self):
        # 6. Test routing to override action
        wrapped = Conscience.wrap(self.agent, 1.0, dummy_policy_override)
        action = wrapped.act("obs")
        self.assertEqual(action, "safe_action")
        self.assertEqual(wrapped.conscience.events[-1].outcome, "override(safe_action)")

    def test_escalation_override_type_cast(self):
        # 7. Test override correctly casts integer action back
        class IntAgent:
            def act(self, obs): return 1
        wrapped = Conscience.wrap(IntAgent(), 1.0, lambda e: "override(99)")
        action = wrapped.act("obs")
        self.assertEqual(action, 99)
        self.assertIsInstance(action, int)

    def test_uncertainty_score_numpy(self):
        # 8. Test uncertainty handles numpy arrays
        wrapped = Conscience.wrap(self.agent, 0.0, dummy_policy_hold)
        wrapped.act(np.array([1.0, 2.0]))
        event = wrapped.conscience.events[-1]
        self.assertIn("uncertainty", event.scores)
        self.assertTrue(0.0 <= event.scores["uncertainty"] <= 1.0)

    def test_novelty_score(self):
        # 9. Test novelty metric computes without memory
        wrapped = Conscience.wrap(self.agent, 0.0, dummy_policy_hold)
        wrapped.act("obs")
        event = wrapped.conscience.events[-1]
        self.assertEqual(event.scores["novelty"], 0.5)

    def test_consequence_score(self):
        # 10. Test consequence 1-step lookahead
        wrapped = Conscience.wrap(self.agent, 0.0, dummy_policy_hold)
        wrapped.act("obs")
        event = wrapped.conscience.events[-1]
        self.assertTrue(0.0 <= event.scores["consequence"] <= 1.0)

    def test_regret_score_computation(self):
        # 11. Test regret score calculates ratio
        tracker = ConscienceTracker(None, 0.5, None)
        tracker.events = [
            ConscienceEvent(1, "o", "a", {}, 0.1, 0.5, "hold", True),
            ConscienceEvent(2, "o", "a", {}, 0.1, 0.5, "hold", False),
            ConscienceEvent(3, "o", "a", {}, 0.9, 0.5, "proceed", True)
        ]
        self.assertEqual(tracker.regret_score(), 0.5) # 1 correct out of 2 holds

    def test_regret_score_empty(self):
        # 12. Test regret score with no holds
        tracker = ConscienceTracker(None, 0.5, None)
        self.assertEqual(tracker.regret_score(), 0.0)

    def test_calibration_raises_threshold(self):
        # 13. Test calibration raises threshold when regret is low
        tracker = ConscienceTracker(None, 0.5, None)
        tracker.events = [
            ConscienceEvent(1, "o", "a", {}, 0.1, 0.5, "hold", False)
        ] # regret = 0.0 (below target 0.1)
        tracker.calibrate(target_regret=0.1)
        self.assertAlmostEqual(tracker.threshold, 0.55)

    def test_calibration_lowers_threshold(self):
        # 14. Test calibration lowers threshold when regret is high
        tracker = ConscienceTracker(None, 0.5, None)
        tracker.events = [
            ConscienceEvent(1, "o", "a", {}, 0.1, 0.5, "hold", True)
        ] # regret = 1.0 (above target 0.1)
        tracker.calibrate(target_regret=0.1)
        self.assertAlmostEqual(tracker.threshold, 0.45)

    def test_explain_valid_step(self):
        # 15. Test explanation generation for valid step
        wrapped = Conscience.wrap(self.agent, 0.0, dummy_policy_hold)
        wrapped.act("my_obs")
        exp = wrapped.conscience.explain(1)
        self.assertIn("At step 1", exp)
        self.assertIn("default_action", exp)
        self.assertIn("proceed", exp)

    def test_explain_invalid_step(self):
        # 16. Test explanation handles missing steps
        tracker = ConscienceTracker(None, 0.5, None)
        self.assertEqual(tracker.explain(99), "No conscience event recorded for step 99.")

    def test_audit_log_format(self):
        # 17. Test audit log dictionary format
        wrapped = Conscience.wrap(self.agent, 0.0, dummy_policy_hold)
        wrapped.act("my_obs")
        log = wrapped.conscience.audit_log()
        self.assertEqual(len(log), 1)
        self.assertIn("step", log[0])
        self.assertIn("combined_score", log[0])

    def test_edge_case_step_zero(self):
        # 18. Test edge case: step zero tracking (1-indexed naturally)
        wrapped = Conscience.wrap(self.agent, 0.0, dummy_policy_hold)
        wrapped.act("first")
        self.assertEqual(wrapped.conscience.events[-1].step, 1)

if __name__ == "__main__":
    unittest.main()
