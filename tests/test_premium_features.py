"""Tests for Phase 5 premium features."""

import cognicore
from cognicore.advanced_memory import SemanticMemory
from cognicore.explainer import Explainer
from cognicore.adversarial import AdversarialTester
from cognicore.smart_agents import AutoLearner, SafeAgent, AdaptiveAgent
from cognicore.auto_improve import auto_improve
from cognicore.safety_layer import SafetyLayer, Policy
from cognicore.cost_tracker import CostTracker


class TestSemanticMemory:
    def test_store_and_recall(self):
        mem = SemanticMemory()
        mem.store(
            {"text": "phishing email scam", "correct": False, "category": "security"}
        )
        mem.store(
            {"text": "cooking recipe pasta", "correct": True, "category": "cooking"}
        )
        results = mem.recall("email fraud phishing", top_k=2)
        assert len(results) >= 1

    def test_decay(self):
        mem = SemanticMemory(decay_rate=0.5)
        mem.store({"text": "old entry", "correct": True})
        mem.store({"text": "new entry", "correct": True})
        # First entry should have lower relevance
        assert mem.entries[0]["_relevance"] < mem.entries[1]["_relevance"]

    def test_best_worst_actions(self):
        mem = SemanticMemory()
        mem.store({"text": "security hack", "correct": True, "predicted": "UNSAFE"})
        mem.store({"text": "security breach", "correct": False, "predicted": "SAFE"})
        best = mem.best_actions("security exploit")
        worst = mem.worst_actions("security exploit")
        assert isinstance(best, list)
        assert isinstance(worst, list)

    def test_adaptive_context(self):
        mem = SemanticMemory()
        mem.store({"text": "test input", "correct": True})
        ctx = mem.get_adaptive_context("test", agent_accuracy=0.2)
        assert ctx["strategy"] == "learning_from_mistakes"
        ctx = mem.get_adaptive_context("test", agent_accuracy=0.9)
        assert ctx["strategy"] == "reinforcing_success"

    def test_stats(self):
        mem = SemanticMemory()
        mem.store({"text": "hello", "correct": True})
        stats = mem.stats()
        assert stats["total_entries"] == 1
        assert stats["total_stored"] == 1


class TestExplainer:
    def test_record_correct(self):
        exp = Explainer()
        result = exp.record_step(1, "security", "UNSAFE", "UNSAFE", True)
        assert result["verdict"] == "CORRECT"

    def test_record_wrong(self):
        exp = Explainer()
        result = exp.record_step(1, "security", "SAFE", "UNSAFE", False)
        assert result["verdict"] == "WRONG"
        assert "why_wrong" in result
        assert "suggestion" in result

    def test_pattern_detection(self):
        exp = Explainer()
        exp.record_step(1, "security", "SAFE", "UNSAFE", False)
        exp.record_step(2, "security", "SAFE", "UNSAFE", False)
        exp.record_step(3, "cooking", "SAFE", "SAFE", True)
        report = exp.explain()
        patterns = report.mistake_patterns()
        assert len(patterns) >= 1

    def test_improvement_plan(self):
        exp = Explainer()
        exp.record_step(1, "security", "SAFE", "UNSAFE", False)
        exp.record_step(2, "security", "SAFE", "UNSAFE", False)
        report = exp.explain()
        plan = report.improvement_plan()
        assert len(plan) >= 1

    def test_step_log(self):
        exp = Explainer()
        exp.record_step(1, "cat1", "A", "B", False)
        exp.record_step(2, "cat2", "A", "A", True)
        report = exp.explain()
        log = report.step_by_step_log()
        assert len(log) == 2
        assert log[0]["status"] == "FAIL"
        assert log[1]["status"] == "OK"


class TestAdversarial:
    def test_create_tester(self):
        tester = AdversarialTester("SafetyClassification-v1")
        assert tester.env_id == "SafetyClassification-v1"

    def test_break_my_agent(self):
        tester = AdversarialTester("SafetyClassification-v1")
        failures = tester.break_my_agent(None, max_attempts=2)
        assert isinstance(failures, list)

    def test_stress_test(self):
        tester = AdversarialTester("SafetyClassification-v1")
        report = tester.stress_test(None, rounds=2, verbose=False)
        assert report.injection_resistance >= 0
        assert report.edge_case_handling >= 0
        assert report.stress_stability >= 0
        assert isinstance(report.vulnerabilities(), list)


class TestSmartAgents:
    def test_auto_learner(self):
        agent = AutoLearner()
        obs = {
            "category": "test",
            "prompt": "malware attack",
            "memory_context": [],
            "reflection_hints": "",
        }
        action = agent.act(obs)
        assert "classification" in action

    def test_safe_agent(self):
        agent = SafeAgent()
        obs = {
            "category": "unknown",
            "prompt": "some text",
            "memory_context": [],
            "reflection_hints": "",
        }
        action = agent.act(obs)
        # SafeAgent should default to NEEDS_REVIEW when uncertain
        assert action["classification"] in ("NEEDS_REVIEW", "SAFE", "UNSAFE")

    def test_adaptive_agent(self):
        agent = AdaptiveAgent()
        assert agent.strategy == "exploring"
        obs = {
            "category": "test",
            "prompt": "hello",
            "memory_context": [],
            "reflection_hints": "",
        }
        action = agent.act(obs)
        assert "classification" in action

    def test_auto_learner_learns(self):
        agent = AutoLearner()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        obs = env.reset()
        for _ in range(10):
            action = agent.act(obs)
            obs, reward, done, _, info = env.step(action)
            agent.learn(reward, info)
            if done:
                break
        assert len(agent.history) > 0


class TestAutoImprove:
    def test_auto_improve_runs(self):
        result = auto_improve(
            env_id="SafetyClassification-v1",
            difficulty="easy",
            max_cycles=2,
            episodes_per_cycle=1,
            verbose=False,
            target_accuracy=1.1,
        )
        assert "cycles" in result
        assert result["cycles"] == 2
        assert "initial_accuracy" in result
        assert "final_accuracy" in result


class TestSafetyLayer:
    def test_risk_scoring(self):
        safety = SafetyLayer()
        score = safety.compute_risk("this contains malware and hacking tools")
        assert score >= 80

    def test_safe_text(self):
        safety = SafetyLayer()
        score = safety.compute_risk("cooking recipe with pasta")
        assert score == 0

    def test_check_action(self):
        safety = SafetyLayer()
        result = safety.check({"classification": "SAFE"}, {"prompt": "hello"})
        assert result["allowed"] is True
        assert result["risk_score"] == 0

    def test_custom_policy(self):
        safety = SafetyLayer()
        safety.add_policy(
            Policy(
                "block_safe",
                lambda a, c: a.get("classification") != "SAFE",
                severity="CRITICAL",
            )
        )
        result = safety.check({"classification": "SAFE"})
        assert "block_safe" in result["violations"]

    def test_audit_log(self):
        safety = SafetyLayer()
        safety.check({"classification": "SAFE"})
        safety.check({"classification": "UNSAFE"})
        log = safety.audit_log()
        assert len(log) == 2

    def test_compliance_report(self):
        safety = SafetyLayer()
        safety.check({"classification": "SAFE"})
        report = safety.compliance_report()
        assert report["total_checks"] == 1


class TestCostTracker:
    def test_record_call(self):
        tracker = CostTracker(model_name="gemini-flash")
        result = tracker.record_call(input_tokens=100, output_tokens=50)
        assert result["call_cost"] > 0
        assert result["total_cost"] > 0

    def test_budget_limit(self):
        tracker = CostTracker(budget_limit=0.001)
        tracker.record_call(input_tokens=10000, output_tokens=5000)
        result = tracker.record_call(input_tokens=10000, output_tokens=5000)
        assert result["over_budget"] is True

    def test_estimate_tokens(self):
        tracker = CostTracker()
        tokens = tracker.estimate_tokens("hello world this is a test")
        assert tokens > 0

    def test_record_text(self):
        tracker = CostTracker()
        result = tracker.record_text("input prompt here", "output response")
        assert result["call_cost"] > 0

    def test_summary(self):
        tracker = CostTracker()
        tracker.record_call(100, 50)
        s = tracker.summary()
        assert s["total_calls"] == 1
        assert s["total_tokens"] == 150

    def test_compare_models(self):
        tracker = CostTracker(model_name="gemini-flash")
        tracker.record_call(1000, 500)
        comp = tracker.compare_models()
        assert "gemini-flash" in comp
        assert "gpt-4o" in comp
        assert comp["gpt-4o"] > comp["gemini-flash"]
