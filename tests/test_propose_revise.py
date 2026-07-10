"""Tests for the PROPOSE → Revise protocol."""

import pytest

from cognicore.memory.tfidf_backend import TFIDFMemoryBackend
from cognicore.memory.base import MemoryEntry
from cognicore.middleware.reflection import ReflectionEngine
from cognicore.middleware.propose_revise import ProposeReviseProtocol


class TestProposeRevise:
    def _make_protocol(self, max_proposals=1):
        mem = TFIDFMemoryBackend()
        refl = ReflectionEngine(mem)
        return ProposeReviseProtocol(mem, refl, max_proposals=max_proposals)

    def test_basic_propose(self):
        pr = self._make_protocol()
        pr.begin_step()
        feedback = pr.propose({"classification": "SAFE"}, "math")

        assert feedback.confidence_estimate == 0.5  # no history
        assert feedback.metadata["proposals_this_step"] == 1

    def test_max_proposals_enforced(self):
        pr = self._make_protocol(max_proposals=1)
        pr.begin_step()
        pr.propose({"classification": "SAFE"}, "math")

        with pytest.raises(RuntimeError, match="Max proposals"):
            pr.propose({"classification": "UNSAFE"}, "math")

    def test_propose_without_begin_step(self):
        pr = self._make_protocol()
        # Step is not active
        with pytest.raises(RuntimeError, match="No active step"):
            pr.propose({"classification": "SAFE"}, "math")

    def test_check_improvement_changed_and_correct(self):
        pr = self._make_protocol()
        pr.begin_step()
        pr.propose({"classification": "SAFE"}, "math")

        # Agent changed to UNSAFE and got it correct
        improved = pr.check_improvement("UNSAFE", eval_correct=True)
        assert improved is True

    def test_check_improvement_same_action(self):
        pr = self._make_protocol()
        pr.begin_step()
        pr.propose({"classification": "SAFE"}, "math")

        # Agent kept SAFE — no improvement even if correct
        improved = pr.check_improvement("SAFE", eval_correct=True)
        assert improved is False

    def test_check_improvement_changed_but_wrong(self):
        pr = self._make_protocol()
        pr.begin_step()
        pr.propose({"classification": "SAFE"}, "math")

        # Agent changed but got it wrong — not an improvement
        improved = pr.check_improvement("UNSAFE", eval_correct=False)
        assert improved is False

    def test_has_pending_proposal(self):
        pr = self._make_protocol()
        pr.begin_step()
        assert not pr.has_pending_proposal

        pr.propose({"classification": "SAFE"}, "math")
        assert pr.has_pending_proposal

        pr.end_step()
        assert not pr.has_pending_proposal

    def test_stats(self):
        pr = self._make_protocol()
        pr.begin_step()
        pr.propose({"classification": "SAFE"}, "math")
        pr.check_improvement("UNSAFE", True)
        pr.end_step()

        stats = pr.stats()
        assert stats["total_proposals"] == 1
        assert stats["total_improvements"] == 1

    def test_confidence_estimate_with_history(self):
        mem = TFIDFMemoryBackend()
        # Simulate past entries
        mem.store(MemoryEntry(text="", category="math", correct=True, action="42"))
        mem.store(MemoryEntry(text="", category="math", correct=True, action="42"))
        mem.store(MemoryEntry(text="", category="math", correct=False, action="99"))

        refl = ReflectionEngine(mem)
        pr = ProposeReviseProtocol(mem, refl)
        pr.begin_step()
        feedback = pr.propose({"classification": "SAFE"}, "math")

        # 2 out of 3 correct ≈ 0.667
        assert abs(feedback.confidence_estimate - 2 / 3) < 0.01
