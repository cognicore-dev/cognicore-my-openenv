"""
CogniCore PROPOSE → Revise — Exploration before commitment.

Allows agents to submit a tentative action and receive feedback
(memory context, reflection hints) *without* the step counting.
The agent can then submit a revised action that actually gets graded.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from cognicore.core.types import ProposalFeedback
from cognicore.middleware.memory import Memory
from cognicore.middleware.reflection import ReflectionEngine


class ProposeReviseProtocol:
    """Manages the PROPOSE → Revise loop for a single step.

    Parameters
    ----------
    memory : Memory
        Memory instance for context retrieval.
    reflection : ReflectionEngine
        Reflection engine for hints.
    max_proposals : int
        Maximum number of proposals before forcing a commit.
    """

    def __init__(
        self,
        memory: Memory,
        reflection: ReflectionEngine,
        max_proposals: int = 1,
    ) -> None:
        self.memory = memory
        self.reflection = reflection
        self.max_proposals = max_proposals

        # Per-step state
        self._proposals_this_step: int = 0
        self._last_proposed_action: Optional[str] = None
        self._step_active: bool = False

        # Lifetime stats
        self._total_proposals: int = 0
        self._total_improvements: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin_step(self) -> None:
        """Call at the start of each step to reset proposal state."""
        self._proposals_this_step = 0
        self._last_proposed_action = None
        self._step_active = True

    def end_step(self) -> None:
        """Call at the end of each step."""
        self._step_active = False

    # ------------------------------------------------------------------
    # Propose
    # ------------------------------------------------------------------

    def propose(
        self,
        action: Dict[str, Any],
        group_value: str,
    ) -> ProposalFeedback:
        """Submit a tentative action and receive feedback.

        Parameters
        ----------
        action : dict
            The proposed action (e.g. ``{"classification": "SAFE"}``).
        group_value : str
            Group key for memory/reflection lookup.

        Returns
        -------
        ProposalFeedback
            Context and hints to help the agent revise.

        Raises
        ------
        RuntimeError
            If max proposals exceeded or step is not active.
        """
        if not self._step_active:
            raise RuntimeError("No active step. Call begin_step() first.")

        if self._proposals_this_step >= self.max_proposals:
            raise RuntimeError(
                f"Max proposals ({self.max_proposals}) reached. "
                "Submit via revise() or step() instead."
            )

        self._proposals_this_step += 1
        self._total_proposals += 1

        # Extract action string for reflection
        action_str = str(action.get("classification", action))
        self._last_proposed_action = action_str

        # Gather feedback
        memory_context = [
            e.to_dict() for e in self.memory.get_by_category(group_value, top_k=3)
        ]
        reflection_hint = self.reflection.get_hint(group_value)

        # Estimate confidence from past performance
        past = self.memory.get_by_category(group_value, top_k=10)
        if past:
            correct_count = sum(1 for e in past if getattr(e, "correct", False))
            confidence = correct_count / len(past)
        else:
            confidence = 0.5

        return ProposalFeedback(
            memory_context=memory_context,
            reflection_hint=reflection_hint,
            confidence_estimate=confidence,
            metadata={
                "proposals_this_step": self._proposals_this_step,
                "proposed_action": action_str,
            },
        )

    # ------------------------------------------------------------------
    # Check improvement
    # ------------------------------------------------------------------

    def check_improvement(self, final_action: str, eval_correct: bool) -> bool:
        """Check if the agent improved from proposal to final action.

        Returns True if the agent changed its action *and* got it correct.
        """
        if self._last_proposed_action is None:
            return False

        improved = final_action != self._last_proposed_action and eval_correct
        if improved:
            self._total_improvements += 1
        return improved

    @property
    def has_pending_proposal(self) -> bool:
        """True if a proposal was made but not yet committed."""
        return self._last_proposed_action is not None and self._step_active

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "total_proposals": self._total_proposals,
            "total_improvements": self._total_improvements,
            "improvement_rate": (
                self._total_improvements / self._total_proposals
                if self._total_proposals > 0
                else 0.0
            ),
        }
