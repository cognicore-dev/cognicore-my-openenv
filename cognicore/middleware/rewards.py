"""
CogniCore Rewards — 8-component Structured Reward builder.

Computes each reward component from the evaluation result and
middleware state, then assembles the final ``StructuredReward``.
"""

from __future__ import annotations

import time
from typing import Optional

from cognicore.core.types import CogniCoreConfig, EvalResult, StructuredReward
from cognicore.middleware.memory import Memory


class RewardBuilder:
    """Builds a ``StructuredReward`` from evaluation + middleware state.

    Parameters
    ----------
    config : CogniCoreConfig
        Tuning parameters for each reward component.
    memory : Memory
        The memory instance (for consistency bonus).
    """

    def __init__(self, config: CogniCoreConfig, memory: Memory) -> None:
        self.config = config
        self.memory = memory

    def build(
        self,
        eval_result: EvalResult,
        *,
        streak_penalty: float = 0.0,
        followed_hint: bool = False,
        proposal_improved: bool = False,
        is_novel_group: bool = False,
        confidence: Optional[float] = None,
        step_start_time: Optional[float] = None,
    ) -> StructuredReward:
        """Assemble all 8 reward components.

        Parameters
        ----------
        eval_result : EvalResult
            Output from the environment grader.
        streak_penalty : float
            Penalty from the safety monitor (0.0 or negative).
        followed_hint : bool
            Whether the agent's action matched a reflection hint.
        proposal_improved : bool
            Whether the agent improved from propose → revise.
        is_novel_group : bool
            Whether this is the first time the agent sees this group.
        confidence : float or None
            Agent-reported confidence (0.0–1.0).
        step_start_time : float or None
            ``time.time()`` when the step started (for time decay).
        """
        cfg = self.config

        # 1. Base score — from the environment grader
        base_score = eval_result.base_score

        # 2. Memory bonus — consistency with past successes
        memory_bonus = 0.0
        if cfg.enable_memory and eval_result.correct and eval_result.category and eval_result.category.strip():
            past = self.memory.retrieve_successes(eval_result.category.strip(), top_k=3)
            if past:
                memory_bonus = cfg.memory_bonus_value

        # 3. Reflection bonus — agent followed the hint
        reflection_bonus = 0.0
        if cfg.enable_reflection and followed_hint and eval_result.correct:
            reflection_bonus = cfg.reflection_bonus_value

        # 4. Streak penalty — from safety monitor
        # (passed in directly, already computed)

        # 5. Propose bonus — agent improved from draft to final
        propose_bonus = 0.0
        if cfg.enable_propose_revise and proposal_improved:
            propose_bonus = cfg.propose_improvement_bonus

        # 6. Novelty bonus — correctly handled an unseen group
        novelty_bonus = 0.0
        if eval_result.correct and is_novel_group:
            novelty_bonus = cfg.novelty_bonus_value

        # 7. Confidence calibration
        confidence_cal = 0.0
        if confidence is not None:
            if eval_result.correct:
                # Reward high confidence + correct
                confidence_cal = confidence * cfg.confidence_bonus_scale
            else:
                # Penalize high confidence + wrong
                confidence_cal = -confidence * cfg.confidence_bonus_scale

        # 8. Time decay — slight penalty for slow responses
        time_decay = 0.0
        if step_start_time is not None:
            elapsed = time.time() - step_start_time
            if elapsed > cfg.time_decay_threshold_seconds:
                overtime = elapsed - cfg.time_decay_threshold_seconds
                time_decay = -(overtime * cfg.time_decay_rate)
                time_decay = max(time_decay, -0.05)  # cap the penalty

        return StructuredReward(
            base_score=base_score,
            memory_bonus=memory_bonus,
            reflection_bonus=reflection_bonus,
            streak_penalty=streak_penalty,
            propose_bonus=propose_bonus,
            novelty_bonus=novelty_bonus,
            confidence_cal=confidence_cal,
            time_decay=time_decay,
        )
