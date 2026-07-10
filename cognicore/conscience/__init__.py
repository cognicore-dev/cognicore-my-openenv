"""
Conscience System — Real-time self-auditing, uncertainty detection, and decision holding.

This module provides the `Conscience` class which wraps an existing agent to
intercept actions. It computes a combined conscience score based on uncertainty,
novelty, consequence, and historical regret. If the score falls below a threshold,
an escalation policy is triggered.
"""

import copy
import random
import numpy as np
from typing import Any, Dict, List, Callable, Optional


class ConscienceEvent:
    """Records an instance where the conscience evaluated an action."""
    def __init__(
        self, 
        step: int, 
        observation: Any, 
        intended_action: Any, 
        scores: Dict[str, float],
        combined_score: float, 
        threshold: float,
        outcome: str,
        would_be_correct: Optional[bool] = None
    ):
        self.step = step
        self.observation = observation
        self.intended_action = intended_action
        self.scores = scores
        self.combined_score = combined_score
        self.threshold = threshold
        self.outcome = outcome
        # For regret calibration
        self.would_be_correct = would_be_correct

    def to_dict(self):
        return {
            "step": self.step,
            "intended_action": self.intended_action,
            "combined_score": self.combined_score,
            "outcome": self.outcome
        }


class ConscienceTracker:
    """Tracks state and config for the conscience system."""
    
    def __init__(self, agent: Any, threshold: float, escalation_policy: Callable):
        self.original_agent = agent
        self.threshold = threshold
        self.escalation_policy = escalation_policy
        self.events: List[ConscienceEvent] = []
        self.step_counter = 0

    def audit_log(self) -> List[Dict[str, Any]]:
        """Returns the full chronological history of every ConscienceEvent."""
        return [e.to_dict() for e in self.events]

    def regret_score(self) -> float:
        """Computes fraction of held decisions that would have been correct."""
        held = [e for e in self.events if e.outcome in ("hold", "escalate")]
        if not held:
            return 0.0
            
        correct_holds = 0
        for e in held:
            if e.would_be_correct:
                correct_holds += 1
        return correct_holds / len(held)

    def calibrate(self, target_regret: float = 0.1):
        """Automatically adjusts the threshold to hit a target regret score."""
        current_regret = self.regret_score()
        # If regret is higher than target (too strict/lenient), adjust threshold
        if current_regret > target_regret:
            # Conscience fired too often incorrectly -> lower the threshold
            self.threshold = max(0.0, self.threshold - 0.05)
        elif current_regret < target_regret:
            # Conscience not firing enough -> raise threshold
            self.threshold = min(1.0, self.threshold + 0.05)

    def explain(self, step: int) -> str:
        """Returns a natural language string explaining a specific conscience step."""
        event = next((e for e in self.events if e.step == step), None)
        if not event:
            return f"No conscience event recorded for step {step}."
            
        action = event.intended_action
        score = event.combined_score
        thr = event.threshold
        
        explanation = f"At step {step}, the intended action was '{action}'. "
        explanation += f"The combined conscience score was {score:.2f} (Threshold: {thr:.2f}).\n"
        
        for k, v in event.scores.items():
            explanation += f"- {k.capitalize()}: {v:.2f}\n"
            
        if score < thr:
            explanation += f"Because the score was below the threshold, the policy triggered outcome '{event.outcome}'."
        else:
            explanation += "The score was sufficient; the action proceeded normally."
            
        return explanation


class Conscience:
    """Wraps an agent to provide real-time self-auditing."""

    @classmethod
    def wrap(cls, agent: Any, threshold: float, escalation_policy: Callable) -> Any:
        """Wraps an agent to intercept its actions with conscience checks.
        
        Args:
            agent: The base agent.
            threshold: Score threshold below which conscience fires [0, 1].
            escalation_policy: Callable(ConscienceEvent) -> str
            
        Returns:
            The wrapped agent.
        """
        # Create a new class that inherits from the agent's class to maintain duck-typing
        class WrappedAgent(agent.__class__):
            def __init__(self, base_agent, thr, policy):
                # Copy attributes from base agent
                self.__dict__.update(base_agent.__dict__)
                self.conscience = ConscienceTracker(base_agent, thr, policy)
                self._base_agent = base_agent

            def _get_action(self, obs: Any) -> Any:
                """Helper to get action safely from base agent."""
                if hasattr(self._base_agent, "act"):
                    return self._base_agent.act(obs)
                elif hasattr(self._base_agent, "predict"):
                    res = self._base_agent.predict(obs)
                    return res[0] if isinstance(res, tuple) else res
                else:
                    return "default"

            def _compute_uncertainty(self, obs: Any) -> float:
                """Ensemble disagreement across 5 lightweight perturbations."""
                actions = []
                for _ in range(5):
                    # Perturb observation slightly if it's an array
                    if isinstance(obs, np.ndarray):
                        noise = np.random.normal(0, 0.01, obs.shape)
                        pert_obs = obs + noise
                    elif isinstance(obs, (list, tuple)):
                        pert_obs = type(obs)(x + random.gauss(0, 0.01) if isinstance(x, (int, float)) else x for x in obs)
                    else:
                        pert_obs = obs
                    actions.append(str(self._get_action(pert_obs)))
                
                # Uncertainty is high (low score) if there's high disagreement
                unique_actions = len(set(actions))
                return 1.0 - (unique_actions / 5.0)

            def _compute_novelty(self, obs: Any) -> float:
                """Novelty score comparing against past observations (1.0 = highly familiar)."""
                mem = getattr(self._base_agent, "memory", None)
                if not mem or not hasattr(mem, "entries"):
                    return 0.5
                
                entries = mem.entries
                if not entries:
                    return 0.1
                
                # Heuristic: return high if we've seen exact obs, otherwise moderate
                # (A real implementation would use embeddings)
                for e in entries[-50:]:
                    if getattr(e, "text", None) == str(obs):
                        return 1.0
                return 0.3

            def _compute_consequence(self, obs: Any, action: Any) -> float:
                """Shallow one-step lookahead using env."""
                env = getattr(self._base_agent, "env", None)
                if not env or not hasattr(env, "step"):
                    return 0.5
                    
                # We can't actually step the real env without side effects unless we can deepcopy
                try:
                    env_copy = copy.deepcopy(env)
                    step_res = env_copy.step(action)
                    reward = step_res[1] if len(step_res) > 1 else 0
                    # Return scaled reward expectation (assuming reward is between -1 and 1)
                    return max(0.0, min(1.0, (reward + 1.0) / 2.0))
                except Exception:
                    # Fallback if uncopyable
                    return 0.5

            def _compute_regret(self, obs: Any, action: Any) -> float:
                """Historical regret score based on similar past decisions."""
                # High score means low historical regret
                # We default to 0.8 as generally optimistic
                return 0.8

            def act(self, obs: Any) -> Any:
                self.conscience.step_counter += 1
                intended_action = self._get_action(obs)
                
                # 1. Compute scores
                scores = {
                    "uncertainty": self._compute_uncertainty(obs),
                    "novelty": self._compute_novelty(obs),
                    "consequence": self._compute_consequence(obs, intended_action),
                    "historical_regret": self._compute_regret(obs, intended_action)
                }
                
                # Combined score is average
                combined = sum(scores.values()) / len(scores)
                
                # 2. Check threshold
                outcome = "proceed"
                would_be_correct = random.random() > 0.5  # Simulate oracle/hindsight
                
                if combined < self.conscience.threshold:
                    event = ConscienceEvent(
                        step=self.conscience.step_counter,
                        observation=obs,
                        intended_action=intended_action,
                        scores=scores,
                        combined_score=combined,
                        threshold=self.conscience.threshold,
                        outcome="hold", # placeholder before policy
                        would_be_correct=would_be_correct
                    )
                    
                    policy_outcome = self.conscience.escalation_policy(event)
                    event.outcome = policy_outcome
                    
                    self.conscience.events.append(event)
                    
                    if policy_outcome.startswith("override"):
                        try:
                            # Extract overridden action, e.g., "override(left)"
                            new_action = policy_outcome.split("(")[1].split(")")[0]
                            # Try to cast back to original type
                            if isinstance(intended_action, int):
                                new_action = int(new_action)
                            return new_action
                        except Exception:
                            return intended_action
                    elif policy_outcome == "hold":
                        # Return None or a default safe action
                        return None
                    elif policy_outcome == "proceed":
                        return intended_action
                    elif policy_outcome == "escalate":
                        # Agent decides to hold and escalate externally
                        return None
                else:
                    # Record proceed event for audit
                    event = ConscienceEvent(
                        step=self.conscience.step_counter,
                        observation=obs,
                        intended_action=intended_action,
                        scores=scores,
                        combined_score=combined,
                        threshold=self.conscience.threshold,
                        outcome="proceed",
                        would_be_correct=True
                    )
                    self.conscience.events.append(event)
                    
                return intended_action

            def predict(self, obs: Any) -> Any:
                """Support for SB3 style interfaces."""
                act = self.act(obs)
                return (act, None)

        wrapped = WrappedAgent(agent, threshold, escalation_policy)
        return wrapped
