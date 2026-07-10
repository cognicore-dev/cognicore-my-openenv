"""
Oracle System — Predictive simulation and counterfactual analysis.

This module provides the `Oracle` class which wraps an agent and an environment
model, granting the agent the ability to simulate outcomes, evaluate counterfactuals,
and select best actions based on rollouts.
"""

from typing import Any, Dict, List, Optional, Tuple


class Oracle:
    """Predictive simulation engine for agents."""
    
    def __init__(self, agent: Any, env_model: Any):
        """Initializes the Oracle.
        
        Args:
            agent: The base agent to evaluate policies.
            env_model: The environment model. Must have `.simulate(state, action)` 
                       returning `(next_state, reward, done)`.
        """
        self.agent = agent
        self.env_model = env_model
        
    def predict_outcome(self, obs: Any, action: Any, steps: int = 10, rollouts: int = 5) -> Dict[str, float]:
        """Predicts the outcome of taking an action from a given state via Monte Carlo rollouts.
        
        Args:
            obs: The initial observation/state.
            action: The immediate action to take.
            steps: Max steps per rollout.
            rollouts: Number of Monte Carlo rollouts.
            
        Returns:
            Dict containing expected_reward, success_probability, and risk_score.
        """
        total_rewards = []
        successes = 0
        failures = 0
        
        for _ in range(rollouts):
            current_obs = obs
            current_action = action
            rollout_reward = 0.0
            done = False
            
            for step_idx in range(steps):
                if not hasattr(self.env_model, "simulate"):
                    break
                    
                next_obs, reward, done = self.env_model.simulate(current_obs, current_action)
                rollout_reward += reward
                current_obs = next_obs
                
                if done:
                    break
                    
                # Ask agent for next action
                if hasattr(self.agent, "act"):
                    current_action = self.agent.act(current_obs)
                elif hasattr(self.agent, "predict"):
                    current_action = self.agent.predict(current_obs)[0]
                else:
                    break
                    
            total_rewards.append(rollout_reward)
            if rollout_reward > 0:
                successes += 1
            elif rollout_reward < 0:
                failures += 1
                
        expected = sum(total_rewards) / max(1, len(total_rewards))
        prob = successes / max(1, rollouts)
        risk = failures / max(1, rollouts)
        
        return {
            "expected_reward": expected,
            "success_probability": prob,
            "risk_score": risk
        }

    def best_action(self, obs: Any, candidate_actions: List[Any], steps: int = 5, rollouts: int = 3) -> Any:
        """Evaluates multiple actions and returns the best one based on expected reward.
        
        Args:
            obs: The current observation/state.
            candidate_actions: List of actions to evaluate.
            steps: Steps to look ahead.
            rollouts: Number of rollouts per action.
            
        Returns:
            The action with the highest expected reward.
        """
        if not candidate_actions:
            return None
            
        best_act = candidate_actions[0]
        best_score = float('-inf')
        
        for action in candidate_actions:
            outcome = self.predict_outcome(obs, action, steps=steps, rollouts=rollouts)
            score = outcome["expected_reward"]
            if score > best_score:
                best_score = score
                best_act = action
                
        return best_act

    def what_if(self, obs: Any, sequence_of_actions: List[Any]) -> Dict[str, Any]:
        """Simulates a forced sequence of actions to see where it leads.
        
        Args:
            obs: The initial observation/state.
            sequence_of_actions: The specific sequence of actions to take.
            
        Returns:
            Dict containing final_state, total_reward, and completed_steps.
        """
        current_obs = obs
        total_reward = 0.0
        steps = 0
        
        for action in sequence_of_actions:
            if not hasattr(self.env_model, "simulate"):
                break
                
            next_obs, reward, done = self.env_model.simulate(current_obs, action)
            total_reward += reward
            current_obs = next_obs
            steps += 1
            
            if done:
                break
                
        return {
            "final_state": current_obs,
            "total_reward": total_reward,
            "completed_steps": steps
        }

    def explain_prediction(self, obs: Any, action: Any, steps: int = 3) -> str:
        """Provides a natural language breakdown of the simulation path for an action.
        
        Args:
            obs: The initial state.
            action: The immediate action.
            steps: Steps to trace.
            
        Returns:
            str: Explanation of the rollout.
        """
        explanation = f"Oracle Simulation for action '{action}' from initial state:\n"
        
        current_obs = obs
        current_action = action
        total_reward = 0.0
        
        for step_idx in range(steps):
            if not hasattr(self.env_model, "simulate"):
                explanation += "Error: Environment model lacks 'simulate' method.\n"
                break
                
            next_obs, reward, done = self.env_model.simulate(current_obs, current_action)
            total_reward += reward
            
            explanation += f"Step {step_idx + 1}: Taking action '{current_action}' yields reward {reward:.2f}.\n"
            
            if done:
                explanation += "Simulation terminated early (Terminal state reached).\n"
                break
                
            current_obs = next_obs
            if hasattr(self.agent, "act"):
                current_action = self.agent.act(current_obs)
            elif hasattr(self.agent, "predict"):
                current_action = self.agent.predict(current_obs)[0]
            else:
                current_action = "default"
                
        explanation += f"Total Expected Reward over {steps} steps: {total_reward:.2f}"
        return explanation
