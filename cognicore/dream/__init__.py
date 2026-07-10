"""
DreamEngine System — Synthetic experience generation and adversarial dreaming.

This module provides the `DreamEngine` class, which uses an environment model
to generate synthetic experiences (dreams), adversarial edge-cases (nightmares),
and goal-directed hallucinatory rollouts.
"""

import random
from typing import Any, List, Optional


class DreamEvent:
    """Represents a single step in a synthetic experience."""
    def __init__(self, state: Any, action: Any, reward: float, next_state: Any, done: bool):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def to_dict(self):
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done
        }


class DreamEngine:
    """Generates synthetic experiences using an environment model."""
    
    def __init__(self, env_model: Any, action_space: Optional[List[Any]] = None):
        """Initializes the DreamEngine.
        
        Args:
            env_model: The environment model. Must have `.simulate(state, action)` 
                       returning `(next_state, reward, done)`.
            action_space: Optional list of all possible actions (required for nightmares and hallucination if agent isn't used).
        """
        self.env_model = env_model
        self.action_space = action_space or []

    def dream(self, initial_state: Any, agent: Any, steps: int = 50, random_exploration: float = 0.2) -> List[DreamEvent]:
        """Generates a synthetic rollout using the agent's policy with epsilon-greedy exploration.
        
        Args:
            initial_state: The starting state.
            agent: The agent to query for actions.
            steps: Max steps to simulate.
            random_exploration: Probability of picking a random action.
            
        Returns:
            List of DreamEvent objects.
        """
        events = []
        current_state = initial_state
        
        if not hasattr(self.env_model, "simulate"):
            return events

        for _ in range(steps):
            # 1. Select Action
            if random.random() < random_exploration and self.action_space:
                action = random.choice(self.action_space)
            else:
                if hasattr(agent, "act"):
                    action = agent.act(current_state)
                elif hasattr(agent, "predict"):
                    action = agent.predict(current_state)[0]
                else:
                    break
                    
            # 2. Simulate
            next_state, reward, done = self.env_model.simulate(current_state, action)
            
            # 3. Record
            event = DreamEvent(current_state, action, reward, next_state, done)
            events.append(event)
            
            if done:
                break
                
            current_state = next_state
            
        return events

    def nightmare(self, initial_state: Any, agent: Any = None, steps: int = 50) -> List[DreamEvent]:
        """Generates an adversarial rollout minimizing reward (finding edge cases).
        
        Args:
            initial_state: The starting state.
            agent: Unused directly for action selection, but could be used in extensions.
            steps: Max steps to simulate.
            
        Returns:
            List of DreamEvent representing the worst-case scenario.
        """
        events = []
        current_state = initial_state
        
        if not hasattr(self.env_model, "simulate") or not self.action_space:
            return events
            
        for _ in range(steps):
            best_worst_action = None
            min_reward = float('inf')
            best_next_state = None
            best_done = False
            
            # 1-step adversarial lookahead
            for action in self.action_space:
                next_state, reward, done = self.env_model.simulate(current_state, action)
                if reward < min_reward:
                    min_reward = reward
                    best_worst_action = action
                    best_next_state = next_state
                    best_done = done
                    
            if best_worst_action is None:
                break
                
            event = DreamEvent(current_state, best_worst_action, min_reward, best_next_state, best_done)
            events.append(event)
            
            if best_done:
                break
                
            current_state = best_next_state
            
        return events

    def hallucinate_goals(self, initial_state: Any, target_reward: float, steps: int = 10, search_budget: int = 100) -> List[Any]:
        """Finds a sequence of actions that achieves at least the target reward.
        
        Uses a random shooting / shallow DFS approach to find a valid trajectory.
        
        Args:
            initial_state: The starting state.
            target_reward: The cumulative reward target.
            steps: Max length of the sequence.
            search_budget: Max number of trajectories to sample.
            
        Returns:
            List of actions that achieve the goal, or empty list if none found.
        """
        if not hasattr(self.env_model, "simulate") or not self.action_space:
            return []
            
        for _ in range(search_budget):
            current_state = initial_state
            current_reward = 0.0
            trajectory = []
            
            for _ in range(steps):
                action = random.choice(self.action_space)
                next_state, reward, done = self.env_model.simulate(current_state, action)
                
                current_reward += reward
                trajectory.append(action)
                
                if current_reward >= target_reward:
                    return trajectory
                    
                if done:
                    break
                    
                current_state = next_state
                
        return []
