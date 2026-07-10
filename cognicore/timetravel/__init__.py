"""
TimeTraveler System — State rewinding, counterfactual branching, and timeline comparison.

This module provides the ability to rewind an agent/environment state,
explore alternative actions, and compare the resulting timelines.
"""

import copy
import uuid
from typing import Any, Dict, List, Optional


class TimelineNode:
    def __init__(self, step: int, obs: Any, action: Any, reward: float, parent_id: Optional[str] = None):
        self.node_id = str(uuid.uuid4())
        self.parent_id = parent_id
        self.step = step
        self.obs = obs
        self.action = action
        self.reward = reward
        self.children = []


class TimelineTree:
    """Tracks state identifiers and relationships between divergent timelines."""
    
    def __init__(self):
        self.nodes = {}
        self.branches = {} # branch_id -> list of node_ids
        
    def add_branch(self, branch_id: str, nodes: List[TimelineNode]):
        self.branches[branch_id] = [n.node_id for n in nodes]
        for n in nodes:
            self.nodes[n.node_id] = n


class TimeTraveler:
    """Enables rewinding and counterfactual branching for agents."""
    
    def __init__(self, agent: Any, env: Any):
        self.agent = agent
        self.env = env
        self.tree = TimelineTree()
        self.original_timeline = []
        self._load_original_timeline()

    def _load_original_timeline(self):
        """Loads the baseline timeline from the agent's EventStore."""
        store = getattr(self.agent, "store", None)
        if not store or not hasattr(store, "get_all"):
            return
            
        events = store.get_all()
        parent_id = None
        for i, e in enumerate(events):
            obs = getattr(e, "input_text", getattr(e, "obs", None))
            action = getattr(e, "output_text", getattr(e, "action", None))
            reward = getattr(e, "reward", 0.0)
            
            node = TimelineNode(i, obs, action, reward, parent_id)
            self.original_timeline.append(node)
            self.tree.nodes[node.node_id] = node
            parent_id = node.node_id
            
        self.tree.branches["original"] = [n.node_id for n in self.original_timeline]

    def _clone_state(self, target: Any) -> Any:
        """Deep copies an object, with fallbacks for uncopyable components."""
        try:
            return copy.deepcopy(target)
        except Exception:
            return target

    def rewind(self, steps_back: int) -> bool:
        """Rewinds the environment and agent state.
        
        Args:
            steps_back: Number of steps to rewind.
            
        Returns:
            bool: True if rewound successfully.
        """
        if not self.original_timeline:
            return False
            
        target_step = max(0, len(self.original_timeline) - steps_back)
        
        # 1. Rewind Environment
        if hasattr(self.env, "load_state") and hasattr(self.env, "save_state"):
            # Requires the environment to have proactively saved states or we assume 
            # we saved them (which we didn't). 
            pass
            
        # Standard approach: Reset env and replay actions up to target_step
        if hasattr(self.env, "reset"):
            self.env.reset()
            for i in range(target_step):
                action = self.original_timeline[i].action
                if hasattr(self.env, "step"):
                    self.env.step(action)
                    
        # 2. Rewind Agent Memory/Store
        store = getattr(self.agent, "store", None)
        if store and hasattr(store, "events"):
            # Truncate events
            store.events = store.events[:target_step]
            
        memory = getattr(self.agent, "memory", None)
        if memory and hasattr(memory, "entries"):
            # Heuristically truncate memory if we can't map exactly
            memory.entries = memory.entries[:target_step]
            
        self.current_step = target_step
        return True

    def branch(self, action: Any, steps_forward: int) -> str:
        """Takes an alternative action and plays forward.
        
        Args:
            action: The counterfactual action to take.
            steps_forward: Number of subsequent steps to simulate.
            
        Returns:
            str: The new branch ID.
        """
        branch_id = f"branch_{uuid.uuid4().hex[:8]}"
        
        # Include prefix nodes up to the current step
        prefix = self.original_timeline[:self.current_step]
        nodes = list(prefix)
        
        # Link to parent node if possible
        parent_id = None
        if self.current_step > 0 and self.current_step <= len(self.original_timeline):
            parent_id = self.original_timeline[self.current_step - 1].node_id
            
        # 1. Take the counterfactual action
        obs = None
        reward = 0.0
        if hasattr(self.env, "step"):
            res = self.env.step(action)
            obs = res[0]
            reward = res[1] if len(res) > 1 else 0.0
            
        node = TimelineNode(self.current_step, obs, action, reward, parent_id)
        nodes.append(node)
        self.tree.nodes[node.node_id] = node
        parent_id = node.node_id
        
        # 2. Play forward
        for i in range(1, steps_forward):
            # Ask agent for next action based on new obs
            if hasattr(self.agent, "act"):
                next_action = self.agent.act(obs)
            elif hasattr(self.agent, "predict"):
                next_action = self.agent.predict(obs)[0]
            else:
                next_action = "default"
                
            if hasattr(self.env, "step"):
                res = self.env.step(next_action)
                obs = res[0]
                reward = res[1] if len(res) > 1 else 0.0
                
            node = TimelineNode(self.current_step + i, obs, next_action, reward, parent_id)
            nodes.append(node)
            self.tree.nodes[node.node_id] = node
            parent_id = node.node_id
            
        self.tree.add_branch(branch_id, nodes)
        return branch_id

    def compare_timelines(self, branch_id_1: str, branch_id_2: str) -> Dict[str, Any]:
        """Compares two timelines and returns the differences.
        
        Args:
            branch_id_1: ID of first branch (e.g. 'original').
            branch_id_2: ID of second branch.
            
        Returns:
            Dict: Comparison results (reward diff, length diff, divergence point).
        """
        if branch_id_1 not in self.tree.branches or branch_id_2 not in self.tree.branches:
            raise ValueError("Branch ID not found in TimelineTree.")
            
        nodes_1 = [self.tree.nodes[nid] for nid in self.tree.branches[branch_id_1]]
        nodes_2 = [self.tree.nodes[nid] for nid in self.tree.branches[branch_id_2]]
        
        reward_1 = sum(n.reward for n in nodes_1)
        reward_2 = sum(n.reward for n in nodes_2)
        
        # Find divergence step
        divergence_step = 0
        min_len = min(len(nodes_1), len(nodes_2))
        for i in range(min_len):
            if nodes_1[i].action != nodes_2[i].action or nodes_1[i].obs != nodes_2[i].obs:
                divergence_step = nodes_1[i].step
                break
        else:
            divergence_step = min_len
            
        return {
            "reward_diff": reward_2 - reward_1,
            "total_reward_1": reward_1,
            "total_reward_2": reward_2,
            "length_1": len(nodes_1),
            "length_2": len(nodes_2),
            "divergence_step": divergence_step
        }
