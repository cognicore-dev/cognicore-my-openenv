import abc
from typing import Dict, Any, Tuple

class BaseEnvironment(abc.ABC):
    """Abstract base class for benchmark environments."""
    
    @abc.abstractmethod
    def reset(self, task: Dict[str, Any]) -> str:
        """Reset the environment with a new task.
        
        Args:
            task: The task definition dictionary.
            
        Returns:
            The initial state/prompt for the agent.
        """
        pass
        
    @abc.abstractmethod
    def step(self, action: str) -> Tuple[str, bool, bool, Dict[str, Any]]:
        """Take a step in the environment by applying a patch/fix.
        
        Args:
            action: The proposed fix or code from the agent.
            
        Returns:
            Tuple of (observation, success, failure, info).
            - observation: String output (e.g., error message or success confirmation).
            - success: True if the task is solved.
            - failure: True if the action is invalid or causes a fatal error.
            - info: Additional metadata.
        """
        pass
