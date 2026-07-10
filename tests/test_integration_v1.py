import unittest
import numpy as np

# Verify zero import errors across all modules
from cognicore.passport import AgentPassport
from cognicore.dna import AgentDNA, Population
from cognicore.conscience import Conscience
from cognicore.civilization import Civilization, CivilizationServer, InsightReport
from cognicore.timetravel import TimeTraveler
from cognicore.oracle import Oracle
from cognicore.dream import DreamEngine, DreamEvent

# Existing modules (from 0.9.1 as mentioned)
# If they don't exist in our mock environment, that's fine, we are testing
# the new modules integration.

class MockEnv:
    def __init__(self):
        self.state = 0
    def step(self, action):
        self.state += 1
        return (f"obs_{self.state}", 1.0, False, False, {})
    def reset(self):
        self.state = 0
    def simulate(self, state, action):
        return (state + "_sim", 1.0, False)

class MockAgent:
    def __init__(self):
        self.env = MockEnv()
        self.config = {"lr": 0.01}
        # Fake store
        class Store:
            def get_all(self): return []
        self.store = Store()
        # Fake memory
        class Memory:
            def __init__(self): self.entries = []
        self.memory = Memory()
    
    def act(self, obs):
        return "default_action"


class TestV1Integration(unittest.TestCase):
    def test_full_stack_integration(self):
        """Tests that all modules can be instantiated and used together without conflict."""
        agent = MockAgent()
        
        # 1. Conscience
        wrapped_agent = Conscience.wrap(agent, threshold=0.1, escalation_policy=lambda e: "proceed")
        
        # 2. Oracle
        oracle = Oracle(wrapped_agent, wrapped_agent.env)
        
        # 3. DreamEngine
        dream_engine = DreamEngine(wrapped_agent.env, action_space=["a1", "a2"])
        
        # 4. TimeTraveler
        tt = TimeTraveler(wrapped_agent, wrapped_agent.env)
        
        # 5. AgentDNA
        dna = AgentDNA.extract(wrapped_agent, wrapped_agent.store)
        
        # 6. Civilization
        civ = Civilization(wrapped_agent)
        
        # 7. AgentPassport
        passport_path = AgentPassport.checkpoint(wrapped_agent, wrapped_agent.env, wrapped_agent.store, wrapped_agent.memory, None)
        restored = AgentPassport.restore(passport_path)
        
        # Verify
        self.assertEqual(restored["manifest"]["agent_class"], "WrappedAgent") # Should be original class or wrapped
        self.assertIn("risk_tolerance", dna)
        self.assertTrue(hasattr(civ, "join"))
        self.assertTrue(hasattr(tt, "rewind"))
        self.assertTrue(hasattr(oracle, "predict_outcome"))
        self.assertTrue(hasattr(dream_engine, "dream"))
        
        # Clean up passport file
        import os
        if os.path.exists(passport_path):
            os.remove(passport_path)

if __name__ == "__main__":
    unittest.main()
