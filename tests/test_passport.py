import unittest
import os
import shutil
import tempfile
import json
import zipfile
import numpy as np
from unittest.mock import patch

from cognicore.passport import AgentPassport

class DummyAgent:
    def __init__(self):
        self.config = {"learning_rate": 0.01, "mode": "explore"}
        self.reflection_state = {"insights": ["found key"]}
        self.reward_history = [1.0, -0.5, 2.0]

class DummyEnv:
    pass

class DummyEntry:
    def __init__(self, text, category, success):
        self.text = text
        self.category = category
        self.success = success
        self.action = "test_action"
        self.metadata = {"step": 1}

class DummyMemory:
    def __init__(self):
        self.entries = [
            DummyEntry("obs1", "cat1", True),
            DummyEntry("obs2", "cat2", False)
        ]

class DummyEvent:
    def __init__(self, seq, text, reward):
        self.seq = seq
        self.input_text = text
        self.output_text = "out"
        self.reward = reward

class DummyStore:
    def get_all(self):
        return [
            DummyEvent(1, "in1", 1.0),
            DummyEvent(2, "in2", -1.0)
        ]

class DummyImmune:
    def __init__(self):
        self.antibodies = ["ab1", "ab2"]
        self.weights = {"w1": np.array([1.0, 2.0]), "b1": np.array([0.5])}

class TestAgentPassport(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        self.agent = DummyAgent()
        self.env = DummyEnv()
        self.store = DummyStore()
        self.memory = DummyMemory()
        self.immune = DummyImmune()

    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def test_checkpoint_creates_file(self):
        # 1. Test creation
        path = AgentPassport.checkpoint(self.agent, self.env, self.store, self.memory, self.immune)
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith('.passport'))

    def test_checkpoint_contains_all_manifests(self):
        # 2. Test zip contents
        path = AgentPassport.checkpoint(self.agent, self.env, self.store, self.memory, self.immune)
        with zipfile.ZipFile(path, 'r') as zf:
            files = zf.namelist()
            self.assertIn('manifest.json', files)
            self.assertIn('memory.json', files)
            self.assertIn('replay.json', files)
            self.assertIn('immune.json', files)
            self.assertIn('weights.npz', files)
            self.assertIn('reflection.json', files)
            self.assertIn('reward.json', files)

    def test_restore_manifest(self):
        # 3. Test restore parsing manifest
        path = AgentPassport.checkpoint(self.agent, self.env, self.store, self.memory, self.immune)
        state = AgentPassport.restore(path)
        self.assertEqual(state['manifest']['agent_class'], 'DummyAgent')
        self.assertEqual(state['manifest']['config']['learning_rate'], 0.01)

    def test_restore_memory(self):
        # 4. Test restore parsing memory
        path = AgentPassport.checkpoint(self.agent, self.env, self.store, self.memory, self.immune)
        state = AgentPassport.restore(path)
        self.assertEqual(len(state['memory']), 2)
        self.assertEqual(state['memory'][0]['text'], 'obs1')

    def test_restore_replay(self):
        # 5. Test restore parsing replay
        path = AgentPassport.checkpoint(self.agent, self.env, self.store, self.memory, self.immune)
        state = AgentPassport.restore(path)
        self.assertEqual(len(state['replay']), 2)
        self.assertEqual(state['replay'][1]['input_text'], 'in2')

    def test_restore_immune_and_weights(self):
        # 6. Test restore parsing immune and weights (numpy arrays)
        path = AgentPassport.checkpoint(self.agent, self.env, self.store, self.memory, self.immune)
        state = AgentPassport.restore(path)
        self.assertEqual(len(state['immune']['antibodies']), 2)
        self.assertTrue('w1' in state['weights'])
        np.testing.assert_array_equal(state['weights']['w1'], np.array([1.0, 2.0]))

    def test_restore_reflection_and_reward(self):
        # 7. Test restore reflection and reward
        path = AgentPassport.checkpoint(self.agent, self.env, self.store, self.memory, self.immune)
        state = AgentPassport.restore(path)
        self.assertEqual(state['reflection']['insights'][0], 'found key')
        self.assertEqual(state['reward'], [1.0, -0.5, 2.0])

    def test_checkpoint_without_immune(self):
        # 8. Test checkpoint without immune state
        path = AgentPassport.checkpoint(self.agent, self.env, self.store, self.memory, None)
        state = AgentPassport.restore(path)
        self.assertEqual(state['immune'], {})
        self.assertEqual(state['weights'], {})

    def test_restore_file_not_found(self):
        # 9. Test file not found error
        with self.assertRaises(FileNotFoundError):
            AgentPassport.restore("non_existent.passport")

    def test_restore_corrupt_zip(self):
        # 10. Test corrupted zip detection
        bad_path = "corrupt.passport"
        with open(bad_path, "w") as f:
            f.write("not a zip file")
        with self.assertRaises(ValueError):
            AgentPassport.restore(bad_path)

    def test_diff_identical(self):
        # 11. Test diff output for identical checkpoints
        path = AgentPassport.checkpoint(self.agent, self.env, self.store, self.memory, self.immune)
        diff_str = AgentPassport.diff(path, path)
        self.assertIn("Memory entries: 2 -> 2 (+0)", diff_str)
        self.assertIn("Replay events : 2 -> 2 (+0)", diff_str)
        self.assertIn("Total reward  : 2.5 -> 2.5 (+0.0)", diff_str)
        self.assertIn("Configuration identical", diff_str)

    def test_diff_changed(self):
        # 12. Test diff output for changed checkpoints
        path_a = AgentPassport.checkpoint(self.agent, self.env, self.store, self.memory, self.immune)
        
        # Modify agent
        self.agent.reward_history.append(10.0)
        self.memory.entries.append(DummyEntry("obs3", "cat3", True))
        
        path_b = AgentPassport.checkpoint(self.agent, self.env, self.store, self.memory, self.immune)
        diff_str = AgentPassport.diff(path_a, path_b)
        self.assertIn("Memory entries: 2 -> 3 (+1)", diff_str)
        self.assertIn("Total reward  : 2.5 -> 12.5 (+10.0)", diff_str)

    def test_deploy_local(self):
        # 13. Test local:// deploy
        path = AgentPassport.checkpoint(self.agent, self.env, self.store, self.memory, self.immune)
        target = "local://deployed_agent.passport"
        success = AgentPassport.deploy(path, target)
        self.assertTrue(success)
        self.assertTrue(os.path.exists("deployed_agent.passport"))

    def test_deploy_wasm(self):
        # 14. Test wasm:// deploy (stub)
        path = AgentPassport.checkpoint(self.agent, self.env, self.store, self.memory, self.immune)
        target = "wasm://browser"
        success = AgentPassport.deploy(path, target)
        self.assertTrue(success)

    @patch("subprocess.run")
    def test_deploy_ssh(self, mock_run):
        # 15. Test ssh:// deploy routing
        mock_run.return_value = None
        path = AgentPassport.checkpoint(self.agent, self.env, self.store, self.memory, self.immune)
        target = "ssh://user@server:/tmp/agent.passport"
        success = AgentPassport.deploy(path, target)
        self.assertTrue(success)
        mock_run.assert_called_once()

if __name__ == "__main__":
    unittest.main()
