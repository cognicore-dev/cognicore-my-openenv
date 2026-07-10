import unittest
from cognicore.timetravel import TimeTraveler, TimelineNode, TimelineTree

class DummyEvent:
    def __init__(self, action, reward, obs="obs"):
        self.action = action
        self.reward = reward
        self.obs = obs

class DummyStore:
    def __init__(self, events):
        self.events = events
    def get_all(self):
        return self.events

class DummyMemory:
    def __init__(self):
        self.entries = [1, 2, 3, 4, 5]

class DummyAgent:
    def __init__(self):
        self.store = DummyStore([
            DummyEvent("up", 1.0, "o1"),
            DummyEvent("down", -1.0, "o2"),
            DummyEvent("left", 0.5, "o3")
        ])
        self.memory = DummyMemory()
        
    def act(self, obs):
        return "right"

class DummyEnv:
    def __init__(self):
        self.step_count = 0
        self.resets = 0
        
    def reset(self):
        self.step_count = 0
        self.resets += 1
        
    def step(self, action):
        self.step_count += 1
        return (f"new_obs_{self.step_count}", 2.0, False, False, {})

class TestTimeTraveler(unittest.TestCase):
    def setUp(self):
        self.agent = DummyAgent()
        self.env = DummyEnv()
        self.tt = TimeTraveler(self.agent, self.env)

    def test_initialization_loads_timeline(self):
        # 1. Test init loads timeline from agent store
        self.assertEqual(len(self.tt.original_timeline), 3)
        self.assertIn("original", self.tt.tree.branches)

    def test_node_creation(self):
        # 2. Test TimelineNode attributes
        node = TimelineNode(1, "obs", "act", 1.0)
        self.assertIsNotNone(node.node_id)
        self.assertEqual(node.reward, 1.0)

    def test_tree_add_branch(self):
        # 3. Test adding branch to tree
        nodes = [TimelineNode(1, "o", "a", 1.0)]
        self.tt.tree.add_branch("test_branch", nodes)
        self.assertIn("test_branch", self.tt.tree.branches)
        self.assertIn(nodes[0].node_id, self.tt.tree.nodes)

    def test_rewind_steps(self):
        # 4. Test rewinding standard steps
        success = self.tt.rewind(1) # Rewind 1 step (to step 2)
        self.assertTrue(success)
        self.assertEqual(self.tt.current_step, 2)
        self.assertEqual(self.env.resets, 1)
        self.assertEqual(self.env.step_count, 2)

    def test_rewind_agent_memory(self):
        # 5. Test rewinding truncates agent memory
        self.tt.rewind(2) # Rewind 2 steps (to step 1)
        self.assertEqual(len(self.agent.memory.entries), 1)

    def test_rewind_agent_store(self):
        # 6. Test rewinding truncates agent store
        self.tt.rewind(2)
        self.assertEqual(len(self.agent.store.events), 1)

    def test_rewind_too_far(self):
        # 7. Test rewinding past beginning clamped to 0
        self.tt.rewind(10)
        self.assertEqual(self.tt.current_step, 0)
        self.assertEqual(self.env.step_count, 0)

    def test_rewind_empty_timeline(self):
        # 8. Test rewinding fails if no timeline
        empty_agent = DummyAgent()
        empty_agent.store.events = []
        tt = TimeTraveler(empty_agent, self.env)
        self.assertFalse(tt.rewind(1))

    def test_branch_creation(self):
        # 9. Test branching creates new branch ID
        self.tt.rewind(1)
        branch_id = self.tt.branch("counter_act", 1)
        self.assertTrue(branch_id.startswith("branch_"))
        self.assertIn(branch_id, self.tt.tree.branches)

    def test_branch_node_linking(self):
        # 10. Test new branch links to parent correctly
        self.tt.rewind(1)
        parent_id = self.tt.original_timeline[1].node_id
        branch_id = self.tt.branch("act", 1)
        first_new_node_id = self.tt.tree.branches[branch_id][-1]
        first_new_node = self.tt.tree.nodes[first_new_node_id]
        self.assertEqual(first_new_node.parent_id, parent_id)

    def test_branch_plays_forward(self):
        # 11. Test branch plays forward correct number of steps
        self.tt.rewind(1) # current_step = 2
        branch_id = self.tt.branch("act", 3)
        self.assertEqual(len(self.tt.tree.branches[branch_id]), 5) # 2 prefix + 3 new

    def test_compare_identical_timelines(self):
        # 12. Test comparing identical timelines
        self.tt.tree.branches["copy"] = self.tt.tree.branches["original"]
        comp = self.tt.compare_timelines("original", "copy")
        self.assertEqual(comp["reward_diff"], 0.0)
        self.assertEqual(comp["divergence_step"], 3)

    def test_compare_divergent_timelines(self):
        # 13. Test comparing divergent timelines
        self.tt.rewind(2) # Current step = 1
        branch_id = self.tt.branch("alternative", 2)
        comp = self.tt.compare_timelines("original", branch_id)
        # Original: up(1.0), down(-1.0), left(0.5) -> sum = 0.5
        # Branch (starts at step 1): up(1.0) + 'alternative'(2.0) + 'right'(2.0) -> sum = 5.0
        self.assertEqual(comp["total_reward_1"], 0.5)
        self.assertEqual(comp["total_reward_2"], 5.0)
        self.assertEqual(comp["reward_diff"], 4.5)
        # Diverged at step 1
        self.assertEqual(comp["divergence_step"], 1)

    def test_compare_invalid_branch(self):
        # 14. Test compare raises error for unknown branch
        with self.assertRaises(ValueError):
            self.tt.compare_timelines("original", "unknown")

    def test_clone_state_fallback(self):
        # 15. Test clone state fallback
        class Uncopyable:
            def __deepcopy__(self, memo):
                raise TypeError("Cannot copy")
        obj = Uncopyable()
        cloned = self.tt._clone_state(obj)
        self.assertIs(cloned, obj)

if __name__ == "__main__":
    unittest.main()
