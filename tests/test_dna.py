import unittest
import copy
from cognicore.dna import AgentDNA, Population

class DummyEvent:
    def __init__(self, action, reward):
        self.output_text = action
        self.reward = reward

class DummyStore:
    def __init__(self, events):
        self.events = events
    def get_all(self):
        return self.events

class DummyAgent:
    def __init__(self, store_events=None):
        self.store = DummyStore(store_events or [])
        self.id = id(self)

def dummy_fitness(agent, env):
    # Fitness based on how many 'positive' actions are in store
    events = agent.store.get_all()
    if not events: return 0.0
    return sum(1 for e in events if e.reward > 0)

class TestAgentDNA(unittest.TestCase):
    def setUp(self):
        # Setup agent with specific behavioral pattern
        self.events_a = [
            DummyEvent("move_up", 1.0),
            DummyEvent("move_down", -1.0),
            DummyEvent("recall_memory", 0.0),
            DummyEvent("reflect_action", 0.5),
            DummyEvent("move_up", 1.0)
        ]
        self.agent_a = DummyAgent(self.events_a)

    def test_extract_basic(self):
        # 1. Test basic DNA extraction format
        dna = AgentDNA.extract(self.agent_a, self.agent_a.store)
        self.assertIn("risk_tolerance", dna)
        self.assertIn("memory_reliance", dna)
        self.assertIn("value", dna["risk_tolerance"])
        self.assertIn("confidence", dna["risk_tolerance"])
        self.assertIn("evidence", dna["risk_tolerance"])

    def test_extract_memory_reliance(self):
        # 2. Test memory_reliance calculation
        dna = AgentDNA.extract(self.agent_a, self.agent_a.store)
        self.assertAlmostEqual(dna["memory_reliance"]["value"], 0.2) # 1 / 5

    def test_extract_reflection_depth(self):
        # 3. Test reflection_depth calculation
        dna = AgentDNA.extract(self.agent_a, self.agent_a.store)
        self.assertAlmostEqual(dna["reflection_depth"]["value"], 0.4) # 1 / 2.5

    def test_extract_exploration_rate(self):
        # 4. Test exploration_rate calculation
        dna = AgentDNA.extract(self.agent_a, self.agent_a.store)
        self.assertAlmostEqual(dna["exploration_rate"]["value"], 0.8) # 4 unique / 5 total

    def test_extract_recovery_speed(self):
        # 5. Test recovery speed calculation
        dna = AgentDNA.extract(self.agent_a, self.agent_a.store)
        self.assertAlmostEqual(dna["recovery_speed"]["value"], 0.0) # Neg at idx 1, next is 0.0 (not positive)

    def test_extract_consistency(self):
        # 6. Test consistency inverse of exploration
        dna = AgentDNA.extract(self.agent_a, self.agent_a.store)
        self.assertAlmostEqual(dna["consistency"]["value"], 0.2)

    def test_extract_empty_store(self):
        # 7. Test fallback when no history
        empty_agent = DummyAgent()
        dna = AgentDNA.extract(empty_agent, empty_agent.store)
        for trait in AgentDNA.TRAITS:
            self.assertEqual(dna[trait]["value"], 0.5)

    def test_crossover_parent_a(self):
        # 8. Test crossover strict inheritance from parent A
        dna_a = {"risk_tolerance": {"value": 0.8}}
        dna_b = {"risk_tolerance": {"value": 0.2}}
        child = AgentDNA.crossover(dna_a, dna_b, {"risk_tolerance": "parent_a"}, 0.0)
        self.assertEqual(child["risk_tolerance"]["value"], 0.8)

    def test_crossover_parent_b(self):
        # 9. Test crossover strict inheritance from parent B
        dna_a = {"risk_tolerance": {"value": 0.8}}
        dna_b = {"risk_tolerance": {"value": 0.2}}
        child = AgentDNA.crossover(dna_a, dna_b, {"risk_tolerance": "parent_b"}, 0.0)
        self.assertEqual(child["risk_tolerance"]["value"], 0.2)

    def test_crossover_blend(self):
        # 10. Test crossover blend inheritance
        dna_a = {"risk_tolerance": {"value": 0.8}}
        dna_b = {"risk_tolerance": {"value": 0.2}}
        child = AgentDNA.crossover(dna_a, dna_b, {"risk_tolerance": "blend"}, 0.0)
        self.assertAlmostEqual(child["risk_tolerance"]["value"], 0.5)

    def test_crossover_mutate_bounds(self):
        # 11. Test mutation stays within [0,1] bounds
        dna_a = {"risk_tolerance": {"value": 0.95}}
        dna_b = {"risk_tolerance": {"value": 0.95}}
        
        # Test many times to ensure bounds
        for _ in range(50):
            child = AgentDNA.crossover(dna_a, dna_b, {"risk_tolerance": "blend"}, 1.0)
            val = child["risk_tolerance"]["value"]
            self.assertTrue(0.0 <= val <= 1.0)

    def test_evolve_tournament_selection(self):
        # 12. Test evolution loop with tournament selection
        agents = [DummyAgent([DummyEvent("a", 1.0)]*i) for i in range(5)]
        pop = AgentDNA.evolve(agents, dummy_fitness, None, generations=3, population_size=5, selection_strategy="tournament")
        fittest = pop.fittest()
        # Should converge towards the agent with most events (i=4)
        self.assertGreaterEqual(len(fittest.store.get_all()), 2)

    def test_evolve_elite_selection(self):
        # 13. Test evolution loop with elite selection
        agents = [DummyAgent([DummyEvent("a", 1.0)]*i) for i in range(10)]
        pop = AgentDNA.evolve(agents, dummy_fitness, None, generations=2, population_size=10, selection_strategy="elite")
        fittest = pop.fittest()
        self.assertGreaterEqual(len(fittest.store.get_all()), 8)

    def test_evolve_roulette_selection(self):
        # 14. Test evolution loop with roulette selection
        agents = [DummyAgent([DummyEvent("a", 1.0)]*i) for i in range(5)]
        pop = AgentDNA.evolve(agents, dummy_fitness, None, generations=3, population_size=5, selection_strategy="roulette")
        self.assertEqual(len(pop.generations), 3)

    def test_diversity_score(self):
        # 15. Test diversity score is valid
        agents = [DummyAgent([DummyEvent("a", 1.0)]) for _ in range(5)]
        pop = AgentDNA.evolve(agents, dummy_fitness, None, generations=2, population_size=5)
        ds = pop.diversity_score()
        self.assertTrue(0.0 <= ds <= 1.0)

    def test_generation_report(self):
        # 16. Test generation report accuracy
        agents = [DummyAgent([DummyEvent("a", 1.0)])]
        pop = AgentDNA.evolve(agents, dummy_fitness, None, generations=2, population_size=5)
        rep = pop.generation_report(0)
        self.assertEqual(rep["population_size"], 5)
        self.assertEqual(rep["generation"], 0)

    def test_visualize_svg(self):
        # 17. Test visualization returns valid HTML/SVG string
        dna = AgentDNA.extract(self.agent_a, self.agent_a.store)
        svg = AgentDNA.visualize(dna)
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))
        self.assertIn("risk_tolerance", svg)

    def test_visualize_empty(self):
        # 18. Test visualize handles empty DNA
        svg = AgentDNA.visualize({})
        self.assertIn("No DNA traits found", svg)

    def test_compatibility_score(self):
        # 19. Test compatibility metric
        dna_a = {"risk_tolerance": {"value": 0.8}, "consistency": {"value": 0.5}}
        dna_b = {"risk_tolerance": {"value": 0.2}, "consistency": {"value": 0.5}}
        score = AgentDNA.compatibility(dna_a, dna_b)
        self.assertTrue(0.0 <= score <= 1.0)
        # diffs: risk=0.6, const=0.0
        # dist: abs(0.6-0.3)=0.3, abs(0.0-0.3)=0.3. avg_dist = 0.3. score = 0.7
        self.assertAlmostEqual(score, 0.7)

    def test_compatibility_empty(self):
        # 20. Test compatibility with no overlapping traits
        dna_a = {"trait_a": {"value": 0.8}}
        dna_b = {"trait_b": {"value": 0.2}}
        score = AgentDNA.compatibility(dna_a, dna_b)
        self.assertEqual(score, 0.5)

if __name__ == "__main__":
    unittest.main()
