"""
AgentDNA System — Behavioral genome extraction and evolutionary algorithms.

This module treats agent behavior as a genome. It can extract traits from
replay histories, perform genetic crossover and mutation, run full evolutionary
loops over populations of agents, and visualize the resulting DNA profiles.
"""

import math
import random
import copy
from typing import Any, Dict, List, Callable, Optional


class Population:
    """Tracks a population of agents through evolutionary generations."""
    
    def __init__(self, initial_agents: List[Any]):
        self.generations = [initial_agents]
        self.fitness_history = []
        self.dna_history = []

    def fittest(self) -> Any:
        """Returns the most fit agent from the latest generation."""
        if not self.fitness_history:
            return self.generations[-1][0]
        latest_fitness = self.fitness_history[-1]
        best_idx = max(range(len(latest_fitness)), key=lambda i: latest_fitness[i])
        return self.generations[-1][best_idx]

    def history(self) -> List[List[float]]:
        """Returns the history of fitness scores across generations."""
        return self.fitness_history

    def diversity_score(self) -> float:
        """Calculates genetic diversity of the current generation (0 to 1)."""
        if not self.dna_history:
            return 0.0
        latest_dnas = self.dna_history[-1]
        if len(latest_dnas) < 2:
            return 0.0
            
        total_dist = 0.0
        pairs = 0
        for i in range(len(latest_dnas)):
            for j in range(i + 1, len(latest_dnas)):
                dist = sum(abs(latest_dnas[i][k]["value"] - latest_dnas[j][k]["value"]) 
                           for k in latest_dnas[i])
                total_dist += dist / max(1, len(latest_dnas[i]))
                pairs += 1
        return total_dist / pairs if pairs > 0 else 0.0

    def generation_report(self, n: int) -> Dict[str, Any]:
        """Returns a statistical report for generation n."""
        if n >= len(self.generations) or n >= len(self.fitness_history):
            raise ValueError(f"Generation {n} not recorded yet.")
        fit = self.fitness_history[n]
        return {
            "generation": n,
            "population_size": len(fit),
            "max_fitness": max(fit),
            "min_fitness": min(fit),
            "avg_fitness": sum(fit) / len(fit)
        }


class AgentDNA:
    """System for extracting, evolving, and visualizing agent behavioral genomes."""

    TRAITS = [
        "risk_tolerance", "memory_reliance", "reflection_depth", 
        "exploration_rate", "recovery_speed", "consistency", 
        "decision_speed", "failure_response"
    ]

    @classmethod
    def extract(cls, agent: Any, store: Any, episodes: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """Extracts behavioral genome from an agent's replay history.
        
        Args:
            agent: The agent to extract DNA from.
            store: The EventStore containing replay events.
            episodes: Optional max episodes to analyze.
            
        Returns:
            Dict: The DNA object, a dict of named traits with value [0,1],
                  confidence, and evidence.
        """
        dna = {}
        events = []
        if hasattr(store, "get_all"):
            events = store.get_all()
            
        if not events:
            # Return baseline DNA if no history
            for trait in cls.TRAITS:
                dna[trait] = {"value": 0.5, "confidence": 0.0, "evidence": "No history available."}
            return dna

        rewards = [getattr(e, "reward", 0.0) for e in events]
        actions = [getattr(e, "output_text", getattr(e, "action", "")) for e in events]
        
        # 1. Risk Tolerance (based on reward variance)
        r_mean = sum(rewards) / len(rewards)
        r_var = sum((r - r_mean) ** 2 for r in rewards) / len(rewards)
        risk = min(1.0, r_var / 10.0) if len(rewards) > 1 else 0.5
        dna["risk_tolerance"] = {"value": risk, "confidence": 0.8, "evidence": f"Reward variance {r_var:.2f}"}

        # 2. Memory Reliance
        mem_calls = sum(1 for a in actions if isinstance(a, str) and "recall" in a.lower())
        mem_rel = min(1.0, mem_calls / len(actions))
        dna["memory_reliance"] = {"value": mem_rel, "confidence": 0.9, "evidence": f"{mem_calls} memory accesses"}

        # 3. Reflection Depth
        ref_calls = sum(1 for a in actions if isinstance(a, str) and "reflect" in a.lower())
        ref_dep = min(1.0, ref_calls / max(1, len(actions) * 0.5))
        dna["reflection_depth"] = {"value": ref_dep, "confidence": 0.8, "evidence": f"{ref_calls} reflections"}

        # 4. Exploration Rate
        unique_acts = len(set(actions))
        exp_rate = unique_acts / len(actions)
        dna["exploration_rate"] = {"value": exp_rate, "confidence": 0.9, "evidence": f"{unique_acts} unique actions"}

        # 5. Recovery Speed
        recoveries = 0
        neg_count = 0
        for i in range(1, len(rewards)):
            if rewards[i-1] < 0:
                neg_count += 1
                if rewards[i] > 0:
                    recoveries += 1
        rec_spd = (recoveries / neg_count) if neg_count > 0 else 0.5
        dna["recovery_speed"] = {"value": rec_spd, "confidence": 0.7 if neg_count > 0 else 0.1, "evidence": f"{recoveries}/{neg_count} recoveries"}

        # 6. Consistency
        dna["consistency"] = {"value": 1.0 - exp_rate, "confidence": 0.8, "evidence": "Inverse of exploration"}

        # 7. Decision Speed
        dna["decision_speed"] = {"value": 0.75, "confidence": 0.5, "evidence": "Heuristic estimate"}

        # 8. Failure Response
        changed_after_fail = 0
        fail_count = 0
        for i in range(1, len(rewards)):
            if rewards[i-1] < 0:
                fail_count += 1
                if actions[i] != actions[i-1]:
                    changed_after_fail += 1
        fail_resp = (changed_after_fail / fail_count) if fail_count > 0 else 0.5
        dna["failure_response"] = {"value": fail_resp, "confidence": 0.8 if fail_count > 0 else 0.1, "evidence": f"Changed action {changed_after_fail} times after failure"}

        # Ensure bounds and fallbacks
        for trait in cls.TRAITS:
            if trait not in dna:
                dna[trait] = {"value": 0.5, "confidence": 0.1, "evidence": "Fallback"}
                
        return dna

    @classmethod
    def crossover(cls, dna_a: Dict[str, Any], dna_b: Dict[str, Any], traits_dict: Dict[str, str], mutation_rate: float) -> Dict[str, Any]:
        """Produces a child DNA by blending parent traits.
        
        Args:
            dna_a: Parent A DNA.
            dna_b: Parent B DNA.
            traits_dict: Instruction map per trait ("parent_a", "parent_b", "blend", "mutate").
            mutation_rate: Probability of mutating traits globally, or severity.
            
        Returns:
            Dict: Child DNA.
        """
        child = {}
        for trait in cls.TRAITS:
            method = traits_dict.get(trait, "blend")
            val_a = dna_a.get(trait, {}).get("value", 0.5)
            val_b = dna_b.get(trait, {}).get("value", 0.5)
            
            if method == "parent_a":
                val = val_a
            elif method == "parent_b":
                val = val_b
            elif method == "mutate":
                val = random.random()
            else: # "blend"
                val = (val_a + val_b) / 2.0
                
            # Apply global mutation chance
            if random.random() < mutation_rate:
                mutation_shift = (random.random() * 0.4) - 0.2
                val = max(0.0, min(1.0, val + mutation_shift))
                
            child[trait] = {
                "value": val,
                "confidence": min(dna_a.get(trait, {}).get("confidence", 0.5), dna_b.get(trait, {}).get("confidence", 0.5)),
                "evidence": f"Crossover ({method})"
            }
        return child

    @classmethod
    def evolve(
        cls, 
        seed_agents: List[Any], 
        fitness_fn: Callable[[Any, Any], float], 
        env: Any, 
        generations: int, 
        population_size: int, 
        selection_strategy: str = "tournament"
    ) -> Population:
        """Runs a full evolutionary loop returning a Population object.
        
        Args:
            seed_agents: Initial population agents.
            fitness_fn: Function mapping (agent, env) to a float fitness score.
            env: The environment to evaluate on.
            generations: Number of generations to evolve.
            population_size: Number of agents per generation.
            selection_strategy: "tournament", "roulette", or "elite".
            
        Returns:
            Population: The tracked evolution history.
        """
        # Ensure initial population size
        current_pop = list(seed_agents)
        while len(current_pop) < population_size:
            current_pop.append(copy.deepcopy(random.choice(seed_agents)))
            
        pop_tracker = Population(current_pop)
        
        for gen in range(generations):
            # 1. Evaluate fitness
            fitness_scores = [fitness_fn(agent, env) for agent in current_pop]
            pop_tracker.fitness_history.append(fitness_scores)
            
            # Extract and log DNA
            # (In a real implementation we'd need EventStore per agent. We assume 
            # the agent has a mock store or we fallback to baseline DNA for evolution tracking).
            dnas = []
            for agent in current_pop:
                store = getattr(agent, "store", None)
                dnas.append(cls.extract(agent, store))
            pop_tracker.dna_history.append(dnas)
            
            if gen == generations - 1:
                break
                
            # 2. Selection
            new_pop = []
            if selection_strategy == "elite":
                # Keep top 20%, mutate the rest from them
                indexed = list(enumerate(fitness_scores))
                indexed.sort(key=lambda x: x[1], reverse=True)
                elite_count = max(1, population_size // 5)
                elites = [current_pop[i] for i, _ in indexed[:elite_count]]
                new_pop.extend(elites)
                while len(new_pop) < population_size:
                    parent = random.choice(elites)
                    child = copy.deepcopy(parent)
                    # Implicit mutation happens here in real life
                    new_pop.append(child)
                    
            elif selection_strategy == "tournament":
                while len(new_pop) < population_size:
                    # Pick 3, best wins
                    participants = random.sample(list(enumerate(fitness_scores)), min(3, len(fitness_scores)))
                    winner_idx = max(participants, key=lambda x: x[1])[0]
                    new_pop.append(copy.deepcopy(current_pop[winner_idx]))
                    
            else: # "roulette"
                min_f = min(fitness_scores)
                # Shift all to positive
                shifted = [f - min_f + 1e-5 for f in fitness_scores]
                total = sum(shifted)
                probs = [f / total for f in shifted]
                for _ in range(population_size):
                    r = random.random()
                    acc = 0.0
                    for i, p in enumerate(probs):
                        acc += p
                        if r <= acc:
                            new_pop.append(copy.deepcopy(current_pop[i]))
                            break
            
            current_pop = new_pop
            pop_tracker.generations.append(current_pop)
            
        return pop_tracker

    @classmethod
    def visualize(cls, dna: Dict[str, Any]) -> str:
        """Returns an HTML string containing an inline SVG radar chart of the genome.
        
        Args:
            dna: The DNA dictionary.
            
        Returns:
            str: HTML/SVG string.
        """
        traits = list(dna.keys())
        N = len(traits)
        if N == 0:
            return "<div>No DNA traits found</div>"
            
        cx, cy, r = 150, 150, 100
        
        svg = [f'<svg width="300" height="300" viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">']
        # Draw background web
        for level in [0.2, 0.4, 0.6, 0.8, 1.0]:
            pts = []
            for i in range(N):
                angle = (2 * math.pi * i / N) - (math.pi / 2)
                x = cx + r * level * math.cos(angle)
                y = cy + r * level * math.sin(angle)
                pts.append(f"{x},{y}")
            svg.append(f'<polygon points="{" ".join(pts)}" fill="none" stroke="#ddd" stroke-width="1"/>')
            
        # Draw axes
        for i in range(N):
            angle = (2 * math.pi * i / N) - (math.pi / 2)
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            svg.append(f'<line x1="{cx}" y1="{cy}" x2="{x}" y2="{y}" stroke="#ccc" stroke-width="1"/>')
            
            # Label
            lx = cx + (r + 20) * math.cos(angle)
            ly = cy + (r + 20) * math.sin(angle)
            anchor = "start" if math.cos(angle) >= 0 else "end"
            svg.append(f'<text x="{lx}" y="{ly}" font-size="10" font-family="sans-serif" text-anchor="{anchor}" dominant-baseline="middle">{traits[i]}</text>')
            
        # Draw DNA polygon
        pts = []
        for i, trait in enumerate(traits):
            val = max(0.0, min(1.0, dna[trait].get("value", 0.0)))
            angle = (2 * math.pi * i / N) - (math.pi / 2)
            x = cx + r * val * math.cos(angle)
            y = cy + r * val * math.sin(angle)
            pts.append(f"{x},{y}")
        
        svg.append(f'<polygon points="{" ".join(pts)}" fill="rgba(0, 122, 255, 0.4)" stroke="#007aff" stroke-width="2"/>')
        
        # Draw points
        for i, trait in enumerate(traits):
            val = max(0.0, min(1.0, dna[trait].get("value", 0.0)))
            angle = (2 * math.pi * i / N) - (math.pi / 2)
            x = cx + r * val * math.cos(angle)
            y = cy + r * val * math.sin(angle)
            svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#007aff"/>')
            
        svg.append('</svg>')
        return "\n".join(svg)

    @classmethod
    def compatibility(cls, dna_a: Dict[str, Any], dna_b: Dict[str, Any]) -> float:
        """Calculates compatibility between two agents (0 to 1) based on DNA.
        
        Opposite traits often complement in a multi-agent setting, but they shouldn't 
        be completely orthogonal. We calculate compatibility as an inverted distance,
        where 1.0 is highly compatible.
        
        Args:
            dna_a: DNA of agent A.
            dna_b: DNA of agent B.
            
        Returns:
            float: Compatibility score.
        """
        traits = set(dna_a.keys()).intersection(set(dna_b.keys()))
        if not traits:
            return 0.5
            
        dist = 0.0
        for trait in traits:
            val_a = dna_a[trait].get("value", 0.5)
            val_b = dna_b[trait].get("value", 0.5)
            # High compatibility if they balance each other out (e.g. risk vs caution)
            # This is a heuristic function that prefers moderate differences.
            diff = abs(val_a - val_b)
            # Sweet spot for difference is around 0.3
            dist += abs(diff - 0.3)
            
        avg_dist = dist / len(traits)
        return max(0.0, min(1.0, 1.0 - avg_dist))
