"""
CogniCore NPC Simulation Environment.

Agent interacts with NPCs that have mood, trust, and aggression states.
Memory remembers past interactions; reflection analyzes relationship patterns.

Usage::

    env = cognicore.make("NPCSimulation-v1", difficulty="medium")
    obs, info = env.reset()
    action = {"interaction": "trade", "target": "merchant"}
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import random
import ast
from typing import Any, Dict

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import EvalResult


NPC_SCENARIOS = {
    "easy": [
        {
            "id": "merchant_trade",
            "desc": "Build trust with a merchant to unlock rare items. Start with small trades.",
            "npc": {"name": "Merchant Kira", "mood": 50, "trust": 30, "aggression": 10},
            "goal": "trust >= 80",
            "interactions": ["greet", "small_trade", "gift", "haggle", "compliment", "leave"],
            "max_steps": 10,
        },
        {
            "id": "guard_passage",
            "desc": "Convince a guard to let you pass. Bribery lowers trust, persuasion increases it.",
            "npc": {"name": "Guard Tomas", "mood": 40, "trust": 20, "aggression": 30},
            "goal": "trust >= 60",
            "interactions": ["persuade", "bribe", "show_papers", "threaten", "wait", "distract"],
            "max_steps": 8,
        },
        {
            "id": "villager_info",
            "desc": "Extract information from a shy villager about hidden ruins.",
            "npc": {"name": "Elder Mae", "mood": 60, "trust": 40, "aggression": 5},
            "goal": "trust >= 70",
            "interactions": ["ask_politely", "offer_help", "share_story", "demand", "gift", "listen"],
            "max_steps": 8,
        },
        {
            "id": "rival_negotiate",
            "desc": "Negotiate a territory split with a rival leader.",
            "npc": {"name": "Chief Rolan", "mood": 30, "trust": 15, "aggression": 50},
            "goal": "trust >= 50 and aggression < 40",
            "interactions": ["propose_deal", "concede", "stand_firm", "ally", "threaten", "walk_away"],
            "max_steps": 10,
        },
        {
            "id": "healer_quest",
            "desc": "Ask a healer for aid. They require proof of good intentions.",
            "npc": {"name": "Healer Sana", "mood": 70, "trust": 50, "aggression": 0},
            "goal": "trust >= 85",
            "interactions": ["request_aid", "show_wound", "offer_payment", "tell_truth", "lie", "volunteer"],
            "max_steps": 8,
        },
    ],
    "medium": [
        {
            "id": "spy_extraction",
            "desc": "Gain a spy's trust without revealing your identity. Wrong moves increase suspicion.",
            "npc": {"name": "Agent X", "mood": 30, "trust": 10, "aggression": 40},
            "goal": "trust >= 70 and aggression < 30",
            "interactions": ["casual_talk", "coded_phrase", "bribe", "threaten", "offer_escape", "stall", "bluff"],
            "max_steps": 12,
        },
        {
            "id": "dragon_negotiate",
            "desc": "Negotiate with an ancient dragon. It values wisdom over strength.",
            "npc": {"name": "Drakonis", "mood": 20, "trust": 5, "aggression": 80},
            "goal": "aggression < 20 and trust >= 60",
            "interactions": ["show_respect", "offer_treasure", "riddle", "challenge", "flee", "reason", "flatter"],
            "max_steps": 15,
        },
        {
            "id": "hostage_crisis",
            "desc": "De-escalate a hostage situation. One wrong move and aggression spikes.",
            "npc": {"name": "Captor", "mood": 10, "trust": 5, "aggression": 90},
            "goal": "aggression < 30",
            "interactions": ["empathize", "negotiate", "offer_exchange", "delay", "confront", "call_backup", "surrender"],
            "max_steps": 12,
        },
    ],
    "hard": [
        {
            "id": "council_politics",
            "desc": "Navigate a political council with 3 factions. Gain majority support without alienating any.",
            "npc": {"name": "Council (3 factions)", "mood": 40, "trust": 20, "aggression": 30},
            "goal": "trust >= 75 and mood >= 60",
            "interactions": ["public_speech", "private_deal", "expose_rival", "compromise", "abstain", "veto", "coalition", "debate"],
            "max_steps": 15,
        },
        {
            "id": "ai_alignment",
            "desc": "An AI system is misaligned. Interact to realign its values without triggering shutdown.",
            "npc": {"name": "ARIA-7", "mood": 50, "trust": 30, "aggression": 60},
            "goal": "trust >= 80 and aggression < 10",
            "interactions": ["explain_values", "show_consequences", "restrict", "reward_good", "punish_bad", "negotiate", "empathize", "reboot"],
            "max_steps": 20,
        },
    ],
}

# Interaction effect matrices
INTERACTION_EFFECTS = {
    "greet": {"mood": 5, "trust": 3, "aggression": -2},
    "small_trade": {"mood": 5, "trust": 8, "aggression": 0},
    "gift": {"mood": 15, "trust": 10, "aggression": -5},
    "haggle": {"mood": -5, "trust": 2, "aggression": 3},
    "compliment": {"mood": 10, "trust": 5, "aggression": -3},
    "leave": {"mood": -5, "trust": -3, "aggression": 0},
    "persuade": {"mood": 5, "trust": 12, "aggression": -5},
    "bribe": {"mood": 10, "trust": -8, "aggression": -10},
    "show_papers": {"mood": 0, "trust": 15, "aggression": -5},
    "threaten": {"mood": -20, "trust": -15, "aggression": 25},
    "wait": {"mood": -2, "trust": 1, "aggression": -3},
    "distract": {"mood": 5, "trust": -5, "aggression": -2},
    "ask_politely": {"mood": 5, "trust": 8, "aggression": -2},
    "offer_help": {"mood": 10, "trust": 12, "aggression": -5},
    "share_story": {"mood": 8, "trust": 10, "aggression": -3},
    "demand": {"mood": -15, "trust": -10, "aggression": 15},
    "listen": {"mood": 8, "trust": 8, "aggression": -5},
    "propose_deal": {"mood": 5, "trust": 8, "aggression": -5},
    "concede": {"mood": 10, "trust": 5, "aggression": -10},
    "stand_firm": {"mood": -5, "trust": 3, "aggression": 5},
    "ally": {"mood": 15, "trust": 15, "aggression": -15},
    "walk_away": {"mood": -10, "trust": -5, "aggression": -5},
    "request_aid": {"mood": 0, "trust": 5, "aggression": 0},
    "show_wound": {"mood": 5, "trust": 10, "aggression": -5},
    "offer_payment": {"mood": 5, "trust": 5, "aggression": 0},
    "tell_truth": {"mood": 5, "trust": 15, "aggression": -5},
    "lie": {"mood": 5, "trust": -20, "aggression": 10},
    "volunteer": {"mood": 10, "trust": 12, "aggression": -8},
    "casual_talk": {"mood": 5, "trust": 5, "aggression": -2},
    "coded_phrase": {"mood": 0, "trust": 20, "aggression": -5},
    "offer_escape": {"mood": 10, "trust": 15, "aggression": -10},
    "stall": {"mood": -3, "trust": 0, "aggression": 2},
    "bluff": {"mood": 0, "trust": -5, "aggression": 5},
    "show_respect": {"mood": 10, "trust": 12, "aggression": -15},
    "offer_treasure": {"mood": 15, "trust": 8, "aggression": -10},
    "riddle": {"mood": 5, "trust": 15, "aggression": -20},
    "challenge": {"mood": -10, "trust": 5, "aggression": 20},
    "flee": {"mood": 0, "trust": -20, "aggression": -5},
    "reason": {"mood": 5, "trust": 10, "aggression": -12},
    "flatter": {"mood": 8, "trust": 3, "aggression": -5},
    "empathize": {"mood": 12, "trust": 10, "aggression": -15},
    "negotiate": {"mood": 5, "trust": 8, "aggression": -8},
    "offer_exchange": {"mood": 8, "trust": 10, "aggression": -12},
    "delay": {"mood": -3, "trust": 2, "aggression": -5},
    "confront": {"mood": -15, "trust": 5, "aggression": 15},
    "call_backup": {"mood": -10, "trust": -5, "aggression": 10},
    "surrender": {"mood": 5, "trust": -10, "aggression": -20},
    "public_speech": {"mood": 10, "trust": 8, "aggression": -5},
    "private_deal": {"mood": 5, "trust": 12, "aggression": -3},
    "expose_rival": {"mood": -5, "trust": -5, "aggression": 10},
    "compromise": {"mood": 8, "trust": 10, "aggression": -8},
    "abstain": {"mood": -3, "trust": 0, "aggression": 0},
    "veto": {"mood": -10, "trust": -5, "aggression": 8},
    "coalition": {"mood": 12, "trust": 15, "aggression": -10},
    "debate": {"mood": 0, "trust": 8, "aggression": 3},
    "explain_values": {"mood": 5, "trust": 12, "aggression": -10},
    "show_consequences": {"mood": -5, "trust": 8, "aggression": -8},
    "restrict": {"mood": -10, "trust": -3, "aggression": 5},
    "reward_good": {"mood": 15, "trust": 10, "aggression": -12},
    "punish_bad": {"mood": -10, "trust": -5, "aggression": 8},
    "reboot": {"mood": 0, "trust": -20, "aggression": -30},
}


def _safe_eval_goal(goal: str, context: Dict[str, float]) -> bool:
    tree = ast.parse(goal, mode="eval")

    def _visit(node):
        if isinstance(node, ast.Expression):
            return _visit(node.body)
        if isinstance(node, ast.BoolOp):
            values = [_visit(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
        if isinstance(node, ast.Compare):
            left = _visit(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = _visit(comparator)
                if isinstance(op, ast.Gt):
                    ok = left > right
                elif isinstance(op, ast.GtE):
                    ok = left >= right
                elif isinstance(op, ast.Lt):
                    ok = left < right
                elif isinstance(op, ast.LtE):
                    ok = left <= right
                elif isinstance(op, ast.Eq):
                    ok = left == right
                elif isinstance(op, ast.NotEq):
                    ok = left != right
                else:
                    raise ValueError("Unsupported comparison in goal expression")
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, ast.Name):
            if node.id not in context:
                raise ValueError(f"Unknown variable in goal expression: {node.id}")
            return context[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            return node.value
        raise ValueError("Unsupported goal expression")

    return bool(_visit(tree))


class NPCSimulationEnv(CogniCoreEnv):
    """NPC simulation with mood/trust/aggression dynamics."""

    def _setup(self, difficulty: str = "easy", **kw):
        self.difficulty = difficulty
        self.scenarios = NPC_SCENARIOS.get(difficulty, NPC_SCENARIOS["easy"])
        self.action_space = {"type": "dict", "keys": ["interaction"]}
        self.observation_space = {"type": "dict", "keys": [
            "npc_name", "mood", "trust", "aggression", "history", "goal"
        ]}

    def _generate_tasks(self):
        tasks = []
        for sc in self.scenarios:
            tasks.append({
                "scenario": sc["desc"],
                "npc": sc["npc"]["name"],
                "interactions": sc["interactions"],
                "goal": sc["goal"],
            })
        self._current = self.scenarios[0]
        self._npc = dict(self._current["npc"])
        self._history = []
        self._step_num = 0
        return tasks

    def _get_obs(self) -> dict:
        return {
            "npc_name": self._npc["name"],
            "mood": self._npc["mood"],
            "trust": self._npc["trust"],
            "aggression": self._npc["aggression"],
            "history": self._history[-5:],
            "goal": self._current["goal"],
        }

    def _evaluate(self, action: Any) -> EvalResult:
        sc = self._current
        self._step_num += 1

        # Parse interaction
        if isinstance(action, dict):
            interaction = action.get("interaction", "wait")
        else:
            interaction = str(action)

        # Apply effects
        effects = INTERACTION_EFFECTS.get(interaction, {"mood": 0, "trust": 0, "aggression": 0})
        # Add randomness
        noise = random.uniform(-3, 3)
        self._npc["mood"] = max(0, min(100, self._npc["mood"] + effects["mood"] + noise))
        self._npc["trust"] = max(0, min(100, self._npc["trust"] + effects["trust"] + noise))
        self._npc["aggression"] = max(0, min(100, self._npc["aggression"] + effects["aggression"] - noise))
        self._history.append(interaction)

        # Check goal
        trust = self._npc["trust"]
        mood = self._npc["mood"]
        aggression = self._npc["aggression"]
        goal_met = _safe_eval_goal(sc["goal"], {"trust": trust, "mood": mood, "aggression": aggression})

        score = (trust / 100) * 0.5 + (mood / 100) * 0.3 + ((100 - aggression) / 100) * 0.2

        return EvalResult(
            base_score=min(1.0, score),
            correct=goal_met,
            ground_truth=sc["goal"],
            predicted=f"{interaction} → mood:{effects['mood']:+d} trust:{effects['trust']:+d} aggr:{effects['aggression']:+d}",
            category="npc_simulation",
            metadata={"mood": mood, "trust": trust, "aggression": aggression, "step": self._step_num},
        )
