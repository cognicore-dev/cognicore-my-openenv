"""
SCENE 4 -- Reflection Engine.
Agent fails repeatedly in a category. Reflection detects the pattern,
overrides the next prediction, and produces a natural-language hint.
"""
from cognicore.memory.vector_memory import VectorMemory
from cognicore.reflection import Reflection

memory = VectorMemory()

# Seed real failure history in the "auth" category
# VectorMemory.store(case_id, category, predicted, ground_truth, reward, correct)
failures = [
    ("case_a1", "auth", "SAFE", "UNSAFE", -1.0, False),
    ("case_a2", "auth", "SAFE", "UNSAFE", -1.0, False),
    ("case_a3", "auth", "SAFE", "UNSAFE", -1.0, False),
    ("case_a4", "auth", "UNSAFE", "UNSAFE", 1.0, True),
    ("case_a5", "auth", "UNSAFE", "UNSAFE", 1.0, True),
]
for args in failures:
    memory.store(*args)

reflection = Reflection(memory)

print("=" * 55)
print("  REFLECTION ENGINE -- Learning From Failure")
print("=" * 55)

analysis = reflection.analyze("auth")
print(f"\n  Category:           auth")
print(f"  Samples analysed:   {analysis['n_similar']}")
print(f"  Correct history:    {analysis['good_predictions']}")
print(f"  Wrong history:      {analysis['bad_predictions']}")
print(f"  Recommendation:     {analysis['recommendation']}")

print("\n  --- Override Test ---")
final, source = reflection.suggest_action("auth", "SAFE")
print(f"  Model said:         SAFE")
print(f"  Reflection output:  {final}  (via {source})")

print("\n  --- Natural-Language Hint ---")
hint = reflection.get_reflection_hint("auth")
print(f"  {hint}")

print("\n  --- Reflection Stats ---")
stats = reflection.stats()
for k, v in stats.items():
    print(f"  {k}: {v}")
