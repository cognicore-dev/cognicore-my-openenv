import json
import sys
from collections import defaultdict

def main():
    verdicts_file = r"C:\Users\kaush\OneDrive\Documents\safetymind\verdicts-beam-small-cognicore.jsonl"
    
    total = 0
    correct = 0
    
    ability_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    tier_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    try:
        with open(verdicts_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                is_correct = (data.get("verdict") == "correct")
                ability = data.get("ability", "unknown")
                tier = data.get("tier", "unknown")
                
                total += 1
                if is_correct:
                    correct += 1
                    
                ability_stats[ability]["total"] += 1
                if is_correct:
                    ability_stats[ability]["correct"] += 1
                    
                tier_stats[tier]["total"] += 1
                if is_correct:
                    tier_stats[tier]["correct"] += 1
                    
    except FileNotFoundError:
        print(f"File not found: {verdicts_file}")
        sys.exit(1)
        
    print("=" * 40)
    print("BENCHMARK RESULTS")
    print("=" * 40)
    print(f"Total Examples: {total}")
    if total > 0:
        print(f"Overall Accuracy: {correct/total*100:.2f}% ({correct}/{total})")
    
    print("\n--- Accuracy by Ability ---")
    for ability, stats in sorted(ability_stats.items()):
        acc = stats["correct"] / stats["total"] * 100
        print(f"{ability:<20}: {acc:>6.2f}% ({stats['correct']}/{stats['total']})")
        
    print("\n--- Accuracy by Tier ---")
    for tier, stats in sorted(tier_stats.items()):
        acc = stats["correct"] / stats["total"] * 100
        print(f"{tier:<20}: {acc:>6.2f}% ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    main()
