import json
import os
from collections import defaultdict

def analyze_results(jsonl_path: str = "results/results.jsonl"):
    if not os.path.exists(jsonl_path):
        print(f"Results file {jsonl_path} not found.")
        return

    # metrics[arm][domain] = { "successes": 0, "first_action": 0, "rfr": 0.0, "retries": 0, "cost": 0.0, "count": 0 }
    metrics = defaultdict(lambda: defaultdict(lambda: {
        "successes": 0, 
        "first_action": 0, 
        "rfr_sum": 0.0, 
        "retries_sum": 0, 
        "cost_usd": 0.0, 
        "count": 0
    }))
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            arm = record["arm"]
            domain = record["domain"]
            
            m = metrics[arm][domain]
            m["count"] += 1
            if record["solve_status"]: m["successes"] += 1
            if record["first_action_accuracy"]: m["first_action"] += 1
            m["rfr_sum"] += record["repeated_failure_rate"]
            m["retries_sum"] += record["retry_count"]
            m["cost_usd"] += record["cost_usd"]
            
    print("=== CogniCore-Bench v0.1 Analysis ===")
    
    for domain in ["software_debugging", "deployment_infrastructure"]:
        print(f"\n--- Domain: {domain} ---")
        print(f"{'Arm':<30} | {'Solve %':<8} | {'1st Act %':<10} | {'Avg RFR':<8} | {'Avg Retries':<12} | {'Total Cost':<10}")
        print("-" * 90)
        
        # Sort arms logically
        arms_sorted = sorted(metrics.keys())
        for arm in arms_sorted:
            m = metrics[arm].get(domain)
            if not m or m["count"] == 0: continue
            
            count = m["count"]
            solve_pct = (m["successes"] / count) * 100
            first_pct = (m["first_action"] / count) * 100
            avg_rfr = m["rfr_sum"] / count
            avg_retries = m["retries_sum"] / count
            cost = m["cost_usd"]
            
            print(f"{arm:<30} | {solve_pct:>6.1f}% | {first_pct:>8.1f}% | {avg_rfr:>8.2f} | {avg_retries:>12.1f} | ${cost:>8.4f}")

if __name__ == "__main__":
    analyze_results()
