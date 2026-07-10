import json
import os
from collections import defaultdict

def analyze_trajectories(jsonl_path: str = "results/results.jsonl"):
    if not os.path.exists(jsonl_path):
        print(f"Results file {jsonl_path} not found.")
        return

    # metrics[domain][arm][period] = { "successes": 0, "rfr_sum": 0.0, "count": 0 }
    # period: "Ep 1-10" or "Ep 11-20"
    metrics = defaultdict(lambda: defaultdict(lambda: {
        "Ep 1-10": {"successes": 0, "rfr_sum": 0.0, "count": 0},
        "Ep 11-20": {"successes": 0, "rfr_sum": 0.0, "count": 0}
    }))
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            arm = record["arm"]
            domain = record["domain"]
            episode = record["episode"] # 1 to 20
            
            period = "Ep 1-10" if episode <= 10 else "Ep 11-20"
            
            m = metrics[domain][arm][period]
            m["count"] += 1
            if record["solve_status"]: m["successes"] += 1
            m["rfr_sum"] += record["repeated_failure_rate"]
            
    print("=== CogniCore-Bench v0.1 Trajectory Analysis ===")
    
    for domain in ["software_debugging", "deployment_infrastructure"]:
        print(f"\n--- Domain: {domain} ---")
        print(f"{'Arm':<30} | {'Solve % (Ep 1-10)':<17} | {'Solve % (Ep 11-20)':<18} | {'Avg RFR (Ep 1-10)':<17} | {'Avg RFR (Ep 11-20)':<18}")
        print("-" * 110)
        
        arms_sorted = sorted(metrics[domain].keys())
        for arm in arms_sorted:
            data = metrics[domain][arm]
            
            # Period 1
            p1 = data["Ep 1-10"]
            p1_solve = (p1["successes"] / p1["count"] * 100) if p1["count"] > 0 else 0.0
            p1_rfr = (p1["rfr_sum"] / p1["count"]) if p1["count"] > 0 else 0.0
            
            # Period 2
            p2 = data["Ep 11-20"]
            p2_solve = (p2["successes"] / p2["count"] * 100) if p2["count"] > 0 else 0.0
            p2_rfr = (p2["rfr_sum"] / p2["count"]) if p2["count"] > 0 else 0.0
            
            print(f"{arm:<30} | {p1_solve:>16.1f}% | {p2_solve:>17.1f}% | {p1_rfr:>17.2f} | {p2_rfr:>18.2f}")

if __name__ == "__main__":
    analyze_trajectories()
