import os
import subprocess
import re
import glob

# Window sizes to test
configs = [5, 10]

results = {}

for window_size in configs:
    for adapter in ["CognicoreZeroShotAdapter", "CognicoreMultiHopAdapter"]:
        print(f"\n==============================================")
        print(f"Running Eval: Adapter={adapter}, Window={window_size}")
        print(f"==============================================\n")
        
        env = os.environ.copy()
        env["COGNICORE_WINDOW_SIZE"] = str(window_size)
        
        # Overlap dynamically
        overlap = 2 if window_size == 5 else 5
        env["COGNICORE_OVERLAP"] = str(overlap)
        
        # Disable sub-queries on ZeroShot for fairness since MultiHop replaces it
        env["COGNICORE_DISABLE_SUBQUERY"] = "1"
        
        print("Generating predictions...")
        subprocess.run(
            ["python", "cognicore_benchmarks/longmemeval/runner.py", "--adapter", adapter, "--mode", "official"],
            env=env,
            check=True
        )
        
        # Find the latest prediction file
        pred_files = glob.glob(f"results/longmemeval/OFFICIAL_{adapter}_predictions_*.jsonl")
        if not pred_files:
            print("Failed to find prediction file!")
            continue
            
        latest_pred = max(pred_files, key=os.path.getmtime)
        
        # Evaluate strict R@5
        print(f"Evaluating predictions: {latest_pred}")
        result = subprocess.run(
            ["python", "scratch/eval_r5.py", "--preds", latest_pred],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        
        # Extract STRICT R@5 from output
        match = re.search(r"STRICT R@5: ([\d.]+)%", result.stdout)
        if match:
            score = float(match.group(1))
            results[f"{adapter}_Win{window_size}"] = score
            print(f"--> Score for {adapter} at Window {window_size}: {score}%")
        else:
            print("Failed to parse score.")

print("\n\nFINAL MULTI-HOP EVALUATION RESULTS:")
for k, v in results.items():
    print(f"{k}: {v}%")
    
with open("scratch/multihop_results.md", "w") as f:
    f.write("# Multi-Hop Retrieval Evaluation\n\n")
    for k, v in results.items():
        f.write(f"- **{k}**: {v}%\n")

print("\nResults saved to scratch/multihop_results.md")
