import os
import subprocess
import re
import glob

# Window sizes and overlaps to test
configs = [
    (5, 2),
    (10, 5),
    (15, 7),
    (20, 10)
]

results = {}

for window_size, overlap in configs:
    print(f"\n==============================================")
    print(f"Running Ablation: Window={window_size}, Overlap={overlap}")
    print(f"==============================================\n")
    
    # 1. Run the benchmark
    env = os.environ.copy()
    env["COGNICORE_WINDOW_SIZE"] = str(window_size)
    env["COGNICORE_OVERLAP"] = str(overlap)
    
    print("Generating predictions...")
    subprocess.run(
        ["python", "cognicore_benchmarks/longmemeval/runner.py", "--adapter", "CognicoreZeroShotAdapter", "--mode", "official"],
        env=env,
        check=True
    )
    
    # 2. Find the latest prediction file
    pred_files = glob.glob("results/longmemeval/OFFICIAL_CognicoreZeroShotAdapter_predictions_*.jsonl")
    latest_pred = max(pred_files, key=os.path.getmtime)
    
    # 3. Evaluate strict R@5
    print(f"Evaluating predictions: {latest_pred}")
    result = subprocess.run(
        ["python", "scratch/eval_r5.py", "--preds", latest_pred],
        capture_output=True,
        text=True,
        check=True
    )
    
    # Extract STRICT R@5 from output
    match = re.search(r"STRICT R@5: ([\d.]+)%", result.stdout)
    if match:
        score = float(match.group(1))
        results[window_size] = score
        print(f"--> Score for Window={window_size}: {score}%")
    else:
        print("Failed to parse score.")

print("\n\nFINAL ABLATION RESULTS:")
for w, s in results.items():
    print(f"Window={w}: {s}%")
    
# Generate mermaid markdown for plotting
mermaid = "```mermaid\nxychart-beta\n    title \"Strict R@5 vs. Window Size\"\n    x-axis \"Window Size (turns)\" [" + ", ".join(str(k) for k in results.keys()) + "]\n    y-axis \"Strict R@5 (%)\" 0 --> 100\n    line [" + ", ".join(str(v) for v in results.values()) + "]\n```"

with open("scratch/ablation_results.md", "w") as f:
    f.write("# Ablation Study Results\n\n")
    for w, s in results.items():
        f.write(f"- **Window Size {w}**: {s}%\n")
    f.write("\n" + mermaid + "\n")

print("\nResults saved to scratch/ablation_results.md")
