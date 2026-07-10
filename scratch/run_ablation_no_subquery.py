import os
import subprocess
import glob

def run():
    print("==============================================")
    print("Running Ablation: Window=5, Overlap=2, NO SUB-QUERY EXPANSION")
    print("==============================================")
    
    print("\nGenerating predictions...")
    env = os.environ.copy()
    env["COGNICORE_WINDOW_SIZE"] = "5"
    env["COGNICORE_OVERLAP"] = "2"
    env["COGNICORE_DISABLE_SUBQUERY"] = "1"
    
    subprocess.run(["python", "cognicore_benchmarks/longmemeval/runner.py", "--adapter", "CognicoreZeroShotAdapter", "--mode", "official"], env=env, check=True)
    
    # Get the latest generated predictions file
    list_of_files = glob.glob('results/longmemeval/OFFICIAL_CognicoreZeroShotAdapter_predictions_*.jsonl')
    latest_file = max(list_of_files, key=os.path.getmtime)
    
    print(f"Evaluating predictions: {latest_file}")
    
    # Run the strict eval
    result = subprocess.run(["python", "scratch/eval_r5.py", "--preds", latest_file], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

if __name__ == "__main__":
    run()
