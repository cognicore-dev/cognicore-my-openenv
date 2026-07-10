import os
import subprocess
import re
import glob

# Window sizes to test
window_size = 10
configs = [
    "CognicoreZeroShotAdapter",
    "CognicoreMultiHopAdapter"
]

results = {}

for adapter in configs:
    print(f"\n=============================================")
    print(f"EVALUATING ADAPTER: {adapter} (Window {window_size})")
    print(f"=============================================")
    
    # Environment variables to pass
    env = os.environ.copy()
    env["COGNICORE_WINDOW_SIZE"] = str(window_size)
    env["COGNICORE_API_KEY"] = env.get("GEMINI_API_KEY", "") # Pass key directly
    env["COGNICORE_EVAL_MODE"] = "official" 
    
    # Disable sub-queries on ZeroShot for fairness
    env["COGNICORE_DISABLE_SUBQUERY"] = "1"
    
    # Remove LM Studio overrides
    env.pop("OPENAI_BASE_URL", None)
    env.pop("OPENAI_API_KEY", None)
    
    # Pass OpenRouter API Key
    env["OPENROUTER_API_KEY"] = "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"
    env.pop("GROQ_API_KEY", None)
    env.pop("GEMINI_API_KEY", None)
    
    # Use OpenRouter free Llama 3.3 model (fast, massive context, free)
    or_model = "meta-llama/llama-3.3-70b-instruct:free"
    env["COGNICORE_EVAL_MODEL"] = or_model
    
    print(f"Generating answers using OpenRouter (Model: {or_model})...")
    # NOTE: limit to 50 samples to run the full official evaluation
    limit = "50" 
    
    subprocess.run(
        ["python", "-u", "cognicore_benchmarks/longmemeval/runner.py", "--adapter", adapter, "--mode", "official", "--limit", limit],
        env=env,
        check=True
    )
    
    # Find the latest prediction file
    pred_files = glob.glob(f"results/longmemeval/OFFICIAL_{adapter}_predictions_*.jsonl")
    if not pred_files:
        print("Failed to find prediction file!")
        continue
        
    latest_pred = max(pred_files, key=os.path.getmtime)
    
    # Evaluate Accuracy using LLM-as-a-judge
    print(f"Evaluating QA Accuracy of: {latest_pred}")
    result = subprocess.run(
        ["python", "-u", "cognicore_benchmarks/longmemeval/evaluate.py", "--predictions", latest_pred, "--mode", "official", "--judge_model", or_model],
        capture_output=True,
        text=True,
        env=env,
        check=True
    )
    
    print(result.stdout)
    
    # Extract Accuracy from output
    match = re.search(r"Accuracy: ([\d.]+)%", result.stdout)
    if match:
        score = float(match.group(1))
        results[f"{adapter}"] = score
        print(f"--> QA Accuracy for {adapter} at Window {window_size}: {score}%")
    else:
        print("Failed to parse score.")

print("\n\nFINAL STANDARDIZED QA EVALUATION RESULTS (WIN=10):")
for k, v in results.items():
    print(f"{k}: {v}%")
    
with open("scratch/standard_eval_win10.md", "w") as f:
    f.write("# Standardized LLM Generation Evaluation (Window 10)\n\n")
    f.write(f"**Model**: `{or_model}`\n")
    for k, v in results.items():
        f.write(f"- **{k}**: {v}%\n")

print("\nResults saved to scratch/standard_eval_win10.md")
