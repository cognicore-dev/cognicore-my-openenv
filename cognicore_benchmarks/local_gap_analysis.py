import json
import re
from pathlib import Path
from collections import defaultdict, Counter

def is_match(expected, text):
    if not expected or not text: return False
    # Simple heuristic
    return expected.lower() in text.lower()

def main():
    pred_file = Path("results/longmemeval/OFFICIAL_CognicoreFullAdapter_predictions_20260619T111242Z.jsonl")
    dataset_file = Path("cognicore_benchmarks/data/longmemeval/longmemeval_oracle.json")
    if not pred_file.exists() or not dataset_file.exists():
        print("Required files missing.")
        return

    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = {item["question_id"]: item for item in json.load(f)}

    predictions = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            p = json.loads(line)
            if p.get("hypothesis"): # Only keep valid ones
                predictions.append(p)

    print(f"Loaded {len(predictions)} valid predictions.")
    
    results = []
    failures = []
    categories = defaultdict(list)
    
    for p in predictions:
        qid = p["question_id"]
        ref = dataset[qid]
        
        expected = ref["answer"]
        hyp = p["hypothesis"]
        tel = p.get("_telemetry", {})
        retrieved = tel.get("retrieved_memories", [])
        
        # Deterministic check
        is_correct = is_match(expected, hyp)
        
        # Track by category
        q_type = ref["question_type"]
        categories[q_type].append(1 if is_correct else 0)
        
        if not is_correct:
            # Memory Existed check: Look in original conversation
            history_text = " ".join([m["content"] for m in ref.get("context", [])])
            if not history_text: 
                # If context is not in oracle directly, assume it existed for oracle
                history_text = expected 
                
            mem_existed = is_match(expected, history_text)
            
            # Retrieval check
            retrieved_idx = -1
            for i, r in enumerate(retrieved):
                if is_match(expected, r):
                    retrieved_idx = i
                    break
                    
            is_retrieved = retrieved_idx != -1
            
            # Ranking check
            is_ranked = is_retrieved and retrieved_idx <= 1
            
            # Context Noise
            total_mems = len(retrieved)
            relevant_mems = sum([1 for r in retrieved if is_match(expected, r)])
            noise_ratio = (total_mems - relevant_mems) / total_mems if total_mems > 0 else 1.0
            
            # Categorize
            if not is_retrieved:
                cat = "Retrieval Failure"
            elif q_type == "temporal-reasoning":
                cat = "Temporal Reasoning Failure"
            elif q_type == "knowledge-update":
                cat = "Knowledge Update Failure"
            elif not is_ranked:
                cat = "Ranking Failure"
            else:
                cat = "LLM Reasoning Failure"
                
            failures.append({
                "question_id": qid,
                "question": ref["question"],
                "expected": expected,
                "actual": hyp,
                "category": cat,
                "mem_existed": "YES" if mem_existed else "NO",
                "retrieved": "YES" if is_retrieved else "NO",
                "ranked": "YES" if is_ranked else "NO",
                "rank_idx": retrieved_idx,
                "total_mem": total_mems,
                "noise_ratio": noise_ratio
            })
            
    # Metrics
    accuracy = (len(predictions) - len(failures)) / len(predictions) if predictions else 0
    
    # Reports
    out_dir = Path(r"C:\Users\kaush\.gemini\antigravity\brain\bf226565-59da-4385-a320-4851e0a6a53c")
    
    # 1. benchmark_gap_analysis.md
    cat_counts = Counter([f["category"] for f in failures]) if failures else {}
    with open(out_dir / "benchmark_gap_analysis.md", "w", encoding="utf-8") as f:
        f.write(f"# CogniCore LongMemEval Gap Analysis\n\n")
        f.write(f"**Analyzed Samples:** {len(predictions)}\n")
        f.write(f"**Current Score (Deterministic):** {accuracy:.2%}\n")
        f.write(f"**Total Failures:** {len(failures)}\n\n")
        
        f.write("## Failure Distribution\n")
        f.write("| Failure Type | Count | Percent |\n|---|---|---|\n")
        for cat, count in cat_counts.items():
            pct = count / len(failures) * 100
            f.write(f"| {cat} | {count} | {pct:.1f}% |\n")
            
    # 2. category_breakdown.md
    with open(out_dir / "category_breakdown.md", "w", encoding="utf-8") as f:
        f.write("# Category Breakdown\n\n")
        f.write("| Category | Accuracy |\n|---|---|\n")
        for c, scores in categories.items():
            acc = sum(scores) / len(scores)
            f.write(f"| {c} | {acc:.2%} ({len(scores)} samples) |\n")
            
    # 3. retrieval_report.md
    if failures:
        avg_rank = sum([f["rank_idx"] for f in failures if f["retrieved"] == "YES"]) / sum([1 for f in failures if f["retrieved"] == "YES"]) if any(f["retrieved"] == "YES" for f in failures) else -1
        avg_noise = sum([f["noise_ratio"] for f in failures]) / len(failures)
        ret_rate = sum([1 for f in failures if f["retrieved"] == "YES"]) / len(failures)
    else:
        avg_rank = avg_noise = ret_rate = 0
        
    with open(out_dir / "retrieval_report.md", "w", encoding="utf-8") as f:
        f.write("# Retrieval Audit Report\n\n")
        f.write(f"- **Average Retrieval Rank (when found):** {avg_rank:.2f}\n")
        f.write(f"- **Average Noise Ratio:** {avg_noise:.2f}\n")
        f.write(f"- **Retrieval Success Rate (on failures):** {ret_rate:.2%}\n")
        
    # 4. improvement_roadmap.md
    with open(out_dir / "improvement_roadmap.md", "w", encoding="utf-8") as f:
        f.write("# Improvement Roadmap\n\n")
        f.write("| Improvement | Potential Gain | Effort | Priority |\n")
        f.write("| --- | --- | --- | --- |\n")
        
        ret_gain = f"+{(cat_counts.get('Retrieval Failure', 0) / len(predictions))*100:.1f}%" if len(predictions) > 0 else "0%"
        f.write(f"| **Better Semantic Retrieval** | {ret_gain} | Medium | High |\n")
        
        rank_gain = f"+{(cat_counts.get('Ranking Failure', 0) / len(predictions))*100:.1f}%" if len(predictions) > 0 else "0%"
        f.write(f"| **Context Reranking** | {rank_gain} | Low | High |\n")
        
        temp_gain = f"+{(cat_counts.get('Temporal Reasoning Failure', 0) / len(predictions))*100:.1f}%" if len(predictions) > 0 else "0%"
        f.write(f"| **Temporal Indexing** | {temp_gain} | High | Medium |\n")
        
        know_gain = f"+{(cat_counts.get('Knowledge Update Failure', 0) / len(predictions))*100:.1f}%" if len(predictions) > 0 else "0%"
        f.write(f"| **Knowledge Update Tracking** | {know_gain} | High | Medium |\n")

    print("Deterministic analysis complete.")

if __name__ == "__main__":
    main()
