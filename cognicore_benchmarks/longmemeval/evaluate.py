import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
import pandas as pd

from cognicore_benchmarks.common.llm_client import LLMClient
from cognicore_benchmarks.longmemeval.metrics import evaluate_metrics # For dev fallback

def get_anscheck_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    """Official LongMemEval LLM-as-a-judge prompts."""
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        else:
            # Fallback for unknown tasks in dev mode
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        return template.format(question, answer, response)

def load_references(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return {
        item["question_id"]: {
            "question": item["question"],
            "answer": item["answer"],
            "question_type": item["question_type"],
            "abstention": "_abs" in item["question_id"]
        } for item in data
    }

def main(predictions_file: str, dataset_file: str, mode: str, judge_model: str):
    pred_path = Path(predictions_file)
    ref_path = Path(dataset_file)
    
    if mode == "official" and not ref_path.exists():
        print(f"ERROR: Official dataset {ref_path} not found.")
        sys.exit(1)
        
    if not pred_path.exists():
        print(f"ERROR: Predictions file {pred_path} not found.")
        sys.exit(1)
        
    predictions = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
                
    if mode in ["official", "heuristic"]:
        references = load_references(ref_path)
    else:
        # Dev mode mock references
        references = {
            "q1": {"question": "What is my favorite color?", "answer": "blue", "question_type": "single-session-user", "abstention": False},
            "q2_abs": {"question": "What is my dog's name?", "answer": "", "question_type": "single-session-user", "abstention": True}
        }
        
    client = LLMClient(model_name=judge_model)
    if mode == "official" and client.is_mock:
        print("ERROR: Official Mode requires a valid OPENAI_API_KEY for the LLM judge.")
        sys.exit(1)
        
    results = []
    total_score = 0
    total_samples = 0
    
    # Check for cache file
    cache_file = pred_path.parent / f"cache_{mode}_{pred_path.stem}.jsonl"
    evaluated_qids = set()
    
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    res = json.loads(line)
                    results.append(res)
                    evaluated_qids.add(res["question_id"])
                    total_score += res["score"]
                    total_samples += 1
        print(f"Resumed {len(evaluated_qids)} evaluations from cache.")
    
    print(f"Evaluating {len(predictions) - len(evaluated_qids)} remaining predictions using {mode} mode...")
    
    for pred in predictions:
        qid = pred["question_id"]
        hyp = pred["hypothesis"]
        
        if qid in evaluated_qids:
            continue
            
        if qid not in references:
            print(f"Warning: Skipping {qid} as it is not in reference data.")
            continue
            
        ref = references[qid]
        is_abstain = ref["abstention"]
        
        if mode == "official":
            prompt = get_anscheck_prompt(ref["question_type"], ref["question"], ref["answer"], hyp, is_abstain)
            res = client.generate(prompt=prompt, system_prompt=None)
            eval_text = res["content"].strip().lower()
            score = 1.0 if "yes" in eval_text else 0.0
            eval_method = "llm-judge"
        else:
            # Dev fallback: fast string matching
            metrics = evaluate_metrics(hyp, ref["answer"], expects_abstain=is_abstain)
            score = metrics["score"]
            eval_text = "N/A (Dev heuristic)"
            eval_method = "heuristic"
            
        total_score += score
        total_samples += 1
        
        # Merge telemetry from prediction if available
        telemetry = pred.get("_telemetry", {})
        
        result_record = {
            "question_id": qid,
            "question_type": ref["question_type"],
            "score": score,
            "hypothesis": hyp,
            "eval_reason": eval_text,
            "eval_method": eval_method,
            "latency_s": telemetry.get("latency_s", 0),
            "tokens": telemetry.get("tokens", 0),
            "retrieved_memories": telemetry.get("retrieved_memories", []),
            "ranking_scores": telemetry.get("ranking_scores", []),
            "reflection_hint": telemetry.get("reflection_hint", ""),
            "draft_answer": telemetry.get("draft_answer", "")
        }
        
        results.append(result_record)
        evaluated_qids.add(qid)
        
        # Save to cache immediately
        with open(cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_record) + "\n")
        
    # Build final output wrapper
    metadata = {
        "benchmark": "LongMemEval",
        "benchmark_commit": "9e0b455f4ef0e2ab8f2e582289761153549043fc",
        "dataset_sha256": "UNKNOWN_IN_DEV" if mode == "dev" else "LOADED_FROM_METADATA", # Real script reads metadata.json
        "judge_model": judge_model if mode == "official" else "heuristic",
        "cognicore_commit": "local", # Should be extracted via git
        "timestamp": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "mode": mode
    }
    
    # Try to load real SHA256 if official
    if mode == "official":
        meta_path = Path("cognicore_benchmarks/data/longmemeval/metadata.json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                md = json.load(f)
                metadata["dataset_sha256"] = md.get("datasets", {}).get(ref_path.name, {}).get("sha256", "MISSING")
    
    accuracy = total_score / total_samples if total_samples > 0 else 0
    
    final_payload = {
        "reproducibility": metadata,
        "aggregate_stats": {
            "samples_evaluated": total_samples,
            "accuracy": accuracy
        },
        "results": results
    }
    
    out_dir = pred_path.parent
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    prefix = "UNOFFICIAL_" if mode == "dev" else "OFFICIAL_"
    out_file = out_dir / f"{prefix}evaluation_{timestamp}.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
        
    print(f"Evaluation complete. Accuracy: {accuracy:.2%}")
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LongMemEval Predictions.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions.jsonl")
    parser.add_argument("--dataset", type=str, default="cognicore_benchmarks/data/longmemeval/longmemeval_oracle.json", help="Path to reference dataset JSON")
    parser.add_argument("--mode", type=str, choices=["dev", "official", "heuristic"], default="dev", help="Execution mode.")
    parser.add_argument("--judge_model", type=str, default="openai/gpt-4o-mini", help="Model to use for LLM judging (official mode).")
    
    args = parser.parse_args()
    main(args.predictions, args.dataset, args.mode, args.judge_model)
