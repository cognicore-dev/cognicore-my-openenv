import os
import time
import json
import argparse
from typing import Type, List, Dict, Any
from pathlib import Path
from datetime import datetime

from cognicore_benchmarks.longmemeval.adapters.base_adapter import BaseAgentAdapter

# Mock data for Dev Mode
MOCK_DATA = [
    {
        "question_id": "q1",
        "question_type": "single-session-user",
        "history": [{"role": "user", "content": "My favorite color is blue."}, {"role": "assistant", "content": "Got it."}],
        "question": "What is my favorite color?",
        "answer": "blue",
    },
    {
        "question_id": "q2_abs",
        "question_type": "single-session-user",
        "history": [{"role": "user", "content": "I like pizza."}],
        "question": "What is my dog's name?",
        "answer": "",
    }
]

class LongMemEvalRunner:
    """Orchestrates the generation phase of the LongMemEval benchmark."""

    def __init__(self, adapter_cls: Type[BaseAgentAdapter], adapter_kwargs: dict = None, mode: str = "dev"):
        self.mode = mode
        self.adapter_kwargs = adapter_kwargs or {}
        self.adapter = adapter_cls(**self.adapter_kwargs)
        self.adapter_name = adapter_cls.__name__

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads the dataset depending on the mode."""
        if self.mode == "dev":
            print("[UNOFFICIAL_DEV_RUN] Using embedded mock dataset.")
            return MOCK_DATA
            
        print("[OFFICIAL_MODE] Loading longmemeval_oracle.json...")
        data_path = Path("cognicore_benchmarks/data/longmemeval/longmemeval_oracle.json")
        if not data_path.exists():
            raise FileNotFoundError(
                f"Official dataset missing at {data_path}. "
                "Please run cognicore_benchmarks/data/longmemeval/download_datasets.py first."
            )
            
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        return dataset

    def run(self, limit: int = None, output_dir: str = "results/longmemeval", resume_file: str = None) -> str:
        """
        Executes the benchmark generation phase and writes predictions.jsonl.
        """
        dataset = self.load_dataset()
        if limit:
            dataset = dataset[:limit]
            
        print(f"Starting Generation Phase: {len(dataset)} samples using {self.adapter_name}...")
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        completed_ids = set()
        if resume_file and Path(resume_file).exists():
            print(f"Resuming from {resume_file}...")
            output_file = Path(resume_file)
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            completed_ids.add(str(record.get("question_id")))
                        except json.JSONDecodeError:
                            pass
            print(f"Loaded {len(completed_ids)} completed samples.")
            file_mode = "a"
        else:
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            prefix = "UNOFFICIAL_" if self.mode == "dev" else "OFFICIAL_"
            filename = f"{prefix}{self.adapter_name}_predictions_{timestamp}.jsonl"
            output_file = out_path / filename
            file_mode = "w"
        
        with open(output_file, file_mode, encoding="utf-8") as f:
            for idx, item in enumerate(dataset):
                q_id = str(item.get("question_id", idx))
                if q_id in completed_ids:
                    print(f"  Skipping [{idx+1}/{len(dataset)}] ID: {q_id} (already completed)")
                    continue
                    
                print(f"  Processing [{idx+1}/{len(dataset)}] ID: {q_id}")
                
                # Format history: official dataset has 'haystack_sessions' 
                # Each element in haystack_sessions is a list of turns. We flatten it.
                if self.mode == "dev":
                    history = item.get("history", [])
                else:
                    raw_sessions = item.get("haystack_sessions", [])
                    history = []
                    for session in raw_sessions:
                        for turn in session:
                            history.append({"role": turn["role"], "content": turn["content"]})
                
                # Parse question date to timestamp
                question_date_str = item.get("question_date")
                question_ts = None
                if question_date_str:
                    try:
                        import re
                        clean_date = re.sub(r'\s*\([A-Za-z]+\)\s*', ' ', question_date_str)
                        question_ts = datetime.strptime(clean_date, "%Y/%m/%d %H:%M").timestamp()
                    except Exception as e:
                        pass
                
                # Ingest & Answer
                if hasattr(self.adapter, "ingest_history_structured") and self.mode != "dev":
                    self.adapter.ingest_history_structured(
                        raw_sessions=item.get("haystack_sessions", []),
                        dates=item.get("haystack_dates", []),
                        session_ids=item.get("haystack_session_ids", [])
                    )
                else:
                    self.adapter.ingest_history(history)
                
                # Answer question
                import inspect
                sig = inspect.signature(self.adapter.answer_question)
                if "question_timestamp" in sig.parameters:
                    response = self.adapter.answer_question(item["question"], question_timestamp=question_ts)
                else:
                    response = self.adapter.answer_question(item["question"])
                
                # Format exactly as required by official evaluation script:
                # `question_id` and `hypothesis`
                record = {
                    "question_id": item.get("question_id", str(idx)),
                    "hypothesis": response["answer"],
                    # Telemetry fields (ignored by official evaluator but useful for us)
                    "_telemetry": {
                        "latency_s": response.get("latency_s", 0),
                        "tokens": response.get("tokens", 0),
                        "adapter": self.adapter_name,
                        "mode": self.mode,
                        "retrieved_memories": response.get("retrieved_memories", []),
                        "ranking_scores": response.get("ranking_scores", []),
                        "reflection_hint": response.get("reflection_hint", ""),
                        "draft_answer": response.get("draft_answer", "")
                    }
                }
                
                # Write JSONL
                f.write(json.dumps(record) + "\n")
                f.flush()

        print(f"Generation complete. Predictions saved to: {output_file}")
        return str(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LongMemEval Generation Phase.")
    parser.add_argument("--mode", type=str, choices=["dev", "official"], default="dev", help="Execution mode.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process.")
    
    parser.add_argument("--adapter", type=str, default="BaselineAdapter", help="Adapter class to run.")
    parser.add_argument("--resume", type=str, default=None, help="Path to predictions.jsonl file to resume from.")
    args = parser.parse_args()
    
    adapter_map = {
        "BaselineAdapter": "cognicore_benchmarks.longmemeval.adapters.baseline_adapter",
        "CognicoreFullAdapter": "cognicore_benchmarks.longmemeval.adapters.cognicore_full_adapter",
        "CognicoreFastAdapter": "cognicore_benchmarks.longmemeval.adapters.cognicore_fast_adapter",
        "CognicoreLongMemEvalV2": "cognicore_benchmarks.longmemeval.adapters.cognicore_v2_adapter",
        "CognicoreZeroShotAdapter": "cognicore_benchmarks.longmemeval.adapters.cognicore_zeroshot_adapter",
        "CognicoreMultiHopAdapter": "cognicore_benchmarks.longmemeval.adapters.cognicore_multihop_adapter"
    }
    
    if args.adapter not in adapter_map:
        raise ValueError(f"Unknown adapter {args.adapter}")
        
    import importlib
    module = importlib.import_module(adapter_map[args.adapter])
    adapter_cls = getattr(module, args.adapter)
    
    runner = LongMemEvalRunner(adapter_cls=adapter_cls, mode=args.mode)
    runner.run(limit=args.limit, resume_file=args.resume)
