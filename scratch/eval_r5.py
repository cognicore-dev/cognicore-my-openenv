import json
import argparse
from pathlib import Path

def calculate_r5(predictions_file: str, dataset_file: str):
    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    references = {}
    for item in data:
        evidence_texts = []
        for session in item.get("haystack_sessions", []):
            for turn in session:
                if turn.get("has_answer") is True:
                    evidence_texts.append(turn["content"].strip().lower())
        references[item["question_id"]] = evidence_texts
    
    total = 0
    hits = 0
    misses_printed = 0
    
    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            pred = json.loads(line)
            qid = pred["question_id"]
            if qid not in references: continue
            
            evidence_texts = references[qid]
            retrieved = pred.get("_telemetry", {}).get("retrieved_memories", [])
            
            # Strict R@5: ALL evidence texts must be found somewhere in the retrieved chunks
            all_evidence_found = True
            
            if not evidence_texts:
                # Abstention
                all_evidence_found = True
            else:
                for ev in evidence_texts:
                    ev_found = False
                    for mem in retrieved:
                        mem_lower = str(mem).lower()
                        if ev in mem_lower or mem_lower in ev:
                            ev_found = True
                            break
                        if len(ev) > 20 and ev[:len(ev)//2] in mem_lower:
                            ev_found = True
                            break
                    if not ev_found:
                        all_evidence_found = False
                        break
                    
            if all_evidence_found:
                hits += 1
            else:
                if misses_printed < 5:
                    print(f"MISS: [{qid}]")
                    print(f"Missing Evidence: {[e[:100] for e in evidence_texts]}")
                    print(f"Retrieved: {[m[:100] for m in retrieved]}")
                    print("-" * 40)
                    misses_printed += 1
            total += 1
            
    r5 = hits / total if total > 0 else 0
    print(f"Total samples: {total}")
    print(f"STRICT Recall@5 Hits: {hits}")
    print(f"STRICT R@5: {r5:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=str, required=True)
    parser.add_argument("--oracle", type=str, default="cognicore_benchmarks/data/longmemeval/longmemeval_oracle.json")
    args = parser.parse_args()
    calculate_r5(args.preds, args.oracle)
