import json
from cognicore_benchmarks.longmemeval.metrics import evaluate_metrics

with open('cognicore_benchmarks/data/longmemeval/longmemeval_oracle.json') as f:
    oracle = {item['question_id']: item for item in json.load(f)}

with open('results/longmemeval/OFFICIAL_CognicoreFastAdapter_predictions_20260619T115949Z.jsonl') as f:
    preds = [json.loads(l) for l in f]

failures = []
for p in preds:
    if "API Error" in p['hypothesis']:
        continue
    qid = p['question_id']
    ref = oracle[qid]
    
    score = evaluate_metrics(p['hypothesis'], ref['answer'])['score']
    
    if score == 0:
        failures.append({
            "question": ref['question'],
            "type": ref['question_type'],
            "retrieved_memories": p.get('retrieved_memories', []),
            "hypothesis": p['hypothesis'],
            "answer": ref['answer']
        })

# Auto-classify based on heuristic keywords for the report
types_count = {
    "Retrieval Failure": 0,
    "Ranking Failure": 0,
    "Context Assembly Failure": 0,
    "Temporal Reasoning Failure": 0,
    "Knowledge Update Failure": 0,
    "Multi-Hop Reasoning Failure": 0,
    "LLM Reasoning Failure": 0,
    "Other": 0
}

report = "# Failure Attribution Report\n\n"

for f in failures:
    q = f['question']
    t = f['type']
    ans = f['answer']
    
    # Check if the correct answer or required dates are in the retrieved memories
    found_in_mem = False
    
    # We don't have the exact source chunks the oracle expects, but we can do a heuristic check.
    # If the LLM output is wrong, we want to know if the retriever failed to grab the facts, 
    # or if the LLM failed to synthesize them.
    # Let's check if the raw retrieved text contains keywords from the question or answer.
    # To be precise, if the ranking/retrieval missed the core entities, it's a Ranking Failure.
    # For temporal math ("how many days"), the answer is a number, but the retrieved text has dates.
    # So we can't just check if "14" is in the text. We check if the entities from the question are in the text.
    
    # A simple heuristic: Did the retrieved text contain the core nouns of the question?
    import re
    q_words = set(re.findall(r'\b[A-Z][a-z]+\b', q)) # Proper nouns
    
    if q_words:
        matches = sum(1 for w in q_words if any(w.lower() in m.lower() for m in f['retrieved_memories']))
        if matches < len(q_words) * 0.5: # Less than 50% of proper nouns found -> Retrieval/Ranking Failed
            classification = "Ranking Failure"
        else:
            classification = "LLM Reasoning / Context Assembly Failure"
    else:
        classification = "Ranking Failure" # Fallback
            
    types_count[classification] += 1
    
    report += f"### Q: {q}\n"
    report += f"- **Type**: {t}\n"
    report += f"- **Classification**: {classification}\n"
    report += f"- **Correct Answer**: {ans}\n\n"
    
report += "## Summary\n\n"
report += "| Failure Type | Count | Percent |\n"
report += "| --- | --- | --- |\n"
total_fails = len(failures)
for k, v in types_count.items():
    if v > 0:
        pct = (v / total_fails) * 100
        report += f"| {k} | {v} | {pct:.1f}% |\n"
        
with open("C:/Users/kaush/.gemini/antigravity/brain/bf226565-59da-4385-a320-4851e0a6a53c/failure_attribution_report.md", "w", encoding="utf-8") as f:
    f.write(report)
