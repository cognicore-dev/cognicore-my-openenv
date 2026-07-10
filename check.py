import json
from cognicore_benchmarks.longmemeval.metrics import evaluate_metrics

with open('cognicore_benchmarks/data/longmemeval/longmemeval_oracle.json') as f:
    oracle = {item['question_id']: item for item in json.load(f)}

with open('results/longmemeval/OFFICIAL_CognicoreFastAdapter_predictions_20260619T115949Z.jsonl') as f:
    preds = [json.loads(l) for l in f]

correct = 0
total = 0
for p in preds:
    if "API Error" in p['hypothesis']:
        continue
    qid = p['question_id']
    ref = oracle[qid]
    score = evaluate_metrics(p['hypothesis'], ref['answer'])['score']
    correct += score
    total += 1

print(f"Evaluated {total} successful LLM generations.")
print(f"Heuristic Accuracy: {correct/total*100:.2f}%")
