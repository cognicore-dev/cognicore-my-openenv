import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter
from cognicore_benchmarks.common.llm_client import LLMClient

PROMPT_TEMPLATE = """You are a Principal AI Research Engineer diagnosing benchmark failures.

We are evaluating a memory agent on the LongMemEval benchmark.
The agent failed the following question.

Question Type: {question_type}
Question: {question}
Expected Answer: {expected}
Agent Final Hypothesis: {hypothesis}

Deep Telemetry:
Retrieved Contexts: 
{retrieved}

Draft Answer (Before Reflection): {draft}
Reflection Hint: {hint}

Your task is to classify this failure into exactly ONE primary category:
1. Memory Missing (Required memory was never stored in long-term history)
2. Retrieval Failure (Memory existed in long-term history, but retrieved context is completely wrong/empty)
3. Ranking Failure (Memory retrieved but ranked too low / drowned by noise)
4. Context Assembly Failure (Correct memory retrieved but context length/format caused failure)
5. Temporal Reasoning Failure (Retrieved info but failed to reason about time/order)
6. Knowledge Update Failure (Failed to handle updated or contradictory information)
7. Preference Resolution Failure (Failed to recall or apply user preferences)
8. Reflection Failure (Reflection introduced incorrect guidance or hallucinatory self-correction)
9. LLM Reasoning Failure (Correct context retrieved, but LLM failed to infer the answer)
10. Other

You must also answer these binary questions:
- memory_existed: Did the required information likely exist in the user's history?
- retrieved: Was the information present in the 'Retrieved Contexts' above?
- ranked: Was the critical information prominent in the retrieved contexts?
- reasoned: Did the agent fail at basic logic despite having the right context?

Output strictly in JSON format:
{
  "category_id": 1-10,
  "category_name": "...",
  "reasoning": "...",
  "memory_existed": true/false,
  "retrieved": true/false,
  "ranked": true/false,
  "reasoned": true/false
}
"""

def main(eval_file: str):
    eval_path = Path(eval_file)
    if not eval_path.exists():
        print(f"File not found: {eval_path}")
        sys.exit(1)
        
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    results = data.get("results", [])
    failures = [r for r in results if r.get("score", 0.0) == 0.0]
    
    print(f"Found {len(failures)} failures out of {len(results)} evaluated samples.")
    
    client = LLMClient(model_name="openai/gpt-4o-mini")
    if client.is_mock:
        print("ERROR: Requires valid OPENROUTER_API_KEY")
        sys.exit(1)
        
    audit_log = []
    
    for i, failure in enumerate(failures):
        print(f"Analyzing failure {i+1}/{len(failures)} (ID: {failure['question_id']})...")
        
        telemetry = failure
        retrieved = telemetry.get("retrieved_memories", [])
        retrieved_text = "\n".join([f"- {r}" for r in retrieved]) if retrieved else "NONE"
        
        prompt = PROMPT_TEMPLATE.format(
            question_type=failure.get("question_type", "unknown"),
            question=failure.get("question", "unknown"),
            expected=failure.get("expected", "unknown"), # wait, eval JSON doesn't store expected answer currently!
            hypothesis=failure.get("hypothesis", "unknown"),
            retrieved=retrieved_text,
            draft=telemetry.get("draft_answer", "NONE"),
            hint=telemetry.get("reflection_hint", "NONE")
        )
        
    # Load references to get the expected answer and original question
    dataset_path = Path("cognicore_benchmarks/data/longmemeval/longmemeval_oracle.json")
    references = {}
    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            references = {item["question_id"]: item for item in raw_data}

    for i, failure in enumerate(failures):
        qid = failure['question_id']
        print(f"Analyzing failure {i+1}/{len(failures)} (ID: {qid})...")
        
        telemetry = failure
        retrieved = telemetry.get("retrieved_memories", [])
        retrieved_text = "\n".join([f"- {r}" for r in retrieved]) if retrieved else "NONE"
        
        ref = references.get(qid, {})
        expected_answer = ref.get("answer", "UNKNOWN")
        question_text = ref.get("question", failure.get("question_id"))
        
        prompt = PROMPT_TEMPLATE.format(
            question_type=failure.get("question_type", "unknown"),
            question=question_text,
            expected=expected_answer,
            hypothesis=failure.get("hypothesis", "unknown"),
            retrieved=retrieved_text,
            draft=telemetry.get("draft_answer", "NONE"),
            hint=telemetry.get("reflection_hint", "NONE")
        )
        
        res = client.generate(prompt=prompt, system_prompt="Output only valid JSON.")
        
        try:
            # Strip markdown code blocks if present
            content = res["content"].strip()
            if content.startswith("```json"): content = content[7:]
            if content.endswith("```"): content = content[:-3]
            analysis = json.loads(content)
        except json.JSONDecodeError:
            print(f"  [!] Failed to parse LLM response: {res['content']}")
            analysis = {"category_id": 10, "category_name": "Parse Error", "reasoning": "Failed to parse JSON."}
            
        failure_record = {
            "question_id": qid,
            "analysis": analysis,
            "telemetry": telemetry
        }
        audit_log.append(failure_record)
        
    # Generate Reports
    out_dir = eval_path.parent
    
    # 1. retrieval_audit.json
    with open(out_dir / "retrieval_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2)
        
    # 2. benchmark_gap_analysis.md
    categories = Counter([r["analysis"].get("category_name", "Unknown") for r in audit_log])
    
    md = "# CogniCore Benchmark Gap Analysis\n\n"
    md += f"**Total Failures Analyzed:** {len(failures)}\n\n"
    
    md += "## Error Distribution\n"
    for cat, count in categories.most_common():
        pct = count / len(failures) * 100
        md += f"- **{cat}**: {count} ({pct:.1f}%)\n"
        
    md += "\n## Top Failure Modes\n"
    for r in audit_log[:5]:
        md += f"### {r['analysis'].get('category_name')} (ID: {r['question_id']})\n"
        md += f"- **Reasoning:** {r['analysis'].get('reasoning')}\n"
        
    with open(out_dir / "benchmark_gap_analysis.md", "w", encoding="utf-8") as f:
        f.write(md)
        
    print(f"\nAnalysis complete. Generated {out_dir / 'benchmark_gap_analysis.md'} and retrieval_audit.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, required=True)
    args = parser.parse_args()
    main(args.eval_file)
