import json
import time
from cognicore.memory.base import MemoryEntry
from cognicore.memory.embedding_backend import BasicEmbeddingBackend
from cognicore.memory.providers.sentence_transformers import SentenceTransformerProvider

def run_synthetic_tests():
    provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
    backend = BasicEmbeddingBackend(provider=provider)
    
    # Generate Synthetic Dataset
    print("Generating Synthetic Tests...")
    
    # 1. Knowledge Update
    e1 = MemoryEntry(text="User lives in Mumbai", memory_type="fact", timestamp=1000)
    id1 = backend.store(e1)
    
    e2 = MemoryEntry(text="User lives in Pune", memory_type="fact", timestamp=5000, supersedes=id1)
    id2 = backend.store(e2)
    
    # 2. Preference Decay
    e3 = MemoryEntry(text="Favorite language is Python", memory_type="preference", timestamp=time.time() - 60*24*3600) # 60 days ago
    id3 = backend.store(e3)
    
    e4 = MemoryEntry(text="Favorite language is Rust", memory_type="preference", timestamp=time.time()) # Now
    id4 = backend.store(e4)
    
    # Run tests
    results = []
    
    # Test 1: Where does user live?
    res1 = backend.search("Where does the user currently live?", top_k=5)
    filtered_1 = [r for r in res1 if "Mumbai" in r.entry.text]
    valid_1 = [r for r in res1 if "Pune" in r.entry.text]
    
    results.append({
        "test": "Knowledge Update (Mumbai -> Pune)",
        "retrieved": [r.entry.text for r in res1],
        "passed": len(filtered_1) == 0 and len(valid_1) > 0
    })
    
    # Test 2: Favorite Language
    res2 = backend.search("What is my current favorite language?", top_k=5)
    
    rust_score = next((r.score for r in res2 if "Rust" in r.entry.text), 0)
    py_score = next((r.score for r in res2 if "Python" in r.entry.text), 0)
    
    results.append({
        "test": "Preference Update (Python -> Rust)",
        "retrieved": [r.entry.text for r in res2],
        "passed": rust_score > py_score
    })
    
    # Generate validation report
    with open(r"C:\Users\kaush\.gemini\antigravity\brain\bf226565-59da-4385-a320-4851e0a6a53c\temporal_validation_report.md", "w") as f:
        f.write("# Temporal Engine Audit\n\n")
        for r in results:
            f.write(f"### {r['test']}\n")
            f.write(f"- Passed: **{r['passed']}**\n")
            f.write(f"- Final Context Payload:\n")
            for t in r['retrieved']:
                f.write(f"  - {t}\n")
            f.write("\n")

def run_longmemeval_comparison():
    print("Running LongMemEval Comparison...")
    dataset_path = "cognicore_benchmarks/data/longmemeval/longmemeval_oracle.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)[:50]
        
    provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
    emb_backend = BasicEmbeddingBackend(provider=provider)
    
    # Ingest
    for item in dataset:
        raw_sessions = item.get("haystack_sessions", [])
        global_turn = 0
        for session_idx, session in enumerate(raw_sessions):
            for i in range(0, len(session), 5):
                c = session[i:i+5]
                text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in c])
                
                mock_ts = float(global_turn * 3600)
                e = MemoryEntry(
                    text=text, 
                    category="raw",
                    session_id=f"session_{session_idx}",
                    sequence_id=global_turn,
                    timestamp=mock_ts
                )
                emb_backend.store(e)
                global_turn += 1
                
    success_baseline = 0
    success_temporal = 0
    failures = []
    
    t0 = time.time()
    for item in dataset:
        q = item["question"]
        ans = item["answer"]
        
        query_vec = provider.embed(q)
        n = sum(x*x for x in query_vec)**0.5
        if n>0: query_vec = [x/n for x in query_vec]
        
        raw_results = []
        for entry, vector in zip(emb_backend.entries, emb_backend.vectors):
            score = sum(a * b for a, b in zip(query_vec, vector))
            raw_results.append((entry, score))
            
        raw_results.sort(key=lambda x: x[1], reverse=True)
        top_10_raw = raw_results[:10]
        
        top_10_temp = emb_backend.search(q, top_k=10)
        
        ans_vec = provider.embed(ans)
        na = sum(x*x for x in ans_vec)**0.5
        if na>0: ans_vec = [x/na for x in ans_vec]
        
        def is_match(results_list, get_text_func):
            for r in results_list:
                text = get_text_func(r)
                tv = provider.embed(text)
                nt = sum(x*x for x in tv)**0.5
                if nt>0: tv = [x/nt for x in tv]
                sim = sum(a*b for a,b in zip(ans_vec, tv))
                if sim >= 0.35:
                    return True
            return False
            
        b_match = is_match(top_10_raw, lambda x: x[0].text)
        t_match = is_match(top_10_temp, lambda x: x.entry.text)
        
        if b_match: success_baseline += 1
        if t_match: success_temporal += 1
        
        if not t_match:
            failures.append({
                "question": q,
                "type": item["question_type"],
                "reason": "temporal resolution" if item["question_type"] in ["temporal-reasoning", "knowledge-update", "preference-update"] else "retrieval"
            })
            
    lat = time.time() - t0
    
    with open(r"C:\Users\kaush\.gemini\antigravity\brain\bf226565-59da-4385-a320-4851e0a6a53c\temporal_benchmark_comparison.md", "w") as f:
        f.write("# Temporal Benchmark Comparison\n\n")
        f.write(f"**Baseline (Embeddings Only) Success:** {success_baseline / len(dataset) * 100:.1f}%\n")
        f.write(f"**Temporal (Embeddings + Engine) Success:** {success_temporal / len(dataset) * 100:.1f}%\n")
        f.write(f"**Latency:** {lat * 1000 / len(dataset):.1f}ms\n")
        
    with open(r"C:\Users\kaush\.gemini\antigravity\brain\bf226565-59da-4385-a320-4851e0a6a53c\remaining_temporal_failures.md", "w") as f:
        f.write("# Remaining Failures\n\n")
        for fail in failures:
            f.write(f"- [{fail['type']}] {fail['question']} -> {fail['reason']}\n")

if __name__ == "__main__":
    run_synthetic_tests()
    run_longmemeval_comparison()
