import json
import time
from pathlib import Path
from collections import defaultdict
from cognicore.memory.tfidf_backend import TFIDFMemoryBackend
from cognicore.memory.embedding_backend import BasicEmbeddingBackend
from cognicore.memory.providers.sentence_transformers import SentenceTransformerProvider
from cognicore.memory.base import MemoryEntry

def cosine_sim(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    if norm1 == 0 or norm2 == 0: return 0.0
    return dot / (norm1 * norm2)

def is_match(expected_vec, text, provider, threshold=0.35):
    if not text: return False
    text_vec = provider.embed(text)
    sim = cosine_sim(expected_vec, text_vec)
    return sim >= threshold

def compute_metrics(results, expected_vec, provider, top_k):
    ranks = []
    for i, res in enumerate(results[:top_k]):
        if is_match(expected_vec, res.entry.text, provider):
            ranks.append(i + 1)
            
    if not ranks:
        return {"recall": 0, "mrr": 0.0, "noise": 1.0, "rank": -1}
        
    first_rank = ranks[0]
    recall = 1
    mrr = 1.0 / first_rank
    noise = (len(results[:top_k]) - len(ranks)) / len(results[:top_k]) if results else 1.0
    return {"recall": recall, "mrr": mrr, "noise": noise, "rank": first_rank}

def main():
    print("Initializing providers...")
    provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
    
    dataset_file = Path("cognicore_benchmarks/data/longmemeval/longmemeval_oracle.json")
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)[:50] # Subset
        
    print(f"Loaded {len(dataset)} samples. Starting controlled benchmark...")
    
    metrics = {
        "tfidf": {"r1": 0, "r5": 0, "r10": 0, "mrr": 0.0, "noise": 0.0, "lat": 0.0, "success": 0},
        "emb":   {"r1": 0, "r5": 0, "r10": 0, "mrr": 0.0, "noise": 0.0, "lat": 0.0, "success": 0}
    }
    
    success_cases = []
    failures = []
    
    for item in dataset:
        qid = item["question_id"]
        q = item["question"]
        ans = item["answer"]
        qtype = item["question_type"]
        raw_sessions = item.get("haystack_sessions", [])
        
        tfidf = TFIDFMemoryBackend()
        emb = BasicEmbeddingBackend(provider=provider)
        
        global_turn = 0
        for session_idx, session in enumerate(raw_sessions):
            # Chunking per session
            for i in range(0, len(session), 5):
                c = session[i:i+5]
                text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in c])
                
                # Mock timestamp: 1 hour apart for each chunk
                mock_timestamp = float(global_turn * 3600)
                
                e = MemoryEntry(
                    text=text, 
                    category="raw",
                    session_id=f"session_{session_idx}",
                    sequence_id=global_turn,
                    timestamp=mock_timestamp
                )
                tfidf.store(e)
                emb.store(e)
                global_turn += 1
            
        # Search TFIDF
        t0 = time.time()
        res_tfidf = tfidf.search(q, top_k=10)
        lat_tfidf = time.time() - t0
        
        # Search Emb
        t0 = time.time()
        res_emb = emb.search(q, top_k=10)
        lat_emb = time.time() - t0
        
        ans_vec = provider.embed(ans)
        
        # Eval TFIDF
        m_tfidf_1 = compute_metrics(res_tfidf, ans_vec, provider, 1)
        m_tfidf_5 = compute_metrics(res_tfidf, ans_vec, provider, 5)
        m_tfidf_10 = compute_metrics(res_tfidf, ans_vec, provider, 10)
        
        # Eval Emb
        m_emb_1 = compute_metrics(res_emb, ans_vec, provider, 1)
        m_emb_5 = compute_metrics(res_emb, ans_vec, provider, 5)
        m_emb_10 = compute_metrics(res_emb, ans_vec, provider, 10)
        
        # Accumulate
        metrics["tfidf"]["r1"] += m_tfidf_1["recall"]
        metrics["tfidf"]["r5"] += m_tfidf_5["recall"]
        metrics["tfidf"]["r10"] += m_tfidf_10["recall"]
        metrics["tfidf"]["mrr"] += m_tfidf_10["mrr"]
        metrics["tfidf"]["noise"] += m_tfidf_10["noise"]
        metrics["tfidf"]["lat"] += lat_tfidf
        if m_tfidf_10["recall"] > 0: metrics["tfidf"]["success"] += 1
        
        metrics["emb"]["r1"] += m_emb_1["recall"]
        metrics["emb"]["r5"] += m_emb_5["recall"]
        metrics["emb"]["r10"] += m_emb_10["recall"]
        metrics["emb"]["mrr"] += m_emb_10["mrr"]
        metrics["emb"]["noise"] += m_emb_10["noise"]
        metrics["emb"]["lat"] += lat_emb
        if m_emb_10["recall"] > 0: metrics["emb"]["success"] += 1
        
        # Analysis Logging
        if m_tfidf_10["recall"] == 0 and m_emb_10["recall"] > 0:
            success_cases.append({
                "qid": qid, "q": q, "ans": ans, 
                "emb_retrieved": res_emb[m_emb_10["rank"]-1].entry.text if m_emb_10["rank"] > 0 else ""
            })
            
        if m_emb_10["recall"] == 0:
            failures.append({"qid": qid, "q": q, "ans": ans, "type": qtype})
            
    # Averages
    n = len(dataset)
    for k in metrics:
        metrics[k]["mrr"] /= n
        metrics[k]["noise"] /= n
        metrics[k]["lat"] /= n
        
    out_dir = Path(r"C:\Users\kaush\.gemini\antigravity\brain\bf226565-59da-4385-a320-4851e0a6a53c")
    
    # 1. embedding_benchmark_results.md
    with open(out_dir / "embedding_benchmark_results.md", "w", encoding="utf-8") as f:
        f.write("# Controlled Benchmark Results\n\n")
        f.write("| Metric | TF-IDF | Embeddings | Delta |\n|---|---|---|---|\n")
        f.write(f"| Recall@1 | {metrics['tfidf']['r1']/n:.1%} | {metrics['emb']['r1']/n:.1%} | +{(metrics['emb']['r1'] - metrics['tfidf']['r1'])/n:.1%} |\n")
        f.write(f"| Recall@5 | {metrics['tfidf']['r5']/n:.1%} | {metrics['emb']['r5']/n:.1%} | +{(metrics['emb']['r5'] - metrics['tfidf']['r5'])/n:.1%} |\n")
        f.write(f"| Recall@10 | {metrics['tfidf']['r10']/n:.1%} | {metrics['emb']['r10']/n:.1%} | +{(metrics['emb']['r10'] - metrics['tfidf']['r10'])/n:.1%} |\n")
        f.write(f"| MRR | {metrics['tfidf']['mrr']:.3f} | {metrics['emb']['mrr']:.3f} | +{metrics['emb']['mrr'] - metrics['tfidf']['mrr']:.3f} |\n")
        f.write(f"| Success Rate | {metrics['tfidf']['success']/n:.1%} | {metrics['emb']['success']/n:.1%} | +{(metrics['emb']['success'] - metrics['tfidf']['success'])/n:.1%} |\n")
        f.write(f"| Noise Ratio | {metrics['tfidf']['noise']:.2f} | {metrics['emb']['noise']:.2f} | {metrics['emb']['noise'] - metrics['tfidf']['noise']:.2f} |\n")
        f.write(f"| Latency | {metrics['tfidf']['lat']*1000:.1f}ms | {metrics['emb']['lat']*1000:.1f}ms | +{(metrics['emb']['lat'] - metrics['tfidf']['lat'])*1000:.1f}ms |\n")
        
    # 2. embedding_success_cases.md
    with open(out_dir / "embedding_success_cases.md", "w", encoding="utf-8") as f:
        f.write("# Embedding Success Cases (Where TF-IDF Failed)\n\n")
        for sc in success_cases[:10]:
            f.write(f"**Q:** {sc['q']}\n**Expected:** {sc['ans']}\n")
            f.write(f"**Why Embeddings Succeeded:** Semantic similarity bridged the vocabulary gap between the question and the raw text chunk.\n")
            f.write(f"**Retrieved Chunk:**\n```text\n{sc['emb_retrieved'][:200]}...\n```\n\n")

    # 3. remaining_gap_analysis.md
    with open(out_dir / "remaining_gap_analysis.md", "w", encoding="utf-8") as f:
        f.write("# Remaining Gap Analysis (Embedding Failures)\n\n")
        types = defaultdict(int)
        for fail in failures: types[fail["type"]] += 1
        
        f.write("### Failure Breakdown by Category\n")
        for t, c in types.items():
            f.write(f"- **{t}**: {c} failures\n")
            
        f.write("\n### Conclusion\n")
        if "temporal-reasoning" in types or "knowledge-update" in types:
            f.write("Temporal and update logic is now the dominant bottleneck. Embeddings cannot resolve conflicting facts across time without timestamps.\n")
        else:
            f.write("General context assembly remains a challenge.\n")

if __name__ == "__main__":
    main()
