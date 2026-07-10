import time
import json
import logging
import numpy as np

from cognicore_benchmarks.common.llm_client import LLMClient
from cognicore.memory.embedding_backend import BasicEmbeddingBackend
from cognicore.memory.providers.sentence_transformers import SentenceTransformerProvider
from cognicore.memory.providers.reranker import SentenceTransformerReranker
from cognicore.memory.base import MemoryEntry

def load_oracle():
    with open('cognicore_benchmarks/data/longmemeval/longmemeval_oracle.json') as f:
        return json.load(f)[:50]

def evaluate_system(use_reranker=False):
    oracle_data = load_oracle()
    
    # Setup
    provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
    reranker = SentenceTransformerReranker() if use_reranker else None
    memory = BasicEmbeddingBackend(provider=provider, reranker=reranker)
    
    metrics = {
        "mrr": 0.0,
        "recall@1": 0.0,
        "recall@5": 0.0,
        "retrieval_success_rate": 0.0,
        "latencies": [],
        "success_cases": [],
        "failures": []
    }
    
    total = len(oracle_data)
    
    for i, item in enumerate(oracle_data):
        # 1. Ingest up to question time
        memory.clear()
        
        chunk_size = 5
        seq_id = 0
        for session in item.get('haystack_sessions', []):
            for j in range(0, len(session), chunk_size):
                chunk = session[j:j+chunk_size]
                text_chunk = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chunk])
                mock_ts = float(seq_id * 3600)
                memory.store(MemoryEntry(text=text_chunk, category="raw_history_chunk", sequence_id=seq_id, timestamp=mock_ts))
                seq_id += 1
            
        # 2. Search
        question = item['question']
        answer = item['answer'].lower()
        
        t0 = time.time()
        results = memory.search(question, top_k=5, candidate_k=20 if use_reranker else 5)
        latency = time.time() - t0
        metrics['latencies'].append(latency)
        
        # Heuristic check for retrieval success
        import re
        q_words = set(re.findall(r'\b[A-Z][a-z]+\b', question))
        
        def chunk_contains_answer(chunk_text):
            # Same heuristic from failure attribution
            text_lower = chunk_text.lower()
            ans_words = [w for w in answer.split() if len(w) > 4]
            if ans_words and any(w in text_lower for w in ans_words):
                return True
            if q_words:
                matches = sum(1 for w in q_words if w.lower() in text_lower)
                if matches >= len(q_words) * 0.5:
                    return True
            return False
            
        found_at_rank = -1
        for rank, res in enumerate(results):
            if chunk_contains_answer(res.entry.text):
                found_at_rank = rank + 1
                break
                
        if found_at_rank > 0:
            if found_at_rank == 1:
                metrics['recall@1'] += 1
            if found_at_rank <= 5:
                metrics['recall@5'] += 1
            metrics['mrr'] += 1.0 / found_at_rank
            metrics['retrieval_success_rate'] += 1
            
            metrics['success_cases'].append({
                "query": question,
                "top_result": results[0].entry.text if results else "None",
                "rank": found_at_rank
            })
        else:
            metrics['failures'].append(item)
            
    # Normalize metrics
    for k in ['recall@1', 'recall@5', 'mrr', 'retrieval_success_rate']:
        metrics[k] /= total
        
    metrics['avg_latency'] = np.mean(metrics['latencies'])
    return metrics

if __name__ == "__main__":
    print("Evaluating System A (No Reranker)...")
    sys_a = evaluate_system(use_reranker=False)
    
    print("Evaluating System B (With Reranker)...")
    sys_b = evaluate_system(use_reranker=True)
    
    with open("results_a.json", "w") as f:
        json.dump(sys_a, f)
        
    with open("results_b.json", "w") as f:
        json.dump(sys_b, f)
        
    print("Done!")
