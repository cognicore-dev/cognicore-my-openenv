import json
from typing import Dict, Any, List
from cognicore_benchmarks.longmemeval.adapters.base_adapter import BaseAgentAdapter
from cognicore.memory.providers.sentence_transformers import SentenceTransformerProvider
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

class ChunkEntry:
    def __init__(self, text: str, timestamp: float, source_turns: List[int]):
        self.text = text
        self.timestamp = timestamp
        self.source_turns = source_turns

class CognicoreZeroShotAdapter(BaseAgentAdapter):
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.dense_provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # We store individual turns to build sliding windows dynamically
        self.turns = []
        
        # Hybrid search indices
        self.chunks = []
        self.chunk_embeddings = []
        self.bm25 = None
        self.is_index_dirty = True

    def _build_overlapping_chunks(self):
        import os
        window_size = int(os.environ.get("COGNICORE_WINDOW_SIZE", "35"))
        overlap = int(os.environ.get("COGNICORE_OVERLAP", "20"))
        step = window_size - overlap
        
        self.chunks = []
        
        if len(self.turns) == 0:
            return
            
        if len(self.turns) < window_size:
            text = "\n".join([f"{t['role']}: {t['content']}" for t in self.turns])
            self.chunks.append(ChunkEntry(text, self.turns[-1]["timestamp"], [i for i in range(len(self.turns))]))
        else:
            for i in range(len(self.turns) - window_size + 1):
                window = self.turns[i:i+window_size]
                text = "\n".join([f"{t['role']}: {t['content']}" for t in window])
                self.chunks.append(ChunkEntry(text, window[-1]["timestamp"], list(range(i, i+window_size))))
                
        # Also embed the chunks
        self.chunk_embeddings = np.array(self.dense_provider.embed_batch([c.text for c in self.chunks]))
        
        # Build BM25 index
        tokenized_corpus = [c.text.lower().split(" ") for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        self.is_index_dirty = False

    def reset(self):
        self.turns = []
        self.chunks = []
        self.chunk_embeddings = []
        self.bm25 = None
        self.is_index_dirty = True

    def ingest_history(self, session_data: List[Dict[str, Any]]):
        self.reset()
        for i, msg in enumerate(session_data):
            self.process_turn("default_session", i, float(i), "2023-01-01", msg)

    def process_turn(self, session_id: str, turn_idx: int, timestamp: float, session_date: str, msg: Dict[str, Any]):
        self.turns.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
            "timestamp": timestamp,
            "turn_idx": turn_idx
        })
        self.is_index_dirty = True

    import re

    def _split_query(self, query: str) -> List[str]:
        """Heuristically split a multi-hop query into sub-queries without an LLM."""
        sub_queries = [query]
        splitters = r'\b(and|or|vs|versus|compared to|before|after|between|first|second)\b'
        parts = self.re.split(splitters, query, flags=self.re.IGNORECASE)
        
        current_part = ""
        for p in parts:
            if p.lower() in ["and", "or", "vs", "versus", "compared to", "before", "after", "between", "first", "second"]:
                if current_part.strip() and len(current_part.strip()) > 4:
                    sub_queries.append(current_part.strip())
                current_part = ""
            else:
                current_part += p
        if current_part.strip() and len(current_part.strip()) > 4:
            sub_queries.append(current_part.strip())
            
        return list(set(sub_queries))

    def answer_question(self, question: str, question_timestamp: float = None) -> Dict[str, Any]:
        if self.is_index_dirty:
            self._build_overlapping_chunks()
            
        if not self.chunks:
            return {"answer": "", "retrieved_memories": [], "latency_s": 0, "tokens": 0}
            
        # 1. Heuristic Query Expansion
        if not hasattr(self, 're'):
            import re
            self.re = re
            
        import os
        if os.environ.get("COGNICORE_DISABLE_SUBQUERY", "0") == "1":
            sub_queries = [question]
            print("SUB-QUERY DISABLED")
        else:
            sub_queries = self._split_query(question)
            print(f"SUB-QUERIES: {len(sub_queries)}")
        
        pooled_indices = set()
        k = 60
        
        # 2. Parallel Hybrid Search for each sub-query
        subquery_pools = []
        for sq in sub_queries:
            # Sparse Search
            tokenized_sq = sq.lower().split(" ")
            bm25_scores = self.bm25.get_scores(tokenized_sq)
            bm25_ranked = np.argsort(bm25_scores)[::-1]
            
            # Dense Search
            sq_emb = np.array(self.dense_provider.embed(sq))
            dense_scores = np.dot(self.chunk_embeddings, sq_emb) / (
                np.linalg.norm(self.chunk_embeddings, axis=1) * np.linalg.norm(sq_emb) + 1e-9
            )
            dense_ranked = np.argsort(dense_scores)[::-1]
            
            # RRF
            rrf_scores = np.zeros(len(self.chunks))
            for rank, chunk_idx in enumerate(bm25_ranked):
                rrf_scores[chunk_idx] += 1.0 / (k + rank + 1)
            for rank, chunk_idx in enumerate(dense_ranked):
                rrf_scores[chunk_idx] += 1.0 / (k + rank + 1)
                
            top_sq_idx = np.argsort(rrf_scores)[::-1][:20]
            subquery_pools.append(list(top_sq_idx))
            pooled_indices.update(top_sq_idx)
            
        pooled_list = list(pooled_indices)
        
        # 3. Cross-Encoder Re-Ranking on the Master Pool
        cross_inp = [[question, self.chunks[idx].text] for idx in pooled_list]
        cross_scores = self.cross_encoder.predict(cross_inp)
        
        # Create a mapping of chunk_idx to its cross-encoder score
        score_map = {pooled_list[i]: float(cross_scores[i]) for i in range(len(pooled_list))}
        
        final_top_k_indices = []
        
        # DIVERSITY POOLING: Ensure the best chunk from EACH sub-query gets into the top 5!
        for sq_pool in subquery_pools:
            if len(final_top_k_indices) >= self.top_k:
                break
            # Find the highest cross-encoder scoring chunk in this specific sub-query's pool
            best_chunk_in_sq = max(sq_pool, key=lambda idx: score_map[idx])
            if best_chunk_in_sq not in final_top_k_indices:
                final_top_k_indices.append(best_chunk_in_sq)
                
        # Fill the remaining slots with the highest overall scoring chunks
        best_overall = sorted(pooled_list, key=lambda idx: score_map[idx], reverse=True)
        for idx in best_overall:
            if len(final_top_k_indices) >= self.top_k:
                break
            if idx not in final_top_k_indices:
                final_top_k_indices.append(idx)
        
        retrieved_memories = [self.chunks[idx].text for idx in final_top_k_indices]
        
        return {
            "answer": "N/A - ZeroShot Multi-Hop Local Eval",
            "latency_s": 0.0,
            "tokens": 0,
            "retrieved_memories": retrieved_memories,
            "ranking_scores": [score_map[idx] for idx in final_top_k_indices]
        }
