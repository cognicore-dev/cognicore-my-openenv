import json
import os
import re
from typing import Dict, Any, List, Set, Tuple
from cognicore_benchmarks.longmemeval.adapters.base_adapter import BaseAgentAdapter
from cognicore.memory.providers.sentence_transformers import SentenceTransformerProvider
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
import nltk

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

class ChunkEntry:
    def __init__(self, idx: int, text: str, timestamp: float, source_turns: List[int], session_id: str):
        self.idx = idx
        self.text = text
        self.timestamp = timestamp
        self.source_turns = source_turns
        self.session_id = session_id

class Target:
    def __init__(self, text: str, type_name: str):
        self.text = text
        self.type_name = type_name  # 'entity', 'event', 'temporal', 'comparison'

    def __repr__(self):
        return f"[{self.type_name}] {self.text}"

class CognicoreMultiHopAdapter(BaseAgentAdapter):
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.dense_provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        self.turns = []
        self.chunks = []
        self.chunk_embeddings = None
        self.bm25 = None
        
        # Memory Graph: Chunk ID -> List of neighbor Chunk IDs
        self.graph_next = {}
        self.graph_prev = {}
        self.graph_session = {}
        
        self.is_index_dirty = True

    def reset(self):
        self.turns = []
        self.chunks = []
        self.chunk_embeddings = None
        self.bm25 = None
        self.graph_next = {}
        self.graph_prev = {}
        self.graph_session = {}
        self.is_index_dirty = True

    def ingest_history(self, session_data: List[Dict[str, Any]]):
        self.reset()
        for i, msg in enumerate(session_data):
            # For simplicity in testing, assign default session if missing
            sess_id = msg.get("session_id", "default_session")
            self.process_turn(sess_id, i, float(i), "2023-01-01", msg)

    def process_turn(self, session_id: str, turn_idx: int, timestamp: float, session_date: str, msg: Dict[str, Any]):
        self.turns.append({
            "session_id": session_id,
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
            "timestamp": timestamp,
            "turn_idx": turn_idx
        })
        self.is_index_dirty = True

    def _build_index(self):
        # We explicitly use small windows as requested for this adapter.
        window_size = int(os.environ.get("COGNICORE_WINDOW_SIZE", "5"))
        overlap = int(os.environ.get("COGNICORE_OVERLAP", "2"))
        
        self.chunks = []
        
        if len(self.turns) == 0:
            return
            
        if len(self.turns) < window_size:
            text = "\n".join([f"{t['role']}: {t['content']}" for t in self.turns])
            self.chunks.append(ChunkEntry(0, text, self.turns[-1]["timestamp"], [i for i in range(len(self.turns))], self.turns[0]["session_id"]))
        else:
            idx = 0
            # Step size guarantees overlap
            step = max(1, window_size - overlap)
            for i in range(0, len(self.turns) - window_size + 1, step):
                window = self.turns[i:i+window_size]
                text = "\n".join([f"{t['role']}: {t['content']}" for t in window])
                sess_id = window[-1]["session_id"] # Approximate session
                self.chunks.append(ChunkEntry(idx, text, window[-1]["timestamp"], list(range(i, i+window_size)), sess_id))
                idx += 1
                
        if len(self.chunks) == 0:
            return

        # Embeddings
        self.chunk_embeddings = np.array(self.dense_provider.embed_batch([c.text for c in self.chunks]))
        
        # BM25
        tokenized_corpus = [c.text.lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Build Graph
        self.graph_next = {c.idx: [] for c in self.chunks}
        self.graph_prev = {c.idx: [] for c in self.chunks}
        self.graph_session = {c.idx: [] for c in self.chunks}
        
        # O(N) linear connections
        for i in range(len(self.chunks) - 1):
            if self.chunks[i].session_id == self.chunks[i+1].session_id:
                self.graph_next[self.chunks[i].idx].append(self.chunks[i+1].idx)
                self.graph_prev[self.chunks[i+1].idx].append(self.chunks[i].idx)
        
        self.is_index_dirty = False

    def _extract_targets(self, query: str) -> List[Target]:
        """A heuristic NLP-based analyzer to extract evidence targets from the question."""
        targets = []
        
        # 1. Temporal / Update targets
        temporal_keywords = r'\b(before|after|used to|now|currently|recently|first|last|new)\b'
        matches = re.findall(temporal_keywords, query, re.IGNORECASE)
        for m in matches:
            targets.append(Target(m, "temporal"))
            
        # 2. Preference / Change targets
        pref_keywords = r'\b(prefer|favorite|like|dislike|changed|updated)\b'
        matches = re.findall(pref_keywords, query, re.IGNORECASE)
        for m in matches:
            targets.append(Target(m, "preference"))
            
        # 3. Noun Phrases / Entities using NLTK
        words = nltk.word_tokenize(query)
        tagged = nltk.pos_tag(words)
        grammar = "NP: {<NN.*|JJ.*>*<NN.*>}"
        cp = nltk.RegexpParser(grammar)
        tree = cp.parse(tagged)
        
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                np_phrase = " ".join([word for word, pos in subtree.leaves()])
                # Ignore generic question words
                if np_phrase.lower() not in ["what", "who", "which", "how", "why"]:
                    targets.append(Target(np_phrase, "entity"))
                    
        # 4. If nothing was extracted, fallback to the query itself
        if not targets:
            targets.append(Target(query, "entity"))
            
        return targets

    def _hybrid_search(self, query: str, top_k: int = 10) -> List[int]:
        """Runs Dense + Sparse search for a query and returns chunk indices."""
        tokenized_sq = query.lower().split()
        if not tokenized_sq:
            return []
        bm25_scores = self.bm25.get_scores(tokenized_sq)
        bm25_ranked = np.argsort(bm25_scores)[::-1]
        
        sq_emb = np.array(self.dense_provider.embed(query))
        dense_scores = np.dot(self.chunk_embeddings, sq_emb) / (
            np.linalg.norm(self.chunk_embeddings, axis=1) * np.linalg.norm(sq_emb) + 1e-9
        )
        dense_ranked = np.argsort(dense_scores)[::-1]
        
        k_rrf = 60
        rrf_scores = np.zeros(len(self.chunks))
        for rank, chunk_idx in enumerate(bm25_ranked):
            rrf_scores[chunk_idx] += 1.0 / (k_rrf + rank + 1)
        for rank, chunk_idx in enumerate(dense_ranked):
            rrf_scores[chunk_idx] += 1.0 / (k_rrf + rank + 1)
            
        return list(np.argsort(rrf_scores)[::-1][:top_k])

    def answer_question(self, question: str, question_timestamp: float = None) -> Dict[str, Any]:
        if self.is_index_dirty:
            self._build_index()
            
        if not self.chunks:
            return {"answer": "", "retrieved_memories": [], "latency_s": 0, "tokens": 0}
            
        # --- A. Decompose Question into Evidence Targets ---
        targets = self._extract_targets(question)
        
        # Log to stdout for tracing
        # print(f"DEBUG TARGETS: {targets}")
        
        # --- B. Hop-1 Retrieval ---
        hop1_candidates = set()
        target_to_chunks = {}
        
        for tgt in targets:
            # Emphasize the target but keep context of the question
            hop_query = f"{tgt.text} {question}" if tgt.type_name == "entity" else question
            retrieved_idx = self._hybrid_search(hop_query, top_k=10)
            target_to_chunks[tgt] = retrieved_idx
            hop1_candidates.update(retrieved_idx)
            
        # --- C & D. Hop-2 Retrieval (Graph Traversal) ---
        hop2_candidates = set()
        
        # If the question asks for an update/change, we traverse forward in time from entities
        has_temporal = any(t.type_name in ["temporal", "preference"] for t in targets)
        
        if has_temporal:
            # Take top chunks and expand forwards and backwards
            for idx in list(hop1_candidates):
                # Expand +1 and +2 next
                curr = idx
                for _ in range(2):
                    if curr in self.graph_next and self.graph_next[curr]:
                        curr = self.graph_next[curr][0]
                        hop2_candidates.add(curr)
                    else:
                        break
                        
                # Expand -1 and -2 prev
                curr = idx
                for _ in range(2):
                    if curr in self.graph_prev and self.graph_prev[curr]:
                        curr = self.graph_prev[curr][0]
                        hop2_candidates.add(curr)
                    else:
                        break
                        
        all_candidates = list(hop1_candidates.union(hop2_candidates))
        if not all_candidates:
            # Fallback
            all_candidates = self._hybrid_search(question, top_k=20)
            
        # --- E. Coverage-Aware Final Selection ---
        # Evaluate Cross-Encoder scores for all candidates against the original question
        cross_inp = [[question, self.chunks[idx].text] for idx in all_candidates]
        cross_scores = self.cross_encoder.predict(cross_inp)
        
        # Map idx -> score
        chunk_score_map = {all_candidates[i]: float(cross_scores[i]) for i in range(len(all_candidates))}
        
        # Calculate coverage for each chunk: which targets does it satisfy?
        # We assume a chunk satisfies a target if the chunk has a high cross-encoder score vs the specific target.
        # But to save compute, we can just check if the target's text is in the chunk, OR if it was retrieved for that target.
        chunk_coverage = {idx: set() for idx in all_candidates}
        
        for tgt in targets:
            if tgt.type_name == "entity":
                # Find chunks that explicitly mention the entity
                tgt_words = set(tgt.text.lower().split())
                for idx in all_candidates:
                    text_words = set(self.chunks[idx].text.lower().split())
                    # If strong overlap
                    if len(tgt_words.intersection(text_words)) > 0:
                        chunk_coverage[idx].add(tgt.text)
                        
            # Also add provenance from hop-1
            if tgt in target_to_chunks:
                for idx in target_to_chunks[tgt]:
                    if idx in chunk_coverage:
                        chunk_coverage[idx].add(tgt.text)
        
        # Greedy Selection
        final_top_k = []
        covered_targets = set()
        
        # We need to pick top_k chunks.
        remaining_candidates = list(all_candidates)
        
        while len(final_top_k) < self.top_k and remaining_candidates:
            best_chunk = None
            best_score = -9999
            
            for idx in remaining_candidates:
                # Calculate marginal coverage
                marginal_cov = len(chunk_coverage[idx] - covered_targets)
                # Primary sort by marginal coverage, secondary sort by relevance score
                # Weight relevance heavily so we don't pick garbage just because it has a word
                score = (marginal_cov * 10.0) + chunk_score_map[idx]
                
                if score > best_score:
                    best_score = score
                    best_chunk = idx
                    
            if best_chunk is not None:
                final_top_k.append(best_chunk)
                covered_targets.update(chunk_coverage[best_chunk])
                remaining_candidates.remove(best_chunk)
            else:
                break
                
        retrieved_memories = [self.chunks[idx].text for idx in final_top_k]
        
        # --- F. Generate Answer using LLM ---
        from cognicore_benchmarks.common.llm_client import LLMClient
        model_name = os.environ.get("COGNICORE_EVAL_MODEL", "gemini-3.1-pro-preview")
        client = LLMClient(model_name=model_name)
        
        context_text = "\n\n".join(retrieved_memories)
        prompt = (
            f"Here is the retrieved conversation history:\n\n"
            f"{context_text}\n\n"
            f"Based on the history above, please answer the following question. "
            f"If the information is not in the history, say 'I don't know'.\n\n"
            f"Question: {question}"
        )
        system_prompt = "You are a helpful assistant evaluating a conversation log."
        
        result = client.generate(prompt=prompt, system_prompt=system_prompt)
        
        answer = result.get("content", "")
        if not answer or "error" in result:
            # Fallback when no API key is available
            answer = f"[RAW_RETRIEVAL_FALLBACK]\n{context_text}"
            
        return {
            "answer": answer,
            "latency_s": result.get("latency_s", 0),
            "tokens": result.get("prompt_tokens", 0) + result.get("completion_tokens", 0),
            "retrieved_memories": retrieved_memories,
            "ranking_scores": [chunk_score_map[idx] for idx in final_top_k]
        }
