from typing import Dict, Any, List
import time
import re
from datetime import datetime
from cognicore_benchmarks.longmemeval.adapters.base_adapter import BaseAgentAdapter
from cognicore_benchmarks.common.llm_client import LLMClient
from cognicore.memory.embedding_backend import BasicEmbeddingBackend
from cognicore.memory.providers.sentence_transformers import SentenceTransformerProvider
from cognicore.memory.providers.reranker import SentenceTransformerReranker
from cognicore.memory.base import MemoryEntry

class CognicoreFastAdapter(BaseAgentAdapter):
    """
    Mode: Fast/Low-Cost Retrieval.
    Uses Semantic Memory + Temporal Filter + Cross-Encoder Reranker, but does ONLY ONE LLM generation pass.
    Drastically reduces token consumption compared to FullAdapter and BaselineAdapter.
    """
    def __init__(self, model_name: str = "openai/gpt-4o-mini", top_k: int = 5):
        self.client = LLMClient(model_name=model_name)
        provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
        reranker = SentenceTransformerReranker()
        self.memory = BasicEmbeddingBackend(provider=provider, reranker=reranker)
        self.top_k = top_k

    def ingest_history(self, session_data: List[Dict[str, Any]]):
        self.memory.clear()
        
        chunk_size = 5
        for i in range(0, len(session_data), chunk_size):
            chunk = session_data[i:i+chunk_size]
            text_chunk = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chunk])
            
            mock_ts = float(i * 3600)
            
            self.memory.store(MemoryEntry(
                text=text_chunk,
                category="raw_history_chunk",
                sequence_id=i,
                timestamp=mock_ts
            ))

    def ingest_history_structured(self, raw_sessions: List[List[Dict[str, Any]]], dates: List[str], session_ids: List[str]):
        self.memory.clear()
        
        # 1. Parse dates and pair sessions chronologically
        parsed_sessions = []
        for session, date_str, sess_id in zip(raw_sessions, dates, session_ids):
            clean_date = re.sub(r'\s*\([A-Za-z]+\)\s*', ' ', date_str)
            try:
                dt = datetime.strptime(clean_date, "%Y/%m/%d %H:%M")
                ts = dt.timestamp()
            except Exception as e:
                ts = 0.0
            parsed_sessions.append((session, date_str, sess_id, ts))
            
        # Sort chronologically by timestamp
        parsed_sessions.sort(key=lambda x: x[3])
        
        # 2. Chunk each session and store in memory
        for session, date_str, sess_id, ts in parsed_sessions:
            chunk_size = 5
            for idx in range(0, len(session), chunk_size):
                chunk = session[idx:idx+chunk_size]
                text_chunk = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chunk])
                
                # Slight offset to keep sequence within session
                chunk_ts = ts + float(idx // chunk_size)
                
                self.memory.store(MemoryEntry(
                    text=text_chunk,
                    category="raw_history_chunk",
                    session_id=sess_id,
                    sequence_id=idx // chunk_size,
                    timestamp=chunk_ts
                ))

    def answer_question(self, question: str, question_timestamp: float = None) -> Dict[str, Any]:
        results = self.memory.search(question, top_k=self.top_k, question_timestamp=question_timestamp)
        retrieved_text = "\n".join([f"Fact {idx+1}: {r.entry.text}" for idx, r in enumerate(results)])
        
        prompt = (
            f"Here are the most relevant facts retrieved from the user's long-term memory:\n\n"
            f"{retrieved_text}\n\n"
            f"Based strictly on these facts, please answer the following question. "
            f"If the information requires calculating days/months, perform the math based on the provided dates. "
            f"If the information is not present, say 'I don't know'.\n\n"
            f"Question: {question}"
        )
        
        result = self.client.generate(prompt=prompt, system_prompt="You are a helpful and precise assistant. Pay close attention to chronological order and dates.")
        
        return {
            "answer": result["content"],
            "latency_s": result["latency_s"],
            "tokens": result["prompt_tokens"] + result["completion_tokens"],
            "retrieved_memories": [r.entry.text for r in results],
            "ranking_scores": [r.score for r in results]
        }
