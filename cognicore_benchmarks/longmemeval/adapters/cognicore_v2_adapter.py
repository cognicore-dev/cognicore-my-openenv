from typing import Dict, Any, List
import re
from datetime import datetime
from cognicore_benchmarks.longmemeval.adapters.base_adapter import BaseAgentAdapter
from cognicore_benchmarks.common.llm_client import LLMClient
from cognicore.memory.embedding_backend import BasicEmbeddingBackend
from cognicore.memory.providers.sentence_transformers import SentenceTransformerProvider
from cognicore.memory.providers.reranker import SentenceTransformerReranker
from cognicore.memory.base import MemoryEntry

from cognicore_benchmarks.longmemeval.v2.state_store import V2MemoryIndexes
from cognicore_benchmarks.longmemeval.v2.extractor import (
    extract_fact_memories, extract_update_memories, extract_preference_evidence,
    extract_event_memories, extract_assistant_action, extract_artifact
)
from cognicore_benchmarks.longmemeval.v2.query_classifier import classify_query
from cognicore_benchmarks.longmemeval.v2.retriever import retrieve_evidence
from cognicore_benchmarks.longmemeval.v2.context_builder import build_context
from cognicore_benchmarks.longmemeval.v2.schema import RawTurnMemory

class CognicoreLongMemEvalV2(BaseAgentAdapter):
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", top_k: int = 5):
        self.client = LLMClient(model_name=model_name)
        
        provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
        reranker = SentenceTransformerReranker()
        
        self.indexes = V2MemoryIndexes()
        self.indexes.raw_store = BasicEmbeddingBackend(provider=provider, reranker=reranker)
        self.indexes.typed_store = BasicEmbeddingBackend(provider=provider, reranker=reranker)
        self.top_k = top_k

    def ingest_history(self, session_data: List[Dict[str, Any]]):
        self.indexes.clear_all()
        # Fallback for old runner method
        for idx, msg in enumerate(session_data):
            self._process_turn(msg, "default_session", idx, float(idx), "unknown")

    def ingest_history_structured(self, raw_sessions: List[List[Dict[str, Any]]], dates: List[str], session_ids: List[str]):
        self.indexes.clear_all()
        
        parsed_sessions = []
        for session, date_str, sess_id in zip(raw_sessions, dates, session_ids):
            clean_date = re.sub(r'\s*\([A-Za-z]+\)\s*', ' ', date_str)
            try:
                dt = datetime.strptime(clean_date, "%Y/%m/%d %H:%M")
                ts = dt.timestamp()
            except Exception:
                ts = 0.0
            parsed_sessions.append((session, date_str, sess_id, ts))
            
        parsed_sessions.sort(key=lambda x: x[3])
        
        for session, date_str, sess_id, base_ts in parsed_sessions:
            for turn_idx, msg in enumerate(session):
                ts = base_ts + (turn_idx * 60) # 1 minute per turn mock
                self._process_turn(msg, sess_id, turn_idx, ts, date_str)

    def _process_turn(self, msg: Dict[str, Any], session_id: str, turn_idx: int, timestamp: float, session_date: str):
        role = msg.get("role", "user")
        text = msg.get("content", "")
        
        # 1. Raw Store
        raw = RawTurnMemory(text=text, session_id=session_id, role=role, timestamp=timestamp, source_turn_ids=[turn_idx])
        self.indexes.raw_store.store(MemoryEntry(text=text, category="raw_turn", timestamp=timestamp))
        
        if role == "user":
            # 2. Extract Facts
            facts = extract_fact_memories(text, session_id, turn_idx, timestamp)
            for f in facts:
                self.indexes.current_state.update_fact(f)
                self.indexes.typed_store.store(MemoryEntry(text=f.text, category="fact", timestamp=timestamp))
                
            # 3. Extract Updates
            updates = extract_update_memories(text, session_id, turn_idx, timestamp, self.indexes.current_state)
            for u in updates:
                # Invalidate old fact implicitly by updating the state dict
                new_fact = next((f for f in facts if f.entity == u.entity), None)
                if not new_fact:
                    # Fabricate a new active fact if not caught by fact extractor
                    new_fact = FactMemory(subject=u.subject, entity=u.entity, value=u.new_value, session_id=session_id, timestamp=timestamp, role="user", text=text)
                    self.indexes.current_state.update_fact(new_fact)
                self.indexes.typed_store.store(MemoryEntry(text=u.text, category="update", timestamp=timestamp))
                
            # 4. Extract Events
            events = extract_event_memories(text, session_id, turn_idx, timestamp, session_date)
            for e in events:
                self.indexes.timeline_store.add_event(e)
                
            # 5. Extract Preferences
            prefs = extract_preference_evidence(text, session_id, turn_idx, timestamp)
            for p in prefs:
                self.indexes.typed_store.store(MemoryEntry(text=p.text, category="preference_evidence", timestamp=timestamp))
                
        elif role == "assistant":
            acts = extract_assistant_action(text, session_id, turn_idx, timestamp)
            for a in acts:
                self.indexes.typed_store.store(MemoryEntry(text=a.text, category="assistant_action", timestamp=timestamp))
                
            art = extract_artifact(text, session_id, turn_idx, timestamp)
            if art:
                self.indexes.artifact_store.add_artifact(art)

    def answer_question(self, question: str, question_timestamp: float = None) -> Dict[str, Any]:
        query_type = classify_query(question)
        evidence = retrieve_evidence(question, query_type, self.indexes, self.top_k, question_timestamp)
        prompt = build_context(question, query_type, evidence)
        result = self.client.generate(prompt=prompt, system_prompt="You are answering a benchmark. Output ONLY the exact entity, number, or yes/no. Do NOT output conversational text, explanations, or full sentences. E.g. output '4 days' instead of 'It happened 4 days before'.")
        # Serialize evidence for benchmark logging
        retrieved_memories = [f"{type(e).__name__}: {str(e)}" for e in evidence["primary_evidence"]]
        retrieved_memories.extend([e.text for e in evidence["supporting_evidence"]])
        
        return {
            "answer": result["content"],
            "latency_s": result["latency_s"],
            "tokens": result["prompt_tokens"] + result["completion_tokens"],
            "retrieved_memories": retrieved_memories,
            "ranking_scores": [1.0] * len(retrieved_memories) # mock scores for logs
        }
