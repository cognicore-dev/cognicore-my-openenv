from typing import Dict, Any, List
import re
from datetime import datetime
from cognicore_benchmarks.longmemeval.adapters.base_adapter import BaseAgentAdapter
from cognicore_benchmarks.common.llm_client import LLMClient
from cognicore import CogniCoreRuntime
from cognicore.memory.tfidf_backend import TFIDFMemoryBackend
from cognicore.memory.base import MemoryEntry
from cognicore.middleware.reflection import ReflectionEngine
from cognicore.middleware.propose_revise import ProposeReviseProtocol

from cognicore.memory.embedding_backend import BasicEmbeddingBackend
from cognicore.memory.providers.sentence_transformers import SentenceTransformerProvider
from cognicore.memory.providers.reranker import SentenceTransformerReranker

class CognicoreFullAdapter(BaseAgentAdapter):
    """
    Mode E: CogniCore Full System.
    Uses Semantic Memory, Reflection, and the Propose-Revise Protocol to self-correct
    before outputting the final answer.
    """
    def __init__(self, model_name: str = "openai/gpt-4o-mini", top_k: int = 5):
        self.client = LLMClient(model_name=model_name)
        
        provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
        reranker = SentenceTransformerReranker()
        self.memory = BasicEmbeddingBackend(provider=provider, reranker=reranker)
        
        self.reflection = ReflectionEngine(memory=self.memory, min_samples=1)
        self.protocol = ProposeReviseProtocol(memory=self.memory, reflection=self.reflection)
        self.top_k = top_k

    def ingest_history(self, session_data: List[Dict[str, Any]]):
        self.memory.clear()
        
        # Simple chunking: group every 5 turns
        chunk_size = 5
        for i in range(0, len(session_data), chunk_size):
            chunk = session_data[i:i+chunk_size]
            text_chunk = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chunk])
            
            # Store raw chunk directly
            self.memory.store(MemoryEntry(
                text=text_chunk,
                category="raw_history_chunk"
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
        """Retrieves facts, hints, proposes an answer, evaluates it, and revises."""
        self.protocol.begin_step()
        
        # Step 1: Context gathering with temporal support
        results = self.memory.search(question, top_k=self.top_k, question_timestamp=question_timestamp)
        retrieved_text = "\n".join([r.entry.text for r in results])
        
        # Step 2: Propose Phase
        feedback = self.protocol.propose(
            action={"question": question}, 
            group_value="extracted_fact"
        )
        
        hint = self.reflection.get_hint("extracted_fact")
        hint_text = f"Reflection Hint: {hint}\n\n" if hint else ""
        
        prompt = (
            f"Here are extracted facts from the user's long-term memory:\n\n"
            f"{retrieved_text}\n\n"
            f"{hint_text}"
            f"Based on these facts and hints, please answer the following question. "
            f"If the information is not present, say 'I don't know'.\n\n"
            f"Question: {question}"
        )
        
        # Draft Answer
        draft_result = self.client.generate(prompt=prompt, system_prompt="You are a helpful assistant.")
        draft_answer = draft_result["content"]
        
        # Step 3: Revise Phase (Self-Correction)
        revise_prompt = (
            f"Question: {question}\n\n"
            f"Draft Answer: {draft_answer}\n\n"
            f"Retrieved Facts:\n{retrieved_text}\n\n"
            f"Does the Draft Answer directly and accurately answer the question based ONLY on the Retrieved Facts? "
            f"If yes, output the draft answer exactly. "
            f"If no, or if it hallucinates information not in the facts, output 'I don't know'."
        )
        
        final_result = self.client.generate(prompt=revise_prompt, system_prompt="You are a strict evaluator. Prevent hallucinations.")
        final_answer = final_result["content"]
        
        # Record outcome (dummy correct value since we don't have the ground truth here)
        self.protocol.check_improvement(final_action=final_answer, eval_correct=False)
        self.protocol.end_step()
        
        total_latency = draft_result["latency_s"] + final_result["latency_s"]
        total_tokens = (draft_result["prompt_tokens"] + draft_result["completion_tokens"] + 
                        final_result["prompt_tokens"] + final_result["completion_tokens"])
        
        ranking_scores = [r.score for r in results]
        
        return {
            "answer": final_answer,
            "latency_s": total_latency,
            "tokens": total_tokens,
            "retrieved_memories": [r.entry.text for r in results],
            "ranking_scores": ranking_scores,
            "reflection_hint": hint,
            "draft_answer": draft_answer
        }
