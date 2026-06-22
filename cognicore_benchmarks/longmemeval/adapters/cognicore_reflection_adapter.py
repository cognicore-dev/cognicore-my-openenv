from typing import Dict, Any, List
from cognicore_benchmarks.longmemeval.adapters.base_adapter import BaseAgentAdapter
from cognicore_benchmarks.common.llm_client import LLMClient
from cognicore import CogniCoreRuntime
from cognicore.memory.tfidf_backend import TFIDFMemoryBackend
from cognicore.memory.base import MemoryEntry
from cognicore.middleware.reflection import ReflectionEngine

class CognicoreReflectionAdapter(BaseAgentAdapter):
    """
    Mode D: CogniCore Semantic + Reflection.
    Uses semantic memory + ReflectionEngine to generate hints based on past queries.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", top_k: int = 5):
        self.client = LLMClient(model_name=model_name)
        self.memory = TFIDFMemoryBackend()
        self.reflection = ReflectionEngine(memory=self.memory, min_samples=1)
        self.top_k = top_k

    def ingest_history(self, session_data: List[Dict[str, str]]) -> None:
        """Process history using LLM to extract semantic facts."""
        self.memory.clear()
        
        chunk_size = 5
        for i in range(0, len(session_data), chunk_size):
            chunk = session_data[i:i+chunk_size]
            text_chunk = "\n".join([f"{t.get('role', 'user')}: {t.get('content', '')}" for t in chunk])
            
            prompt = (
                f"Extract key facts, user preferences, and events from this conversation chunk.\n"
                f"Format as concise bullet points. If nothing important, return 'NONE'.\n\n"
                f"Conversation:\n{text_chunk}"
            )
            
            res = self.client.generate(prompt=prompt, system_prompt="You extract facts for a memory system.")
            facts = res["content"].strip()
            
            if facts != "NONE" and facts:
                for fact in facts.split("\n"):
                    fact = fact.strip("- *")
                    if fact:
                        entry = MemoryEntry(
                            text=fact,
                            category="extracted_fact"
                        )
                        self.memory.store(entry)

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Retrieves facts and a reflection hint to answer the question."""
        results = self.memory.search(question, top_k=self.top_k)
        retrieved_text = "\n".join([r.entry.text for r in results])
        
        # Simulate generating a hint based on the question context
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
        
        system_prompt = "You are a helpful assistant answering questions based on memory facts and reflection hints."
        
        result = self.client.generate(prompt=prompt, system_prompt=system_prompt)
        
        # We record the action (answering) to build reflection for the future
        # In a real benchmark, we'd only do this if we knew the ground truth, but here we just store the attempt
        attempt_entry = MemoryEntry(
            text=f"Question: {question} -> Answered",
            category="extracted_fact",
            action=result["content"],
            correct=None # Unknown until evaluated
        )
        self.memory.store(attempt_entry)
        
        return {
            "answer": result["content"],
            "latency_s": result["latency_s"],
            "tokens": result["prompt_tokens"] + result["completion_tokens"]
        }
