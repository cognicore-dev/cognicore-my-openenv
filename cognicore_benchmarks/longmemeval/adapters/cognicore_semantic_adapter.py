from typing import Dict, Any, List
from cognicore_benchmarks.longmemeval.adapters.base_adapter import BaseAgentAdapter
from cognicore_benchmarks.common.llm_client import LLMClient
from cognicore import CogniCoreRuntime
from cognicore.memory.tfidf_backend import TFIDFMemoryBackend
from cognicore.memory.base import MemoryEntry

class CognicoreSemanticAdapter(BaseAgentAdapter):
    """
    Mode C: CogniCore Semantic Memory.
    Uses CogniCore's categorization and semantic extraction, but without reflection.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", top_k: int = 5):
        self.client = LLMClient(model_name=model_name)
        self.memory = TFIDFMemoryBackend()
        self.runtime = CogniCoreRuntime(memory=self.memory)
        self.top_k = top_k

    def ingest_history(self, session_data: List[Dict[str, str]]) -> None:
        """Process history using LLM to extract semantic facts."""
        self.memory.clear()
        
        # In a real scenario, we might batch these or use a smaller/faster model for extraction
        # Here we do a simplistic extraction per N turns
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
        """Retrieves facts via semantic search to answer the question."""
        results = self.memory.search(question, top_k=self.top_k)
        
        retrieved_text = "\n".join([r.entry.text for r in results])
        
        prompt = (
            f"Here are extracted facts from the user's long-term memory:\n\n"
            f"{retrieved_text}\n\n"
            f"Based on these facts, please answer the following question. "
            f"If the information is not present, say 'I don't know'.\n\n"
            f"Question: {question}"
        )
        
        system_prompt = "You are a helpful assistant answering questions based on memory facts."
        
        result = self.client.generate(prompt=prompt, system_prompt=system_prompt)
        
        return {
            "answer": result["content"],
            "latency_s": result["latency_s"],
            "tokens": result["prompt_tokens"] + result["completion_tokens"]
        }
