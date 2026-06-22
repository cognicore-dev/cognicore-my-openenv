from typing import Dict, Any, List
from cognicore_benchmarks.longmemeval.adapters.base_adapter import BaseAgentAdapter
from cognicore_benchmarks.common.llm_client import LLMClient
from cognicore.memory.tfidf_backend import TFIDFMemoryBackend
from cognicore.memory.base import MemoryEntry

class NaiveAdapter(BaseAgentAdapter):
    """
    Mode B: Naive Vector Retrieval.
    Embeds each turn individually and retrieves top-k chunks.
    No cognitive grouping, no reflection.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", top_k: int = 5):
        self.client = LLMClient(model_name=model_name)
        self.memory = TFIDFMemoryBackend()
        self.top_k = top_k

    def ingest_history(self, session_data: List[Dict[str, str]]) -> None:
        """Stores each turn as a raw memory entry."""
        self.memory.clear()
        for i, turn in enumerate(session_data):
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            
            entry = MemoryEntry(
                text=f"{role}: {content}",
                category="raw_conversation",
                metadata={"turn_index": i}
            )
            self.memory.store(entry)

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Retrieves top-k related chunks to answer the question."""
        results = self.memory.search(question, top_k=self.top_k)
        
        retrieved_text = "\n".join([r.entry.text for r in results])
        
        prompt = (
            f"Here are retrieved snippets from the conversation history:\n\n"
            f"{retrieved_text}\n\n"
            f"Based on these snippets, please answer the following question. "
            f"If the information is not present, say 'I don't know'.\n\n"
            f"Question: {question}"
        )
        
        system_prompt = "You are a helpful assistant answering questions based on memory snippets."
        
        result = self.client.generate(prompt=prompt, system_prompt=system_prompt)
        
        return {
            "answer": result["content"],
            "latency_s": result["latency_s"],
            "tokens": result["prompt_tokens"] + result["completion_tokens"]
        }
