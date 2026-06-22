from typing import Dict, Any, List
from cognicore_benchmarks.longmemeval.adapters.base_adapter import BaseAgentAdapter
from cognicore_benchmarks.common.llm_client import LLMClient

class BaselineAdapter(BaseAgentAdapter):
    """
    Mode A: No Memory Baseline.
    Stuffs the entire conversational history into the prompt context.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.client = LLMClient(model_name=model_name)
        self.history_text = ""

    def ingest_history(self, session_data: List[Dict[str, str]]) -> None:
        """Concatenates all turns into a single string."""
        lines = []
        for turn in session_data:
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        self.history_text = "\n".join(lines)

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answers the question using the full history as context."""
        prompt = (
            f"Here is the conversation history:\n\n"
            f"{self.history_text}\n\n"
            f"Based on the history above, please answer the following question. "
            f"If the information is not in the history, say 'I don't know'.\n\n"
            f"Question: {question}"
        )
        
        system_prompt = "You are a helpful assistant evaluating a conversation log."
        
        result = self.client.generate(prompt=prompt, system_prompt=system_prompt)
        
        return {
            "answer": result["content"],
            "latency_s": result["latency_s"],
            "tokens": result["prompt_tokens"] + result["completion_tokens"]
        }
