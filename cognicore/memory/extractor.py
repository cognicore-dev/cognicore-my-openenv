import json
import logging
import time
from typing import Optional, List

from cognicore.llm.gemini import ask_llm
from cognicore.memory.base import MemoryBackend, MemoryEntry, MemoryScope

logger = logging.getLogger("cognicore.memory.extractor")

SYSTEM_PROMPT = """
You are a highly intelligent automated memory extractor.
Your job is to read a conversational transcript between a User and an AI Agent, and extract core facts, rules, and preferences that the AI should remember for the future.

Extract them into a JSON array of objects.
Each object should have:
- "text": The core fact, rule, or preference (written in third person, e.g., "The user prefers Python", or "The project uses React").
- "memory_type": Either "preference", "semantic", or "constraint".

If there are no useful facts to extract, return an empty array [].
Respond ONLY with valid JSON. Do not include markdown formatting or backticks.
"""

class TranscriptExtractor:
    """
    Automated Memory Extraction Pipeline.
    Listens to conversational transcripts, extracts facts via LLM,
    and intelligently stores them in the provided MemoryBackend.
    """
    
    def __init__(self, backend: MemoryBackend):
        self.backend = backend
        
    def extract_and_store(self, transcript: str, agent_id: str = "extractor_agent", scope: MemoryScope = MemoryScope.AGENT) -> List[str]:
        """
        Extracts memories from a transcript and routes them through the core MemoryBackend.
        Returns a list of entry_ids that were created.
        """
        logger.info("Extracting memories from transcript...")
        try:
            response = ask_llm(prompt=transcript, system=SYSTEM_PROMPT, max_tokens=500, temperature=0.1)
        except Exception as e:
            logger.error(f"LLM Extraction failed: {e}")
            return []
            
        try:
            clean_json = response.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.startswith("```"):
                clean_json = clean_json[3:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
                
            memories = json.loads(clean_json.strip())
        except Exception as e:
            logger.error(f"Failed to parse LLM JSON response: {response}")
            return []
            
        if not memories:
            logger.info("No memories extracted.")
            return []
            
        saved_ids = []
        for m in memories:
            text = m.get("text")
            mtype = m.get("memory_type", "semantic")
            if not text:
                continue
                
            entry = MemoryEntry(
                text=text,
                memory_type=mtype,
                scope=scope,
                scope_id=agent_id,
                state="active",
                timestamp=time.time()
            )
            
            entry_id = self.backend.store(entry)
            if entry_id:
                saved_ids.append(entry_id)
                
        logger.info(f"Successfully extracted and stored {len(saved_ids)} memories into backend.")
        return saved_ids

# Backwards compatibility function
def extract_memories(transcript: str, agent_id: str = "extractor_agent") -> List[dict]:
    """Legacy helper. Use TranscriptExtractor instead."""
    from cognicore.memory.chroma_backend import ChromaMemoryBackend
    backend = ChromaMemoryBackend()
    extractor = TranscriptExtractor(backend)
    extractor.extract_and_store(transcript, agent_id)
    return []
