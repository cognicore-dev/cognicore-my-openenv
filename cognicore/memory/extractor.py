import json
import logging
import os
import time

from cognicore.llm.gemini import ask_llm

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

def extract_memories(transcript: str, agent_id: str = "extractor_agent"):
    """
    Reads a conversational transcript, extracts memories via LLM,
    and stores them directly into the ChromaDB vector engine.
    """
    logger.info("Extracting memories from transcript...")
    try:
        response = ask_llm(prompt=transcript, system=SYSTEM_PROMPT, max_tokens=500, temperature=0.1)
    except Exception as e:
        logger.error(f"LLM Extraction failed: {e}")
        return []
        
    try:
        # Clean up any potential markdown backticks from LLM output
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
        
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        chroma_dir = os.path.abspath("./cognicore_data/chroma_db")
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection(name="cognicore_memories")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        docs = []
        metadatas = []
        ids = []
        
        # We need unique IDs. We can just use timestamp + index
        base_id = int(time.time() * 1000)
        
        for idx, m in enumerate(memories):
            text = m.get("text")
            mtype = m.get("memory_type", "semantic")
            if not text:
                continue
                
            docs.append(text)
            metadatas.append({
                "agent_id": agent_id,
                "memory_type": mtype,
                "state": "ACTIVE",
                "timestamp": time.time()
            })
            ids.append(f"extracted_mem_{base_id}_{idx}")
            
        if docs:
            embeddings = model.encode(docs).tolist()
            collection.add(
                documents=docs,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully embedded and saved {len(docs)} memories to ChromaDB!")
            
    except Exception as e:
        logger.error(f"Failed to save memories to ChromaDB: {e}")
        
    return memories
