import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cognicore.memory.base import MemoryBackend, MemoryEntry, SearchResult, MemoryScope
from cognicore.memory.temporal import TemporalResolutionEngine
from cognicore.memory.events import event_bus

logger = logging.getLogger(__name__)

class ChromaMemoryBackend(MemoryBackend):
    """
    Production-ready Vector Store using ChromaDB.
    Handles semantic search, multi-agent pooling, and temporal decay natively.
    """
    
    def __init__(self, persistence_path: str = "./cognicore_data/chroma_db", collection_name: str = "cognicore_memories"):
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("ChromaMemoryBackend requires 'chromadb' and 'sentence-transformers'. Install via pip install cognicore-env[memory]")
            
        self.persistence_path = Path(persistence_path).expanduser().resolve()
        self.persistence_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.persistence_path))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.temporal_engine = TemporalResolutionEngine(self)
        
        logger.info(f"ChromaMemoryBackend initialized at {self.persistence_path}")
        
    def _entry_to_dict(self, entry: MemoryEntry) -> Dict[str, Any]:
        """Convert MemoryEntry to simple types for Chroma Metadata"""
        meta = {
            "category": entry.category,
            "scope": entry.scope.value,
            "scope_id": entry.scope_id,
            "session_id": entry.session_id,
            "memory_type": entry.memory_type,
            "timestamp": entry.timestamp,
            "state": entry.state
        }
        if entry.supersedes:
            meta["supersedes"] = entry.supersedes
        if entry.metadata:
            meta["custom_meta"] = json.dumps(entry.metadata)
        return meta
        
    def _dict_to_entry(self, text: str, meta: Dict[str, Any], entry_id: str) -> MemoryEntry:
        """Reconstruct MemoryEntry from Chroma Metadata"""
        entry = MemoryEntry(
            text=text,
            category=meta.get("category", "general"),
            scope=MemoryScope(meta.get("scope", "global")),
            scope_id=meta.get("scope_id", ""),
            session_id=meta.get("session_id", "default"),
            memory_type=meta.get("memory_type", "semantic"),
            supersedes=meta.get("supersedes"),
            timestamp=meta.get("timestamp", 0.0),
            state=meta.get("state", "active"),
            entry_id=entry_id
        )
        if "custom_meta" in meta:
            try:
                entry.metadata = json.loads(meta["custom_meta"])
            except:
                pass
        return entry

    def store(self, entry: MemoryEntry) -> str:
        if not entry.entry_id:
            import uuid
            entry.entry_id = str(uuid.uuid4())
            
        if not entry.timestamp:
            entry.timestamp = time.time()
            
        text_to_embed = entry.text or entry.action or ""
        if not text_to_embed:
            return entry.entry_id
            
        vector = self.model.encode([text_to_embed])[0].tolist()
        
        self.collection.add(
            documents=[entry.text],
            embeddings=[vector],
            metadatas=[self._entry_to_dict(entry)],
            ids=[entry.entry_id]
        )
        
        event_bus.publish("on_store", entry=entry, entry_id=entry.entry_id)
        return entry.entry_id

    def search(self, query: str, top_k: int = 5,
               category: Optional[str] = None,
               scope: Optional[MemoryScope] = None,
               scope_id: Optional[str] = None,
               question_timestamp: Optional[float] = None,
               candidate_k: int = 20) -> List[SearchResult]:
               
        query_vec = self.model.encode([query])[0].tolist()
        
        where = {}
        if category:
            where["category"] = category
            
        # Cross-Agent Global Pool Logic
        if scope and scope != MemoryScope.GLOBAL:
            where["scope"] = scope.value
        if scope_id and scope != MemoryScope.GLOBAL:
            where["scope_id"] = scope_id
            
        if not where:
            where = None
            
        res = self.collection.query(
            query_embeddings=[query_vec],
            n_results=candidate_k,
            where=where
        )
        
        raw_results = []
        if res and res['documents'] and res['documents'][0]:
            for i in range(len(res['documents'][0])):
                doc = res['documents'][0][i]
                meta = res['metadatas'][0][i]
                eid = res['ids'][0][i]
                dist = res['distances'][0][i]
                
                # Chroma uses L2/Cosine distance. Lower is closer.
                sim = max(0.01, 1.0 - (dist / 2.0))
                
                entry = self._dict_to_entry(doc, meta, eid)
                raw_results.append(SearchResult(entry=entry, score=sim, source="chroma"))
                
        # Apply Temporal Decay and Supersession Logic
        final_results = self.temporal_engine.resolve(raw_results, question_timestamp=question_timestamp)
        return final_results[:top_k]

    def get_superseding(self, entry_id: str) -> Optional[MemoryEntry]:
        res = self.collection.get(
            where={"supersedes": entry_id},
            limit=1
        )
        if res and res['documents']:
            doc = res['documents'][0]
            meta = res['metadatas'][0]
            eid = res['ids'][0]
            return self._dict_to_entry(doc, meta, eid)
        return None

    def update(self, entry_id: str, **kwargs) -> bool:
        res = self.collection.get(ids=[entry_id])
        if not res or not res['documents']:
            return False
            
        doc = res['documents'][0]
        meta = res['metadatas'][0]
        
        for k, v in kwargs.items():
            if k == "text":
                doc = v
            elif k in ["category", "scope", "scope_id", "session_id", "memory_type", "state", "timestamp", "supersedes"]:
                if isinstance(v, MemoryScope):
                    meta[k] = v.value
                else:
                    meta[k] = v
            elif k == "metadata":
                meta["custom_meta"] = json.dumps(v)
                
        # If text changed, re-embed. Else just update
        if "text" in kwargs:
            vector = self.model.encode([doc])[0].tolist()
            self.collection.update(
                ids=[entry_id],
                documents=[doc],
                embeddings=[vector],
                metadatas=[meta]
            )
        else:
            self.collection.update(
                ids=[entry_id],
                metadatas=[meta]
            )
        return True

    def delete(self, entry_id: str) -> bool:
        try:
            self.collection.delete(ids=[entry_id])
            return True
        except:
            return False

    def get_all(self) -> List[MemoryEntry]:
        try:
            res = self.collection.get()
            entries = []
            if res and res['documents']:
                for doc, meta, eid in zip(res['documents'], res['metadatas'], res['ids']):
                    entries.append(self._dict_to_entry(doc, meta, eid))
            return entries
        except Exception as e:
            logger.error(f"Failed to get_all from Chroma: {e}")
            return []
