import logging
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from cognicore.memory import SQLiteMemoryBackend, MemoryEntry, MemoryScope
from cognicore.memory.decompose import decompose

logger = logging.getLogger("cognicore.extension")

# Optional: Semantic search dependency
try:
    from sentence_transformers import SentenceTransformer
    class SentenceTransformerProvider:
        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
            self.model = SentenceTransformer(model_name)
        def embed(self, text: str) -> List[float]:
            return self.model.encode(text).tolist()
        def embed_batch(self, texts: List[str]) -> List[List[float]]:
            return self.model.encode(texts).tolist()
        @property
        def dimension(self) -> int:
            return self.model.get_sentence_embedding_dimension()
    _HAS_SEMANTIC = True
except ImportError:
    _HAS_SEMANTIC = False
    SentenceTransformerProvider = None

# ---------------------------------------------------------------------------
# MCP SDK import (optional dependency)
# ---------------------------------------------------------------------------
try:
    from mcp.server.fastmcp import FastMCP
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


_backend: Optional[SQLiteMemoryBackend] = None

def _get_data_dir() -> Path:
    raw = os.environ.get("COGNICORE_EXTENSION_DIR", "")
    if raw:
        data_dir = Path(raw)
    else:
        data_dir = Path.home() / ".cognicore" / "extension"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def _get_project_id() -> str:
    """Finds the canonical project ID by searching for .git root and hashing its absolute path."""
    import hashlib
    current = Path.cwd().resolve()
    target = current
    
    # Traverse up to find .git
    for p in [current] + list(current.parents):
        if (p / ".git").is_dir():
            target = p
            break
            
    # Create a stable, sanitized hash of the absolute path
    path_str = str(target)
    return hashlib.sha256(path_str.encode("utf-8")).hexdigest()

def _ensure_backend():
    global _backend
    if _backend is not None:
        return
    
    db_path = _get_data_dir() / "memory.db"
    
    provider = None
    if _HAS_SEMANTIC and os.environ.get("COGNICORE_USE_SEMANTIC", "0") == "1":
        try:
            provider = SentenceTransformerProvider()
            logger.info("Semantic search enabled via sentence-transformers")
        except Exception as e:
            logger.warning(f"Failed to load sentence-transformers, falling back to lexical search: {e}")
            
    _backend = SQLiteMemoryBackend(str(db_path), provider=provider)
    logger.info(f"Initialized CogniCore Extension Memory at {db_path}")


def create_extension_server() -> "FastMCP":
    if not _MCP_AVAILABLE:
        raise ImportError(
            "The 'mcp' package is required for MCP server support. "
            "Install it with: pip install mcp"
        )
        
    mcp = FastMCP(
        "cognicore-memory",
        instructions=(
            "Provides persistent, long-term memory for Claude Desktop. "
            "Store, recall, and manage user preferences, project details, and key facts."
        )
    )

    @mcp.tool()
    def cognicore_remember(text: str, category: str = "general", scope: str = "user") -> str:
        """Store a fact, preference, or decision. Auto-decomposes compound text into atomic facts."""
        _ensure_backend()
        
        try:
            mem_scope = MemoryScope(scope.lower())
        except ValueError:
            return "Error: scope must be 'user' or 'project'."
            
        scope_id = _get_project_id() if mem_scope == MemoryScope.PROJECT else ""
        
        # Atomic decomposition: split paragraphs into independently searchable facts
        facts = decompose(text)
        ids = []
        for fact in facts:
            entry = MemoryEntry(
                text=fact,
                category=category,
                scope=mem_scope,
                scope_id=scope_id,
                memory_type="semantic"
            )
            try:
                ids.append(str(_backend.store(entry)))
            except Exception as e:
                logger.error(f"Failed to store fact: {e}")
                return f"Error: {e}"

        if len(ids) == 1:
            return f"OK id={ids[0]}"
        return f"OK {len(ids)} facts: {','.join(ids)}"

    @mcp.tool()
    def cognicore_recall(query: str, category: str = "", scope: str = "user", top_k: int = 3) -> str:
        """Search memory. Returns matching facts."""
        _ensure_backend()
        
        try:
            mem_scope = MemoryScope(scope.lower())
        except ValueError:
            return "Error: scope must be 'user' or 'project'."
            
        scope_id = _get_project_id() if mem_scope == MemoryScope.PROJECT else None
        
        try:
            results = _backend.search(
                query=query, 
                top_k=top_k, 
                category=category if category else None,
                scope=mem_scope,
                scope_id=scope_id
            )
            
            if not results:
                return "(none)"
            # Ultra-compact format: one fact per line, minimal overhead
            return "\n".join(f"#{r.entry.entry_id}: {r.entry.text}" for r in results)
        except Exception as e:
            logger.error(f"Failed to recall memories: {e}")
            return f"Error: {e}"

    @mcp.tool()
    def cognicore_forget(entry_id: str) -> str:
        """Delete a memory by ID."""
        _ensure_backend()
        
        try:
            success = _backend.delete(entry_id)
            return "OK" if success else "Not found"
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return f"Error: {e}"

    @mcp.tool()
    def cognicore_list(limit: int = 10, category: str = "", scope: str = "user") -> str:
        """List recent memories."""
        _ensure_backend()
        
        try:
            mem_scope = MemoryScope(scope.lower())
        except ValueError:
            return "Error: scope must be 'user' or 'project'."
            
        scope_id = _get_project_id() if mem_scope == MemoryScope.PROJECT else None
        
        try:
            results = _backend.search(
                query="", 
                top_k=limit, 
                category=category if category else None,
                scope=mem_scope,
                scope_id=scope_id
            )
            
            if not results:
                return "(empty)"
            return "\n".join(f"#{r.entry.entry_id}: {r.entry.text}" for r in results)
        except Exception as e:
            logger.error(f"Failed to list memories: {e}")
            return f"Error: {e}"

    @mcp.tool()
    def cognicore_stats() -> str:
        """Get statistics about the CogniCore memory extension storage.
        
        Returns:
            Statistics including total count, location, and backend mode.
        """
        _ensure_backend()
        
        try:
            count = _backend.count()
            db_path = _get_data_dir() / "memory.db"
            mode = "Semantic (sentence-transformers)" if _backend.provider else "Lexical (SQLite FTS5)"
            
            return (
                f"CogniCore Extension Memory Stats:\n"
                f"- Total memories stored: {count}\n"
                f"- Search mode: {mode}\n"
                f"- Storage location: {db_path}"
            )
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return f"Error retrieving stats: {str(e)}"

    return mcp

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    # stdio uses stdout for communication, so ensure logger goes to stderr
    logging.getLogger().handlers[0].stream = sys.stderr
    
    server = create_extension_server()
    server.run(transport="stdio")

if __name__ == "__main__":
    main()
