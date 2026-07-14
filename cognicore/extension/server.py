import logging
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from cognicore.memory import SQLiteMemoryBackend, MemoryEntry, MemoryScope

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
        """Store information that is likely to be useful in future conversations or tasks, 
        such as user preferences, project decisions, constraints, recurring facts, 
        successful procedures, or important corrections. Do not store trivial conversational details.
        
        Args:
            text: The fact, preference, or decision to store.
            category: Optional category for grouping related memories. Defaults to 'general'.
            scope: The scope of the memory. Must be 'user' (global preferences) or 'project' (current project).
            
        Returns:
            A success message confirming storage and the assigned memory ID.
        """
        _ensure_backend()
        
        try:
            mem_scope = MemoryScope(scope.lower())
        except ValueError:
            return f"Error: Invalid scope '{scope}'. Must be 'user' or 'project'."
            
        scope_id = _get_project_id() if mem_scope == MemoryScope.PROJECT else ""
        
        entry = MemoryEntry(
            text=text,
            category=category,
            scope=mem_scope,
            scope_id=scope_id,
            memory_type="semantic"
        )
        
        try:
            entry_id = _backend.store(entry)
            return f"Successfully stored memory in category '{category}' (scope: {mem_scope.value}).\nID: {entry_id}"
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return f"Error storing memory: {str(e)}"

    @mcp.tool()
    def cognicore_recall(query: str, category: str = "", scope: str = "user", top_k: int = 5) -> str:
        """Search persistent CogniCore memory for information relevant to the user's current request. 
        Use this when previous preferences, project decisions, constraints, facts, 
        or prior solutions may help answer the request.
        
        Args:
            query: The search terms to find relevant memories.
            category: Optional category filter.
            scope: The scope to search in. Must be 'user' or 'project'.
            top_k: Maximum number of results to return.
            
        Returns:
            A formatted list of matching memories or a message if none found.
        """
        _ensure_backend()
        
        try:
            mem_scope = MemoryScope(scope.lower())
        except ValueError:
            return f"Error: Invalid scope '{scope}'. Must be 'user' or 'project'."
            
        scope_id = _get_project_id() if mem_scope == MemoryScope.PROJECT else None
        
        try:
            category_filter = category if category else None
            results = _backend.search(
                query=query, 
                top_k=top_k, 
                category=category_filter,
                scope=mem_scope,
                scope_id=scope_id
            )
            
            if not results:
                return f"No memories found matching '{query}' in scope '{mem_scope.value}'."
                
            lines = [f"Found {len(results)} relevant memories:"]
            for i, result in enumerate(results, 1):
                cat = result.entry.category
                score = result.score
                lines.append(f"{i}. [ID: {result.entry.entry_id} | Category: {cat} | Score: {score:.2f}]\n   {result.entry.text}")
                
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Failed to recall memories: {e}")
            return f"Error recalling memories: {str(e)}"

    @mcp.tool()
    def cognicore_forget(entry_id: str) -> str:
        """Delete a specific memory by its ID. Use this when the user explicitly asks 
        to forget something or when a memory is no longer true or relevant.
        
        Args:
            entry_id: The ID of the memory to delete (obtained via list or recall).
            
        Returns:
            A success message or an error if the ID was not found.
        """
        _ensure_backend()
        
        try:
            success = _backend.delete(entry_id)
            if success:
                return f"Successfully deleted memory with ID {entry_id}."
            else:
                return f"Could not find or delete memory with ID {entry_id}."
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return f"Error deleting memory: {str(e)}"

    @mcp.tool()
    def cognicore_list(limit: int = 10, category: str = "", scope: str = "user") -> str:
        """List recently stored memories, optionally filtered by category and scope.
        
        Args:
            limit: Maximum number of memories to list. Defaults to 10.
            category: Optional category to filter by.
            scope: The scope to list. Must be 'user' or 'project'.
            
        Returns:
            A formatted list of recent memories.
        """
        _ensure_backend()
        
        try:
            mem_scope = MemoryScope(scope.lower())
        except ValueError:
            return f"Error: Invalid scope '{scope}'. Must be 'user' or 'project'."
            
        scope_id = _get_project_id() if mem_scope == MemoryScope.PROJECT else None
        
        try:
            # Using search with empty query to take advantage of scope filtering in SQLite backend
            category_filter = category if category else None
            results = _backend.search(
                query="", 
                top_k=limit, 
                category=category_filter,
                scope=mem_scope,
                scope_id=scope_id
            )
            
            if not results:
                return f"No memories currently stored in scope '{mem_scope.value}'."
                
            lines = [f"Listing {len(results)} recent memories:"]
            for i, result in enumerate(results, 1):
                entry = result.entry
                lines.append(f"{i}. [ID: {entry.entry_id} | Category: {entry.category}]\n   {entry.text}")
                
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Failed to list memories: {e}")
            return f"Error listing memories: {str(e)}"

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
