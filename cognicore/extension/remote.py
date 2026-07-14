import os
import hashlib
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from mcp.server.fastmcp import FastMCP, Context
import uvicorn
import jwt

from mcp.server.transport_security import TransportSecuritySettings

# We use FastMCP for the core logic, but we inject a context-aware backend
mcp = FastMCP(
    "cognicore-remote",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False)
)
security = HTTPBearer()





# By default, use a local dev secret if none provided (ONLY FOR DEV!)
JWT_SECRET = os.environ.get("COGNICORE_JWT_SECRET", "dev_secret_key_change_in_prod")
JWT_ALGORITHM = "HS256"

def get_user_id(request: Request) -> str:
    """Extract and validate user_id from the Authorization JWT."""
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Bearer token")
    
    token = auth.split(" ")[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="JWT missing 'sub' claim")
        return user_id
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid JWT")

def get_db_path_for_user(user_id: str) -> str:
    """Creates a secure, sanitized path for a user's memory database."""
    base_dir = Path.home() / ".cognicore" / "remote"
    base_dir.mkdir(parents=True, exist_ok=True)
    # Prevent path traversal by hashing the user ID
    safe_id = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    return str(base_dir / f"memory_{safe_id}.db")

def get_backend(ctx: Context):
    """Dynamically resolve the backend for the current request context."""
    request = ctx.request_context
    if not request:
        raise RuntimeError("No request context available. Make sure to use StreamableHTTP or SSE transport with Starlette.")
    
    # In FastMCP, ctx.request_context is the Starlette Request object from the POST /message
    user_id = get_user_id(request)
    db_path = get_db_path_for_user(user_id)
    
    from cognicore.memory import SQLiteMemoryBackend
    return SQLiteMemoryBackend(db_path)

from cognicore.memory import MemoryEntry, MemoryScope

@mcp.tool()
def cognicore_remember(text: str, ctx: Context, category: str = "general", scope: str = "user") -> str:
    """Store information that is likely to be useful in future conversations or tasks."""
    backend = get_backend(ctx)
    try:
        mem_scope = MemoryScope(scope.lower())
    except ValueError:
        return f"Error: Invalid scope '{scope}'. Must be 'user' or 'project'."
        
    entry = MemoryEntry(
        text=text,
        category=category,
        scope=mem_scope,
        scope_id="", # Remote MCP typically acts as a cloud brain, project scope might need client context
        memory_type="semantic"
    )
    entry_id = backend.store(entry)
    return f"Successfully stored memory (ID: {entry_id})"

@mcp.tool()
def cognicore_recall(query: str, ctx: Context, category: str = "", scope: str = "user", top_k: int = 5) -> str:
    """Search persistent CogniCore memory for information relevant to the user's current request."""
    backend = get_backend(ctx)
    try:
        mem_scope = MemoryScope(scope.lower())
    except ValueError:
        return f"Error: Invalid scope '{scope}'. Must be 'user' or 'project'."
        
    results = backend.search(
        query=query, 
        top_k=top_k, 
        category=category if category else None,
        scope=mem_scope
    )
    if not results:
        return f"No memories found matching '{query}'."
    lines = [f"Found {len(results)} relevant memories:"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [ID: {r.entry.entry_id}] {r.entry.text}")
    return "\n".join(lines)

@mcp.tool()
def cognicore_forget(entry_id: str, ctx: Context) -> str:
    """Delete a specific memory by its ID."""
    backend = get_backend(ctx)
    success = backend.delete(entry_id)
    if success:
        return f"Successfully deleted memory with ID {entry_id}."
    return f"Could not find or delete memory with ID {entry_id}."

@mcp.tool()
def cognicore_list(ctx: Context, limit: int = 10, category: str = "", scope: str = "user") -> str:
    """List recently stored memories."""
    backend = get_backend(ctx)
    try:
        mem_scope = MemoryScope(scope.lower())
    except ValueError:
        return f"Error: Invalid scope '{scope}'. Must be 'user' or 'project'."
        
    results = backend.search(
        query="", 
        top_k=limit, 
        category=category if category else None,
        scope=mem_scope
    )
    if not results:
        return "No memories currently stored."
    lines = [f"Listing {len(results)} recent memories:"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [ID: {r.entry.entry_id}] {r.entry.text}")
    return "\n".join(lines)

# Create the FastAPI app
app = FastAPI(title="CogniCore Remote MCP Server")

from fastapi.middleware.cors import CORSMiddleware

class StreamingHeadersMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = message.get("headers", [])
                is_sse = False
                for k, v in headers:
                    if k.lower() == b"content-type" and b"text/event-stream" in v:
                        is_sse = True
                        break
                
                if is_sse:
                    new_headers = []
                    for k, v in headers:
                        if k.lower() not in (b"cache-control", b"x-accel-buffering"):
                            new_headers.append((k, v))
                    new_headers.append((b"cache-control", b"no-cache, no-transform"))
                    new_headers.append((b"x-accel-buffering", b"no"))
                    message["headers"] = new_headers
            await send(message)

        await self.app(scope, receive, send_wrapper)

app.add_middleware(StreamingHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to Claude's domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Starlette app and mount it
mcp_app = mcp.sse_app()
@app.middleware("http")
async def verify_auth_header(request: Request, call_next):
    if request.url.path.startswith("/mcp"):
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Bearer "):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid Bearer token"})
            
        token = auth.split(" ")[1]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            if not payload.get("sub"):
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=401, content={"detail": "JWT missing 'sub' claim"})
        except jwt.PyJWTError as e:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": f"Invalid JWT: {str(e)}"})
            
    return await call_next(request)

app.mount("/mcp", mcp_app)

if __name__ == "__main__":
    uvicorn.run("cognicore.extension.remote:app", host="0.0.0.0", port=8000, reload=True)  # nosec B104
