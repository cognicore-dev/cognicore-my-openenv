import pytest
import jwt
from fastapi.testclient import TestClient
from cognicore.extension.remote import app, JWT_SECRET, JWT_ALGORITHM, get_db_path_for_user

client = TestClient(app)

def create_token(sub: str) -> str:
    return jwt.encode({"sub": sub}, JWT_SECRET, algorithm=JWT_ALGORITHM)

def test_remote_missing_auth():
    # Should get 401 without auth header
    response = client.get("/mcp/sse")
    assert response.status_code == 401
    assert "Missing or invalid Bearer token" in response.text

def test_remote_invalid_jwt():
    response = client.get("/mcp/sse", headers={"Authorization": "Bearer invalid.token.here"})
    assert response.status_code == 401
    assert "Invalid JWT" in response.text
    
def test_remote_missing_sub():
    token = jwt.encode({"other": "value"}, JWT_SECRET, algorithm=JWT_ALGORITHM)
    response = client.get("/mcp/sse", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert "JWT missing 'sub' claim" in response.text

def test_remote_with_valid_jwt():
    token = create_token("user_123")
    try:
        response = client.get("/mcp/sse", headers={"Authorization": f"Bearer {token}"}, timeout=1.0)
    except Exception:
        # Starlette SSE might timeout in TestClient due to endless stream
        pass
    
    # We can also test that POST /mcp/message returns 400 (Bad Request from fastmcp for bad sessionId) instead of 401
    response = client.post("/mcp/message?sessionId=123", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code != 401

def test_db_path_sanitization():
    # Attempting path traversal in subject
    malicious_sub = "../../../etc/passwd"
    path = get_db_path_for_user(malicious_sub)
    
    import hashlib
    safe_hash = hashlib.sha256(malicious_sub.encode("utf-8")).hexdigest()
    
    # Path should only contain the hash, not the traversal string
    assert safe_hash in path
    assert ".." not in path
    assert path.endswith(f"memory_{safe_hash}.db")

