import pytest
from fastapi.testclient import TestClient
from cognicore.studio import create_studio_app
import json
import os
import tempfile
import sys

def test_studio_health_endpoint():
    app = create_studio_app()
    client = TestClient(app)
    
    # Test the root HTML endpoint
    response = client.get("/")
    assert response.status_code == 200
    assert "CogniCore Studio" in response.text

    # Test the API endpoints
    response = client.get("/api/memory/health")
    assert response.status_code == 200
    data = response.json()
    assert "total_memories" in data
    assert "avg_utility" in data
    assert "negative_transfer_count" in data

def test_studio_entries_endpoint():
    app = create_studio_app()
    client = TestClient(app)
    
    response = client.get("/api/memory/entries")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_studio_timeline_endpoint():
    app = create_studio_app()
    client = TestClient(app)
    
    response = client.get("/api/replay/timeline?task_id=test_dummy_id")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data or "error" in data
