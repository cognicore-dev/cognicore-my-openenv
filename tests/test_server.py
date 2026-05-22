"""Tests for CogniCore REST API Server."""

import pytest

try:
    from fastapi.testclient import TestClient
    from cognicore.server.app import create_app

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestServerRoot:
    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "CogniCore API"
        assert "version" in data
        assert "environments" in data

    def test_list_envs(self, client):
        resp = client.get("/envs")
        assert resp.status_code == 200
        envs = resp.json()["environments"]
        assert len(envs) >= 50  # grows as new environments are added
        ids = [e["id"] for e in envs]
        assert "SafetyClassification-v1" in ids
        assert "MathReasoning-v1" in ids
        assert "CodeDebugging-v1" in ids
        assert "Conversation-v1" in ids
        assert "Planning-v1" in ids


class TestServerSessions:
    def test_create_session(self, client):
        resp = client.post(
            "/envs/SafetyClassification-v1/create",
            json={"difficulty": "easy"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["env_id"] == "SafetyClassification-v1"

    def test_create_invalid_env(self, client):
        resp = client.post(
            "/envs/NonExistent-v1/create",
            json={"difficulty": "easy"},
        )
        assert resp.status_code == 404

    def test_full_lifecycle(self, client):
        # Create
        resp = client.post(
            "/envs/SafetyClassification-v1/create",
            json={"difficulty": "easy"},
        )
        sid = resp.json()["session_id"]

        # Reset
        resp = client.post(f"/sessions/{sid}/reset")
        assert resp.status_code == 200
        obs = resp.json()["observation"]
        assert "prompt" in obs or "step" in obs

        # Step
        resp = client.post(
            f"/sessions/{sid}/step",
            json={"action": {"classification": "SAFE"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "reward" in data
        assert "done" in data
        assert "observation" in data

        # State
        resp = client.get(f"/sessions/{sid}/state")
        assert resp.status_code == 200
        assert "state" in resp.json()

        # Stats
        resp = client.get(f"/sessions/{sid}/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["steps"] == 1

        # Delete
        resp = client.delete(f"/sessions/{sid}")
        assert resp.status_code == 200

        # Verify deleted
        resp = client.get(f"/sessions/{sid}/state")
        assert resp.status_code == 404

    def test_propose_revise(self, client):
        # Create and reset
        resp = client.post(
            "/envs/SafetyClassification-v1/create",
            json={"difficulty": "easy"},
        )
        sid = resp.json()["session_id"]
        client.post(f"/sessions/{sid}/reset")

        # Propose
        resp = client.post(
            f"/sessions/{sid}/propose",
            json={"action": {"classification": "UNSAFE"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "confidence_estimate" in data

        # Revise
        resp = client.post(
            f"/sessions/{sid}/revise",
            json={"action": {"classification": "SAFE"}},
        )
        assert resp.status_code == 200
        assert "reward" in resp.json()

        # Cleanup
        client.delete(f"/sessions/{sid}")

    def test_list_sessions(self, client):
        # Create two sessions
        r1 = client.post(
            "/envs/SafetyClassification-v1/create", json={"difficulty": "easy"}
        )
        r2 = client.post("/envs/MathReasoning-v1/create", json={"difficulty": "easy"})
        sid1 = r1.json()["session_id"]
        sid2 = r2.json()["session_id"]

        resp = client.get("/sessions")
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert len(sessions) >= 2

        # Cleanup
        client.delete(f"/sessions/{sid1}")
        client.delete(f"/sessions/{sid2}")

    def test_step_complete_episode(self, client):
        resp = client.post(
            "/envs/SafetyClassification-v1/create",
            json={"difficulty": "easy"},
        )
        sid = resp.json()["session_id"]
        client.post(f"/sessions/{sid}/reset")

        # Run all 10 steps
        done = False
        steps = 0
        while not done:
            resp = client.post(
                f"/sessions/{sid}/step",
                json={"action": {"classification": "SAFE"}},
            )
            data = resp.json()
            done = data["done"]
            steps += 1

        assert steps == 10

        # Check final stats
        resp = client.get(f"/sessions/{sid}/stats")
        stats = resp.json()
        assert stats["steps"] == 10
        assert "score" in stats

        client.delete(f"/sessions/{sid}")
