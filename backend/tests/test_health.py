import pytest


class TestHealth:
    """Tests for GET /health endpoint."""

    def test_health_returns_200(self, test_client):
        r = test_client.get("/health")
        assert r.status_code == 200

    def test_health_response_structure(self, test_client):
        r    = test_client.get("/health")
        data = r.json()
        assert "status"   in data
        assert "model"    in data
        assert "rag"      in data
        assert "version"  in data
        assert "uptime_s" in data

    def test_health_model_ready(self, test_client):
        r    = test_client.get("/health")
        data = r.json()
        assert data["model"] == "ready", \
            f"Model not ready: {data['model']}"

    def test_health_rag_ready(self, test_client):
        r    = test_client.get("/health")
        data = r.json()
        assert data["rag"] == "ready", \
            f"RAG not ready: {data['rag']}"

    def test_health_status_ok(self, test_client):
        r    = test_client.get("/health")
        data = r.json()
        assert data["status"] == "ok"

    def test_health_uptime_positive(self, test_client):
        r    = test_client.get("/health")
        data = r.json()
        assert data["uptime_s"] > 0

    def test_root_endpoint(self, test_client):
        r    = test_client.get("/")
        data = r.json()
        assert r.status_code == 200
        assert "KisanMitra" in data["message"]
        assert "docs"  in data
        assert "health" in data