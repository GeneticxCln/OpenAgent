from fastapi.testclient import TestClient
from openagent.server.app import app

def test_server_health_endpoints():
    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
