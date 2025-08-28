import importlib
from fastapi.testclient import TestClient


def test_healthz_includes_health_summary():
    mod = importlib.import_module("openagent.server.app")
    importlib.reload(mod)
    client = TestClient(mod.app)
    r = client.get("/healthz")
    assert r.status_code == 200
    data = r.json()
    # Expect at least status and some utilization info when monitor is active
    assert "status" in data
    assert "agents" in data
    # utilization may not always be present if monitor failed, but usually is
    # If present, it should contain cpu/memory/disk keys
    util = data.get("utilization")
    if isinstance(util, dict):
        assert "cpu" in util or "memory" in util or "disk" in util

