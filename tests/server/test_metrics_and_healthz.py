import importlib
import time

from fastapi.testclient import TestClient


def test_metrics_endpoint_prometheus_output():
    mod = importlib.import_module("openagent.server.app")
    importlib.reload(mod)
    with TestClient(mod.app) as client:
        r = client.get("/metrics")
        assert r.status_code == 200
        ctype = r.headers.get("content-type", "")
        assert "text/plain" in ctype
        body = r.content.decode("utf-8", errors="ignore")
        # Expect our namespace to appear and at least one resource metric
        assert "openagent_" in body
        assert ("openagent_cpu_percent" in body) or ("openagent_memory_percent" in body)


def test_healthz_schema_with_alerts_and_utilization():
    mod = importlib.import_module("openagent.server.app")
    importlib.reload(mod)
    with TestClient(mod.app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
        # Core fields
        assert "status" in data
        assert "version" in data
        assert "agents" in data
        # Utilization
        util = data.get("utilization")
        if isinstance(util, dict):
            # At least one of these should be present
            assert any(k in util for k in ("cpu", "memory", "disk"))
        # Alerts shape
        alerts = data.get("alerts")
        if isinstance(alerts, dict):
            by_level = alerts.get("by_level")
            if isinstance(by_level, dict):
                # Ensure keys exist even if zero
                for lvl in ("info", "warning", "critical"):
                    assert lvl in by_level
        # Uptime should be a non-negative number if present
        uptime = data.get("uptime")
        if uptime is not None:
            assert isinstance(uptime, (int, float))
            assert uptime >= 0


def test_resource_monitor_startup_shutdown_lifecycle():
    # Load fresh module state
    mod = importlib.import_module("openagent.server.app")
    importlib.reload(mod)

    # Start server and hit healthz
    with TestClient(mod.app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        # Call twice to let monitor collect at least once
        time.sleep(0.1)
        r2 = client.get("/healthz")
        assert r2.status_code == 200

    # After TestClient exits, server shutdown should have been invoked
    rm = getattr(mod, "resource_monitor", None)
    assert rm is not None
    # Its shutdown event should be set
    # Note: _shutdown_event is an internal detail but sufficient for lifecycle verification
    assert getattr(rm, "_shutdown_event").is_set()
