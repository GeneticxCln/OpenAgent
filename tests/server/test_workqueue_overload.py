import asyncio
import importlib
import json

import pytest
from fastapi.testclient import TestClient


class SlowStreamingAgent:
    def __init__(self, tokens, delay=0.05):
        self.tokens = tokens
        self.delay = delay
        self.llm = None

    async def stream_message(self, message: str):
        # Stream tokens slowly to keep a worker busy
        for t in self.tokens:
            await asyncio.sleep(self.delay)
            yield t


@pytest.fixture()
def app_module(monkeypatch):
    # Configure small WorkQueue to trigger overload easily
    monkeypatch.setenv("OPENAGENT_MAX_WORKERS", "1")
    monkeypatch.setenv("OPENAGENT_QUEUE_SIZE", "1")
    # Reload app to apply env vars
    mod = importlib.import_module("openagent.server.app")
    importlib.reload(mod)
    return mod


@pytest.fixture()
def client(app_module):
    return TestClient(app_module.app)


def test_sse_queue_overload_returns_429(client, app_module):
    # Replace default agent with a slow streaming agent
    tokens = ["a", "b", "c", "d"]
    app_module.agents["default"] = SlowStreamingAgent(tokens, delay=0.1)

    # First streaming request keeps the only worker busy
    first = client.stream("POST", "/chat/stream", json={"message": "1"})
    # Second request should be queued (queue size = 1)
    second = client.stream("POST", "/chat/stream", json={"message": "2"})

    # Third request should fail with 429 due to full queue
    third = client.post("/chat/stream", json={"message": "3"}, headers={"Accept": "text/event-stream"})
    try:
        assert third.status_code == 429
        body = third.json()
        assert body.get("error") == "queue_overloaded"
    finally:
        # Close streams
        first.close()
        second.close()


def test_metrics_exposes_workqueue(client, app_module):
    # Fetch metrics and ensure workqueue metrics are present
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    # Prometheus names or fallback
    assert (
        "openagent_workqueue_active_requests" in body
        or "workqueue_active_requests" in body
    )

