from fastapi.testclient import TestClient
from openagent.server.app import app


def test_sse_streaming_basic():
    # Use context manager to ensure startup/shutdown events run
    with TestClient(app) as client:
        payload = {"message": "Stream this response please.", "agent": "default"}
        with client.stream("POST", "/chat/stream", json=payload) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "")
            chunks = list(r.iter_text())
            # Should include start event, at least one data chunk, and end event
            joined = "".join(chunks)
            assert "event: start" in joined
            assert "data:" in joined
            assert "event: end" in joined

