import asyncio
import importlib
import json

import pytest
from fastapi.testclient import TestClient


class FakeStreamingAgent:
    def __init__(self, tokens):
        self.tokens = tokens
        self.llm = None

    async def stream_message(self, message: str):
        for t in self.tokens:
            await asyncio.sleep(0)
            yield t


@pytest.fixture()
def app_module():
    mod = importlib.import_module("openagent.server.app")
    importlib.reload(mod)
    return mod


@pytest.fixture()
def client(app_module):
    return TestClient(app_module.app)


def test_sse_chat_streams_incremental_tokens(client, app_module, monkeypatch):
    # Replace default agent with a fake streaming agent
    tokens = ["Hello ", "world", "! "]
    app_module.agents["default"] = FakeStreamingAgent(tokens)

    resp = client.post(
        "/chat/stream", json={"message": "hi"}, headers={"Accept": "text/event-stream"}
    )
    assert resp.status_code == 200
    body = resp.text
    # Expect multiple data: lines for chunks and final end event
    assert "event: start" in body
    assert body.count("data: ") >= len(tokens)
    assert "event: end" in body


def test_ws_chat_streams_incremental_tokens(client, app_module):
    # Replace default agent with a fake streaming agent
    tokens = ["foo ", "bar ", "baz "]
    app_module.agents["default"] = FakeStreamingAgent(tokens)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({"message": "do it"}))
        seen_chunks = 0
        saw_end = False
        for _ in range(10):
            msg = ws.receive_text()
            data = json.loads(msg)
            if data.get("event") == "start":
                continue
            if data.get("event") == "end":
                saw_end = True
                break
            if data.get("content"):
                seen_chunks += 1
        assert seen_chunks >= len(tokens)
        assert saw_end
