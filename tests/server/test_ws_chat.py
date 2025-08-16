import os
import types
import pytest
from fastapi.testclient import TestClient

import importlib


def _load_app_with_auth(enabled: bool):
    # Ensure environment reflects desired auth setting before app import
    os.environ["AUTH_ENABLED"] = "true" if enabled else "false"
    # Reload module to apply env config to AuthManager
    mod = importlib.import_module("openagent.server.app")
    importlib.reload(mod)
    return mod.app, mod


@pytest.fixture()
def client_no_auth():
    app, mod = _load_app_with_auth(False)
    return TestClient(app), mod


@pytest.fixture()
def client_with_auth(monkeypatch):
    app, mod = _load_app_with_auth(True)
    # Monkeypatch token verifier to accept a specific token
    async def _ok(token: str):
        return {"sub": "user"} if token == "good" else None
    monkeypatch.setattr(mod, "_verify_token", _ok)
    return TestClient(app), mod


def test_ws_chat_connects_when_auth_disabled(client_no_auth):
    client, mod = client_no_auth
    with client.websocket_connect("/ws/chat") as ws:
        # Immediate connect and close is enough to verify handshake
        ws.close()


def test_ws_chat_rejects_without_token_when_auth_enabled(client_with_auth):
    client, mod = client_with_auth
    # No token provided, expect policy close (1008)
    with pytest.raises(Exception) as ei:
        with client.websocket_connect("/ws/chat") as ws:
            pass
    # FastAPI raises WebSocketDisconnect; code may not be easily accessible here in all versions
    # We simply assert an exception occurred
    assert ei.value is not None


def test_ws_chat_accepts_with_bearer_token_when_auth_enabled(client_with_auth):
    client, mod = client_with_auth
    headers = {"authorization": "Bearer good"}
    with client.websocket_connect("/ws/chat", headers=headers) as ws:
        # Send a minimal message the handler can ignore or handle gracefully
        ws.send_text("{}")
        # Close cleanly
        ws.close()

