import asyncio
import json
import types
import pytest

import openagent.cli as cli


class _MockWS:
    def __init__(self, messages):
        # messages: list of str to be returned by recv()
        self._messages = list(messages)
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def send(self, data: str):
        self.sent.append(data)

    async def recv(self):
        if not self._messages:
            await asyncio.sleep(0)  # yield
            return json.dumps({"event": "end"})
        return self._messages.pop(0)


class _MockHTTPStream:
    def __init__(self, status_code=200, lines=None):
        self.status_code = status_code
        self._lines = lines or []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln

    async def aread(self):
        return b""


class _MockHTTPClient:
    def __init__(self, stream_obj: _MockHTTPStream = None, post_result: dict = None, captured_headers: list = None):
        self._stream_obj = stream_obj
        self._post_result = post_result or {"message": "ok", "metadata": {}}
        self._captured_headers = captured_headers if captured_headers is not None else []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, method: str, url: str, json=None, headers=None):
        # capture headers for assertions
        self._captured_headers.append((method, url, headers or {}))
        return self._stream_obj

    async def post(self, url: str, json=None, headers=None):
        self._captured_headers.append(("POST", url, headers or {}))
        # minimal object with json() and raise_for_status()
        class R:
            status_code = 200
            def json(self_non):
                return self._post_result
            def raise_for_status(self_non):
                if self_non.status_code >= 400:
                    raise RuntimeError("http error")
        return R()


@pytest.mark.asyncio
async def test_chat_loop_ws_success_monkeypatch(monkeypatch):
    # Prepare WS messages: one chunk then end
    ws_messages = [
        json.dumps({"content": "Hello", "event": "chunk"}),
        json.dumps({"event": "end"}),
    ]
    mock_ws = _MockWS(ws_messages)

    # Patch websockets.connect to return our async CM
    ws_module = types.SimpleNamespace(
        connect=lambda *a, **k: mock_ws
    )
    monkeypatch.setattr(cli, "websockets", ws_module, raising=False)

    # Capture console prints
    printed = []
    def _capture_print(*args, **kwargs):
        # emulate rich console printing behavior into buffer
        text = " ".join(str(a) for a in args)
        end = kwargs.get("end", "\n")
        printed.append(text + ("" if end == "" else ""))
    monkeypatch.setattr(cli.console, "print", _capture_print)

    # Make console.input return one user message, then raise KeyboardInterrupt to exit
    inputs = ["hi"]
    def _fake_input(prompt):
        if inputs:
            return inputs.pop(0)
        raise KeyboardInterrupt
    monkeypatch.setattr(cli.console, "input", _fake_input)

    # Run a single iteration
    await cli.chat_loop(use_remote=False, api_url="http://localhost:8000", stream=True, ws=True, ws_path="/ws/chat")

    # Assert payload sent on WS contains our message
    assert mock_ws.sent, "Expected a payload to be sent over WS"
    payload = json.loads(mock_ws.sent[0])
    assert payload.get("message") == "hi"


@pytest.mark.asyncio
async def test_chat_loop_ws_fallback_to_sse(monkeypatch):
    # Make websockets.connect raise to trigger fallback
    ws_module = types.SimpleNamespace(
        connect=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ws"))
    )
    monkeypatch.setattr(cli, "websockets", ws_module, raising=False)

    # Prepare SSE stream with one chunk and end
    stream_lines = [
        "data: {\"content\": \"Hello SSE\"}",
        "event: end",
    ]
    captured_headers = []
    mock_stream = _MockHTTPStream(status_code=200, lines=stream_lines)
    mock_client = _MockHTTPClient(stream_obj=mock_stream, captured_headers=captured_headers)

    # Patch httpx.AsyncClient context manager
    monkeypatch.setattr(cli.httpx, "AsyncClient", lambda timeout=None: mock_client)

    # Capture console prints
    printed = []
    def _capture_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        end = kwargs.get("end", "\n")
        # capture as a continuous stream for assertion simplicity
        printed.append((text, end))
    monkeypatch.setattr(cli.console, "print", _capture_print)

    # Inputs
    inputs = ["hello"]
    def _fake_input(prompt):
        if inputs:
            return inputs.pop(0)
        raise KeyboardInterrupt
    monkeypatch.setattr(cli.console, "input", _fake_input)

    await cli.chat_loop(use_remote=False, api_url="http://localhost:8000", stream=True, ws=True)

    # Ensure SSE endpoint was used (POST /chat/stream)
    assert any(u for m, u, h in captured_headers if u.endswith("/chat/stream")), "Expected POST to /chat/stream"


@pytest.mark.asyncio
async def test_chat_loop_non_streaming_auth_header(monkeypatch):
    # Non-streaming path uses POST /chat and should include Authorization header
    captured = []
    mock_client = _MockHTTPClient(post_result={"message": "ok", "metadata": {"x": 1}}, captured_headers=captured)
    monkeypatch.setattr(cli.httpx, "AsyncClient", lambda timeout=None: mock_client)

    # Disable WS
    monkeypatch.setattr(cli, "websockets", None, raising=False)

    # Fake input then stop
    inputs = ["hello"]
    def _fake_input(prompt):
        if inputs:
            return inputs.pop(0)
        raise KeyboardInterrupt
    monkeypatch.setattr(cli.console, "input", _fake_input)

    # Stub console.print to avoid noise
    monkeypatch.setattr(cli.console, "print", lambda *a, **k: None)

    await cli.chat_loop(use_remote=False, api_url="https://api.example.com", stream=False, ws=False, auth_header_value="Bearer TKN")

    # Verify Authorization header was sent
    assert any(h.get("Authorization") == "Bearer TKN" for m, u, h in captured if m == "POST" and u.endswith("/chat")), "Authorization header missing for non-streaming POST /chat"

