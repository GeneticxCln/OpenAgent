import asyncio
import json
from unittest.mock import patch, AsyncMock

import pytest

from openagent.core.llm_ollama import OllamaLLM


@pytest.mark.asyncio
async def test_ollama_generate_non_streaming(monkeypatch):
    class FakeResp:
        def __init__(self):
            self._json = {"response": "hello world"}
        def json(self):
            return self._json
        def raise_for_status(self):
            return None

    class FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False
        async def post(self, url, json=None):
            return FakeResp()

    with patch("httpx.AsyncClient", return_value=FakeClient()):
        llm = OllamaLLM(model_name="llama3")
        resp = await llm.generate_response("hi")
        assert resp.content == "hello world"
        assert resp.metadata["provider"] == "ollama"


@pytest.mark.asyncio
async def test_ollama_generate_streaming(monkeypatch):
    class FakeStreamResp:
        def __init__(self):
            self.lines = [
                json.dumps({"response": "hel"}),
                json.dumps({"response": "lo"}),
                json.dumps({"done": True}),
            ]
            self._i = 0
        def raise_for_status(self):
            return None
        async def aiter_lines(self):
            for line in self.lines:
                yield line

    class FakeStreamCtx:
        async def __aenter__(self):
            return FakeStreamResp()
        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False
        def stream(self, method, url, json=None):
            return FakeStreamCtx()

    with patch("httpx.AsyncClient", return_value=FakeClient()):
        llm = OllamaLLM(model_name="llama3")
        chunks = []
        async for c in llm.stream_generate("hi"):
            chunks.append(c)
        assert "".join(chunks) == "hello"

