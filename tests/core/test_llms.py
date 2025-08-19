import asyncio
import pytest

from openagent.core.llm_ollama import OllamaLLM


@pytest.mark.asyncio
async def test_ollama_generate_response_returns_str(monkeypatch):
    class Dummy(OllamaLLM):
        async def _ensure_server(self) -> None:
            return
        async def generate_response(self, prompt: str, system_prompt=None, **kw) -> str:
            return "hello world"

    llm = Dummy(model_name="llama3")
    out = await llm.generate_response("hi")
    assert isinstance(out, str)


@pytest.mark.asyncio
async def test_ollama_stream_generate_yields_str(monkeypatch):
    class Dummy(OllamaLLM):
        async def _ensure_server(self) -> None:
            return
        async def stream_generate(self, prompt: str, system_prompt=None, **kw):
            for ch in ["a", "b", "c"]:
                yield ch
                await asyncio.sleep(0)

    llm = Dummy(model_name="llama3")
    toks = []
    async for t in llm.stream_generate("hi"):
        toks.append(t)
    assert toks == ["a", "b", "c"]
