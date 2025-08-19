import asyncio
import pytest

from openagent.core.agent import Agent


class FakeOllamaLLM:
    model_name = "llama3"

    async def generate_response(self, prompt: str, max_new_tokens=None, system_prompt=None, context=None) -> str:
        return "OK: " + prompt[:10]

    async def stream_generate(self, prompt: str, system_prompt=None, context=None):
        # simple 3-chunk stream
        for ch in ["A", "B", "C"]:
            yield ch
            await asyncio.sleep(0)

    def get_model_info(self):
        return {"model_name": self.model_name}

    async def unload_model(self):
        return None


@pytest.mark.asyncio
async def test_agent_with_fake_ollama_non_streaming(monkeypatch):
    # Patch get_llm to return our fake
    from openagent.core import llm as llm_module

    def fake_get_llm(model_name: str = "ollama:llama3", **kwargs):
        return FakeOllamaLLM()

    monkeypatch.setattr(llm_module, "get_llm", fake_get_llm)

    agent = Agent(name="Test", tools=[] , model_name="ollama:llama3")
    resp = await agent.process_message("hello world")
    assert resp.role == "assistant"
    assert isinstance(resp.content, str)
    assert resp.content


@pytest.mark.asyncio
async def test_fake_ollama_streaming_interface():
    # Ensure the streaming interface yields strings
    llm = FakeOllamaLLM()
    out = []
    async for tok in llm.stream_generate("hi"):
        out.append(tok)
    assert out == ["A", "B", "C"]
