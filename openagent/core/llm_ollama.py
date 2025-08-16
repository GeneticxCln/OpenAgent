"""
Ollama LLM provider for OpenAgent.

Supports local generation and token streaming via the Ollama HTTP API.
Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncIterator, Dict, Optional

import httpx


async def get_default_ollama_model(host: Optional[str] = None) -> Optional[str]:
    """Return the first installed Ollama model tag (e.g., 'llama3') or None.
    Uses /api/tags.
    """
    base = host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{base}/api/tags")
            r.raise_for_status()
            data = r.json() or {}
            models = data.get("models") or []
            if not models:
                return None
            # names may include ':latest' suffix; strip tag
            name = models[0].get("name", "")
            return name.split(":", 1)[0] if name else None
    except Exception:
        return None

class OllamaLLM:
    """Ollama LLM provider using the local Ollama server.

    Usage:
      - model_name: the Ollama model tag, e.g., "llama3", "mistral", "qwen2.5"
      - host: Ollama server base URL (default http://127.0.0.1:11434)
    """

    def __init__(
        self,
        model_name: str = "llama3",
        host: Optional[str] = None,
        temperature: float = 0.7,
        max_length: int = 2048,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.host = host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.temperature = temperature
        self.max_length = max_length
        self.options = options or {}
        # Normalize options to include sampling params if not present
        self.options.setdefault("temperature", self.temperature)
        self.options.setdefault("num_predict", self.max_length)
        # No heavy init required; Ollama runs as a local server

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Non-streaming generation via Ollama /api/generate (stream=false)."""
        payload = {
            "model": self.model_name,
            "prompt": prompt if not system_prompt else f"{system_prompt}\n\n{prompt}",
            "stream": False,
            "options": self.options,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(f"{self.host}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
        # Ollama returns { response: str, ... }
        class R:
            pass
        resp = R()
        resp.content = data.get("response", "")
        resp.metadata = {"provider": "ollama", "model": self.model_name}
        return resp

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Streaming generation via Ollama /api/generate (stream=true)."""
        payload = {
            "model": self.model_name,
            "prompt": prompt if not system_prompt else f"{system_prompt}\n\n{prompt}",
            "stream": True,
            "options": self.options,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{self.host}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        chunk = obj.get("response")
                        if chunk:
                            yield chunk
                        if obj.get("done"):
                            break
                    except Exception:
                        # Ignore malformed lines
                        continue
        await asyncio.sleep(0)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": "ollama-local",
            "provider": "ollama",
            "host": self.host,
            "loaded": True,
            "max_length": self.max_length,
            "temperature": self.temperature,
        }

    async def unload_model(self) -> None:
        # Ollama manages model lifecycle; nothing to unload on client side
        return None

