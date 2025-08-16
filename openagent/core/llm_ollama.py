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

    Behavior (local-first hardening):
      - On first use, verifies the Ollama server is reachable at host (/api/version)
      - If unreachable and the 'ollama' binary exists, attempts to auto-start it (best-effort)
      - Waits briefly for readiness; if still unreachable, raises a clear, actionable error
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
        self._server_ready = False

    async def _ensure_server(self) -> None:
        """Ensure the Ollama server is reachable; try to auto-start if not.
        Best-effort: if 'ollama' binary is available, start it in background.
        """
        if self._server_ready:
            return
        base = self.host
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                r = await client.get(f"{base}/api/version")
                r.raise_for_status()
                self._server_ready = True
                return
        except Exception:
            pass
        # Try to auto-start ollama serve (non-blocking) if binary exists
        try:
            import shutil, subprocess, time
            if shutil.which("ollama"):
                # Start only if nothing is listening to avoid duplicate servers
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Wait up to ~4.5s for readiness
                for _ in range(30):
                    try:
                        async with httpx.AsyncClient(timeout=0.3) as client:
                            r = await client.get(f"{base}/api/version")
                            if r.status_code == 200:
                                self._server_ready = True
                                return
                    except Exception:
                        pass
                    time.sleep(0.15)
        except Exception:
            pass
        # Still not ready: raise clear error
        raise RuntimeError(
            f"Ollama server not reachable at {self.host}. Start it with 'ollama serve' and ensure your model ('{self.model_name}') is installed."
        )

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Non-streaming generation via Ollama /api/generate (stream=false)."""
        await self._ensure_server()
        payload = {
            "model": self.model_name,
            "prompt": prompt if not system_prompt else f"{system_prompt}\n\n{prompt}",
            "stream": False,
            "options": self.options,
        }
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                r = await client.post(f"{self.host}/api/generate", json=payload)
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response else None
            if status == 404:
                raise RuntimeError(
                    f"Ollama endpoint returned 404 at {self.host}/api/generate. Is Ollama running and serving the HTTP API on this port?"
                ) from e
            raise
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
        await self._ensure_server()
        payload = {
            "model": self.model_name,
            "prompt": prompt if not system_prompt else f"{system_prompt}\n\n{prompt}",
            "stream": True,
            "options": self.options,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{self.host}/api/generate", json=payload) as resp:
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    status = e.response.status_code if e.response else None
                    if status == 404:
                        raise RuntimeError(
                            f"Ollama endpoint returned 404 at {self.host}/api/generate. Is Ollama running and serving the HTTP API on this port?"
                        ) from e
                    raise
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

