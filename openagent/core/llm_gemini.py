"""
Gemini LLM provider for OpenAgent.

Implements the same interface as HuggingFaceLLM for drop-in usage.
"""
from __future__ import annotations

import os
import asyncio
from typing import Any, Dict, List, Optional

from openagent.core.exceptions import AgentError, ConfigError

try:
    import google.generativeai as genai  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency
    genai = None  # type: ignore


class GeminiLLM:
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_length: int = 4096,
        top_p: float = 0.9,
        top_k: int = 40,
        **kwargs: Any,
    ) -> None:
        if genai is None:
            raise ConfigError("google-generativeai is not installed. Install with: pip install 'openagent[gemini]'")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ConfigError("GEMINI_API_KEY environment variable is not set.")

        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_length = max_length
        self.top_p = top_p
        self.top_k = top_k
        self.model = genai.GenerativeModel(model_name)

    async def generate_response(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        parts: List[str] = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        if context:
            for m in context[-3:]:
                role = (m.get("role") or "user").title()
                parts.append(f"{role}: {m.get('content','')}")
        parts.append(f"User: {prompt}")
        text = "\n".join(parts)

        gen_cfg = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            # Gemini SDK handles length internally; we keep for logical parity
        }

        loop = asyncio.get_event_loop()
        try:
            resp = await loop.run_in_executor(None, lambda: self.model.generate_content(text, generation_config=gen_cfg))
            out = getattr(resp, "text", None) or ""
            return out.strip()
        except Exception as e:
            raise AgentError(f"Gemini generation failed: {e}")

    async def analyze_code(self, code: str, language: str = "python") -> str:
        prompt = f"""
Analyze the following {language} code and provide:
1) Issues and potential bugs
2) Performance improvements
3) Security concerns
4) Refactoring suggestions

```{language}
{code}
```
"""
        return await self.generate_response(prompt, system_prompt="You are an expert code reviewer.")

    async def generate_code(self, description: str, language: str = "python") -> str:
        prompt = f"""
Write {language} code to accomplish the following:

{description}

Provide complete, idiomatic {language} code.
"""
        return await self.generate_response(prompt, system_prompt=f"You are an expert {language} developer.")

    async def explain_command(self, command: str) -> str:
        prompt = f"Explain what this command does and any risks: `{command}`"
        return await self.generate_response(prompt, system_prompt="You are a seasoned systems engineer.")

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_name,
            "device": "cloud",
            "model_type": "chat",
            "loaded": True,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

    async def unload_model(self) -> None:
        # No-op for cloud models
        return

