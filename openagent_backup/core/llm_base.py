"""
Base LLM protocol for OpenAgent.

Defines a strict interface that all LLM providers must implement so that
Agent and Server layers can rely on consistent return types and behaviors.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional, Protocol


from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    metadata: Dict[str, Any] | None = None
    error: Optional[str] = None


class BaseLLM(Protocol):
    """Protocol that all LLM providers must satisfy."""

    model_name: str

    async def generate_response(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate a non-streaming response. MUST return a plain string."""
        ...

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncIterator[str]:
        """Stream response chunks. MUST yield plain string chunks."""
        ...

    def get_model_info(self) -> Dict[str, Any]:
        ...

    async def unload_model(self) -> None:
        ...
