"""
Helpers for WebSocket authentication and rate limiting.
"""
from __future__ import annotations

from typing import Optional, Dict, Any, Awaitable, Callable

from .models import WebSocketMessage, MessageType


async def authenticate_on_connect(
    *,
    token: Optional[str],
    authenticate_token: Optional[Callable[[str], Awaitable[Optional[Dict[str, Any]]]]],
) -> Dict[str, Any]:
    """Authenticate a websocket using a bearer token, if provided."""
    user: Optional[Dict[str, Any]] = None
    if token and authenticate_token:
        user = await authenticate_token(token)
    return {"user": user}
