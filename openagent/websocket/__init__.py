"""
WebSocket Support for OpenAgent

Provides real-time communication capabilities including streaming responses,
live agent interactions, and multi-client orchestration.
"""

from .handler import WebSocketHandler
from .manager import WebSocketManager
from .models import ConnectionInfo, MessageType, WebSocketMessage

__all__ = [
    "WebSocketManager",
    "WebSocketHandler",
    "WebSocketMessage",
    "MessageType",
    "ConnectionInfo",
]
