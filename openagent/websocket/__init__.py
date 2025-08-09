"""
WebSocket Support for OpenAgent

Provides real-time communication capabilities including streaming responses,
live agent interactions, and multi-client orchestration.
"""

from .manager import WebSocketManager
from .handler import WebSocketHandler
from .models import WebSocketMessage, MessageType, ConnectionInfo
from .middleware import WebSocketMiddleware

__all__ = [
    "WebSocketManager",
    "WebSocketHandler", 
    "WebSocketMessage",
    "MessageType",
    "ConnectionInfo",
    "WebSocketMiddleware",
]
