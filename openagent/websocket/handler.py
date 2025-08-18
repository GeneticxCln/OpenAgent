"""
WebSocket message handler for OpenAgent

Parses incoming messages and routes them to appropriate operations.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional

from .models import ConnectionInfo, MessageType, WebSocketMessage


class WebSocketHandler:
    """Handle WebSocket messages.

    This class is intentionally framework-agnostic and communicates
    via callbacks you provide when integrating with your server.
    """

    def __init__(
        self,
        *,
        send: Callable[[str, WebSocketMessage], Awaitable[None]],
        # lookup_agent returns an object with an async process_message(message: str, context: dict) -> str
        lookup_agent: Callable[[str], Any],
        authenticate_token: Optional[
            Callable[[str], Awaitable[Optional[Dict[str, Any]]]]
        ] = None,
    ) -> None:
        self._send = send
        self._lookup_agent = lookup_agent
        self._authenticate_token = authenticate_token

    async def handle(
        self, connection_id: str, info: ConnectionInfo, raw_text: str
    ) -> None:
        """Handle a single incoming message from a client."""
        try:
            message = WebSocketMessage.from_json(raw_text)
        except Exception:
            await self._send(
                connection_id,
                WebSocketMessage(
                    type=MessageType.ERROR,
                    data={
                        "error": "INVALID_JSON",
                        "message": "Failed to parse message JSON",
                    },
                ),
            )
            return

        info.increment_received(len(raw_text.encode()))

        # Basic routing
        try:
            if message.type == MessageType.PING:
                await self._send(
                    connection_id,
                    WebSocketMessage.create_response(message, MessageType.PONG, {}),
                )
                return

            if message.type == MessageType.CONNECT:
                # Optional auth via token
                token = message.data.get("token")
                if token and self._authenticate_token:
                    payload = await self._authenticate_token(token)
                    if payload:
                        info.is_authenticated = True
                        info.user_id = payload.get("sub") or info.user_id
                await self._send(
                    connection_id,
                    WebSocketMessage.create_response(
                        message,
                        MessageType.CONNECT,
                        {
                            "connection": info.get_stats(),
                        },
                    ),
                )
                return

            if message.type == MessageType.CHAT:
                agent_name = message.agent or info.current_agent
                agent = self._lookup_agent(agent_name)
                if not agent:
                    await self._send(
                        connection_id,
                        WebSocketMessage.create_error(
                            message,
                            f"Agent '{agent_name}' not found",
                            "AGENT_NOT_FOUND",
                        ),
                    )
                    return
                user_message = message.data.get("message", "")
                context = message.data.get("context", {})
                stream = bool(message.data.get("stream", False))

                if not stream:
                    reply = await agent.process_message(user_message, context)
                    await self._send(
                        connection_id,
                        WebSocketMessage.create_response(
                            message,
                            MessageType.CHAT_RESPONSE,
                            {
                                "message": (
                                    reply
                                    if isinstance(reply, str)
                                    else getattr(reply, "content", str(reply))
                                ),
                                "agent": agent_name,
                            },
                        ),
                    )
                else:
                    # Simple simulated streaming: split by sentences
                    text = await agent.process_message(user_message, context)
                    content = (
                        text
                        if isinstance(text, str)
                        else getattr(text, "content", str(text))
                    )
                    chunks = [c for c in content.split(" ") if c]
                    stream_id = message.id
                    await self._send(
                        connection_id,
                        WebSocketMessage(
                            type=MessageType.CHAT_STREAM,
                            data={"content": ""},
                            stream_id=stream_id,
                            is_stream_start=True,
                        ),
                    )
                    buf = []
                    for i, token in enumerate(chunks, 1):
                        buf.append(token)
                        if len(buf) >= 20 or i == len(chunks):
                            part = " ".join(buf)
                            buf = []
                            await self._send(
                                connection_id,
                                WebSocketMessage(
                                    type=MessageType.CHAT_STREAM,
                                    data={"content": part},
                                    stream_id=stream_id,
                                ),
                            )
                            await asyncio.sleep(0.01)
                    await self._send(
                        connection_id,
                        WebSocketMessage(
                            type=MessageType.CHAT_END,
                            data={},
                            stream_id=stream_id,
                            is_stream_end=True,
                        ),
                    )
                return

            # Unknown type
            await self._send(
                connection_id,
                WebSocketMessage.create_error(
                    message,
                    f"Unsupported message type: {message.type}",
                    "UNSUPPORTED_TYPE",
                ),
            )
        except Exception as e:
            await self._send(
                connection_id,
                WebSocketMessage.create_error(message, str(e), "HANDLER_ERROR"),
            )
