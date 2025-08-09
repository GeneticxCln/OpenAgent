"""
WebSocket Manager for OpenAgent

Manages active connections, rooms, and broadcasting for real-time communication.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Optional, Set, Callable, Any, List
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect

from .models import ConnectionInfo, Room, WebSocketMessage, MessageType


@dataclass
class Client:
    websocket: WebSocket
    info: ConnectionInfo


class WebSocketManager:
    """Tracks WebSocket connections and rooms, provides send/broadcast helpers."""

    def __init__(self):
        self._clients: Dict[str, Client] = {}
        self._rooms: Dict[str, Room] = {}
        # connection_id -> set of room_ids
        self._subscriptions: Dict[str, Set[str]] = {}
        # Guard concurrent access
        self._lock = asyncio.Lock()

    # Connection lifecycle
    async def accept(self, websocket: WebSocket, info: ConnectionInfo) -> str:
        await websocket.accept()
        async with self._lock:
            self._clients[info.connection_id] = Client(websocket=websocket, info=info)
            self._subscriptions.setdefault(info.connection_id, set())
        # Send connected ack
        await self.send_message(info.connection_id, WebSocketMessage(type=MessageType.CONNECT, data={
            "connection_id": info.connection_id,
            "status": info.status.value,
        }))
        return info.connection_id

    async def disconnect(self, connection_id: str) -> None:
        async with self._lock:
            client = self._clients.pop(connection_id, None)
            rooms = self._subscriptions.pop(connection_id, set())
        # Remove from rooms
        for room_id in list(rooms):
            await self.leave_room(connection_id, room_id)
        # Best-effort close
        if client:
            try:
                await client.websocket.close()
            except Exception:
                pass

    # Sending helpers
    async def send_text(self, connection_id: str, text: str) -> None:
        client = self._clients.get(connection_id)
        if not client:
            return
        await client.websocket.send_text(text)
        client.info.increment_sent(len(text.encode()))

    async def send_message(self, connection_id: str, message: WebSocketMessage) -> None:
        await self.send_text(connection_id, message.to_json())

    async def broadcast_message(self, message: WebSocketMessage, room_id: Optional[str] = None) -> None:
        if room_id:
            targets = self._get_room_members(room_id)
        else:
            targets = list(self._clients.keys())
        await asyncio.gather(*(self.send_message(cid, message) for cid in targets))

    # Rooms
    def _get_room_members(self, room_id: str) -> List[str]:
        room = self._rooms.get(room_id)
        if not room:
            return []
        return list(room.participants)

    async def create_room(self, name: str, description: str = "", shared_agent: str = "default", max_participants: int = 10) -> Room:
        room = Room(name=name, description=description, shared_agent=shared_agent, max_participants=max_participants)
        async with self._lock:
            self._rooms[room.room_id] = room
        return room

    async def join_room(self, connection_id: str, room_id: str) -> bool:
        async with self._lock:
            client = self._clients.get(connection_id)
            room = self._rooms.get(room_id)
            if not client or not room:
                return False
            ok = room.add_participant(connection_id)
            if ok:
                self._subscriptions.setdefault(connection_id, set()).add(room_id)
                client.info.active_room = room_id
            return ok

    async def leave_room(self, connection_id: str, room_id: str) -> bool:
        async with self._lock:
            client = self._clients.get(connection_id)
            room = self._rooms.get(room_id)
            if not client or not room:
                return False
            ok = room.remove_participant(connection_id)
            self._subscriptions.get(connection_id, set()).discard(room_id)
            if client.info.active_room == room_id:
                client.info.active_room = None
            # Delete empty room
            if room.get_participant_count() == 0:
                self._rooms.pop(room_id, None)
            return ok

    # Utility
    def get_client(self, connection_id: str) -> Optional[Client]:
        return self._clients.get(connection_id)

    def list_connections(self) -> List[ConnectionInfo]:
        return [c.info for c in self._clients.values()]

    def list_rooms(self) -> List[Room]:
        return list(self._rooms.values())
