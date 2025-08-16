"""
WebSocket Models for OpenAgent

Defines message types, connection info, and data structures
for real-time communication.
"""

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class MessageType(Enum):
    """WebSocket message types."""
    # Connection management
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"
    
    # Agent interactions
    CHAT = "chat"
    CHAT_RESPONSE = "chat_response"
    CHAT_STREAM = "chat_stream"
    CHAT_END = "chat_end"
    
    # Code operations
    CODE_GENERATE = "code_generate"
    CODE_ANALYZE = "code_analyze"
    CODE_RESPONSE = "code_response"
    
    # System operations
    SYSTEM_INFO = "system_info"
    AGENT_STATUS = "agent_status"
    PLUGIN_STATUS = "plugin_status"
    
    # Multi-agent
    AGENT_CREATE = "agent_create"
    AGENT_DELETE = "agent_delete"
    AGENT_SWITCH = "agent_switch"
    
    # Real-time updates
    STATUS_UPDATE = "status_update"
    PROGRESS_UPDATE = "progress_update"
    NOTIFICATION = "notification"
    
    # Error handling
    ERROR = "error"
    WARNING = "warning"
    
    # Collaboration
    BROADCAST = "broadcast"
    ROOM_JOIN = "room_join"
    ROOM_LEAVE = "room_leave"
    
    # Custom
    CUSTOM = "custom"


class ConnectionStatus(Enum):
    """Connection status states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class WebSocketMessage(BaseModel):
    """WebSocket message structure."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = Field(..., description="Message type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Optional fields
    agent: Optional[str] = Field(None, description="Target agent name")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    room_id: Optional[str] = Field(None, description="Room identifier for collaboration")
    
    # Response correlation
    request_id: Optional[str] = Field(None, description="ID of the request this responds to")
    is_response: bool = Field(False, description="Whether this is a response message")
    
    # Streaming support
    stream_id: Optional[str] = Field(None, description="Stream identifier for chunked responses")
    is_stream_start: bool = Field(False, description="Start of a streamed response")
    is_stream_end: bool = Field(False, description="End of a streamed response")
    
    class Config:
        use_enum_values = True
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return self.json()
    
    @classmethod
    def from_json(cls, json_str: str) -> "WebSocketMessage":
        """Create message from JSON string."""
        return cls.parse_raw(json_str)
    
    @classmethod
    def create_response(
        cls,
        request_message: "WebSocketMessage",
        response_type: MessageType,
        data: Dict[str, Any]
    ) -> "WebSocketMessage":
        """Create a response message for a request."""
        return cls(
            type=response_type,
            data=data,
            agent=request_message.agent,
            user_id=request_message.user_id,
            session_id=request_message.session_id,
            room_id=request_message.room_id,
            request_id=request_message.id,
            is_response=True
        )
    
    @classmethod
    def create_error(
        cls,
        request_message: Optional["WebSocketMessage"] = None,
        error_msg: str = "Unknown error",
        error_code: Optional[str] = None
    ) -> "WebSocketMessage":
        """Create an error message."""
        data = {
            "error": error_msg,
            "code": error_code or "UNKNOWN_ERROR"
        }
        
        if request_message:
            return cls.create_response(request_message, MessageType.ERROR, data)
        else:
            return cls(type=MessageType.ERROR, data=data)


class ConnectionInfo(BaseModel):
    """Information about a WebSocket connection."""
    connection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Connection details
    status: ConnectionStatus = ConnectionStatus.CONNECTING
    connected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Client information
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Agent context
    current_agent: str = "default"
    active_room: Optional[str] = None
    
    # Statistics
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    
    # Authentication
    is_authenticated: bool = False
    auth_token: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)
    
    def increment_sent(self, byte_count: int = 0) -> None:
        """Increment sent message counter."""
        self.messages_sent += 1
        self.bytes_sent += byte_count
        self.update_activity()
    
    def increment_received(self, byte_count: int = 0) -> None:
        """Increment received message counter."""
        self.messages_received += 1
        self.bytes_received += byte_count
        self.update_activity()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        uptime = (datetime.now(timezone.utc) - self.connected_at).total_seconds()
        
        return {
            "connection_id": self.connection_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "uptime": uptime,
            "last_activity": self.last_activity.isoformat(),
            "messages": {
                "sent": self.messages_sent,
                "received": self.messages_received,
                "total": self.messages_sent + self.messages_received
            },
            "bytes": {
                "sent": self.bytes_sent,
                "received": self.bytes_received,
                "total": self.bytes_sent + self.bytes_received
            },
            "agent": self.current_agent,
            "room": self.active_room,
            "authenticated": self.is_authenticated
        }


class StreamChunk(BaseModel):
    """A chunk in a streamed response."""
    stream_id: str
    chunk_id: int
    content: str
    is_final: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentInteraction(BaseModel):
    """Represents an agent interaction session."""
    interaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str
    user_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_message_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Message history
    message_count: int = 0
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # Status
    is_active: bool = True
    is_streaming: bool = False
    
    def add_message(self, message: WebSocketMessage) -> None:
        """Add a message to this interaction."""
        self.message_count += 1
        self.last_message_at = datetime.now(timezone.utc)
    
    def get_duration(self) -> float:
        """Get interaction duration in seconds."""
        return (self.last_message_at - self.started_at).total_seconds()


class Room(BaseModel):
    """Collaboration room for multiple users."""
    room_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    
    # Participants
    participants: List[str] = Field(default_factory=list)  # connection_ids
    max_participants: int = 10
    
    # Room state
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    is_private: bool = False
    
    # Shared context
    shared_agent: str = "default"
    shared_context: Dict[str, Any] = Field(default_factory=dict)
    
    def add_participant(self, connection_id: str) -> bool:
        """Add a participant to the room."""
        if len(self.participants) >= self.max_participants:
            return False
        
        if connection_id not in self.participants:
            self.participants.append(connection_id)
        
        return True
    
    def remove_participant(self, connection_id: str) -> bool:
        """Remove a participant from the room."""
        try:
            self.participants.remove(connection_id)
            return True
        except ValueError:
            return False
    
    def get_participant_count(self) -> int:
        """Get number of participants."""
        return len(self.participants)
