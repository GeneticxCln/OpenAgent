"""
Pydantic models for OpenAgent API requests and responses.

Provides type-safe data models for all API endpoints with validation
and documentation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Base models
class BaseRequest(BaseModel):
    """Base class for all API requests."""

    agent: Optional[str] = Field(
        None, description="Name of the agent to use (defaults to 'default')"
    )


class BaseResponse(BaseModel):
    """Base class for all API responses."""

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of the response"
    )


# Authentication models
class User(BaseModel):
    """User model for authentication."""

    id: str
    username: str
    email: Optional[str] = None
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    roles: List[str] = Field(
        default_factory=list,
        description="List of roles for RBAC (e.g., ['admin','user'])",
    )


class LoginRequest(BaseModel):
    """Login request model."""

    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class LoginResponse(BaseResponse):
    """Login response model."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    user: User = Field(..., description="User information")
    expires_in: int = Field(3600, description="Token expiration time in seconds")


# Chat models
class ChatRequest(BaseRequest):
    """Chat request model."""

    message: str = Field(..., description="Message to send to the agent")
    context: Optional[Dict[str, Any]] = Field(
        None, description="Additional context for the message"
    )
    stream: bool = Field(False, description="Whether to stream the response")


class ChatResponse(BaseResponse):
    """Chat response model."""

    message: str = Field(..., description="Agent's response message")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the response"
    )
    processing_time: float = Field(..., description="Time taken to process the request")
    agent: str = Field(..., description="Name of the agent that processed the request")
    tools_used: List[str] = Field(
        default_factory=list, description="List of tools used to generate the response"
    )


# Code generation models
class CodeRequest(BaseRequest):
    """Code generation request model."""

    description: str = Field(..., description="Description of the code to generate")
    language: str = Field("python", description="Programming language")
    style: Optional[str] = Field(
        None,
        description="Code style preferences (e.g., 'functional', 'object-oriented')",
    )
    include_tests: bool = Field(False, description="Whether to include unit tests")
    include_docs: bool = Field(True, description="Whether to include documentation")


class CodeResponse(BaseResponse):
    """Code generation response model."""

    code: str = Field(..., description="Generated code")
    language: str = Field(..., description="Programming language")
    description: str = Field(..., description="Description of the generated code")
    processing_time: float = Field(..., description="Time taken to generate the code")
    suggestions: List[str] = Field(
        default_factory=list, description="Additional suggestions or improvements"
    )


# Code analysis models
class AnalysisRequest(BaseRequest):
    """Code analysis request model."""

    code: str = Field(..., description="Code to analyze")
    language: str = Field("python", description="Programming language")
    focus: Optional[List[str]] = Field(
        None,
        description="Specific aspects to focus on (e.g., 'security', 'performance', 'style')",
    )


class AnalysisResponse(BaseResponse):
    """Code analysis response model."""

    analysis: str = Field(..., description="Analysis results")
    language: str = Field(..., description="Programming language")
    processing_time: float = Field(..., description="Time taken to analyze the code")
    issues: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of identified issues"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Code quality metrics"
    )


# System models
class SystemInfoResponse(BaseResponse):
    """System information response model."""

    content: str = Field(..., description="System information content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional system metadata"
    )


# Agent models
class AgentStatus(BaseModel):
    """Agent status model."""

    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    model: str = Field(..., description="LLM model being used")
    is_processing: bool = Field(
        ..., description="Whether the agent is currently processing"
    )
    tools: List[str] = Field(..., description="List of available tools")
    message_count: int = Field(
        ..., description="Number of messages in conversation history"
    )
    uptime: float = Field(default=0.0, description="Agent uptime in seconds")
    memory_usage: Optional[float] = Field(None, description="Memory usage in MB")


class AgentCreateRequest(BaseModel):
    """Agent creation request model."""

    name: str = Field(..., description="Agent name")
    description: str = Field("", description="Agent description")
    model_name: str = Field("tiny-llama", description="LLM model to use")
    tools: List[str] = Field(
        default_factory=list, description="List of tools to enable"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Agent configuration"
    )


# Model models
class ModelInfo(BaseModel):
    """Model information model."""

    name: str = Field(..., description="Model name")
    path: str = Field(..., description="Hugging Face model path")
    category: str = Field(..., description="Model category (code, chat, lightweight)")
    description: str = Field(..., description="Model description")
    size: Optional[str] = Field(None, description="Model size (e.g., '7B', '13B')")
    parameters: Optional[int] = Field(None, description="Number of parameters")
    memory_required: Optional[str] = Field(
        None, description="Estimated memory requirement"
    )
    is_loaded: bool = Field(False, description="Whether the model is currently loaded")


# File operation models
class FileRequest(BaseRequest):
    """File operation request model."""

    operation: str = Field(..., description="File operation type")
    path: str = Field(..., description="File or directory path")
    content: Optional[str] = Field(None, description="Content for write operations")
    destination: Optional[str] = Field(
        None, description="Destination for copy/move operations"
    )


class FileResponse(BaseResponse):
    """File operation response model."""

    operation: str = Field(..., description="Performed operation")
    path: str = Field(..., description="File or directory path")
    success: bool = Field(..., description="Operation success status")
    content: Optional[str] = Field(
        None, description="File content (for read operations)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional file metadata"
    )


# Command execution models
class CommandRequest(BaseRequest):
    """Command execution request model."""

    command: str = Field(..., description="Shell command to execute")
    working_dir: Optional[str] = Field(None, description="Working directory")
    timeout: int = Field(30, description="Command timeout in seconds")
    explain_only: bool = Field(
        True, description="Only explain the command without executing"
    )


class CommandResponse(BaseResponse):
    """Command execution response model."""

    command: str = Field(..., description="Executed command")
    output: str = Field(..., description="Command output")
    exit_code: int = Field(..., description="Command exit code")
    execution_time: float = Field(..., description="Execution time in seconds")
    explained: bool = Field(False, description="Whether the command was only explained")


# Error models
class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


# Webhook models
class WebhookEvent(BaseModel):
    """Webhook event model."""

    event_type: str = Field(..., description="Type of event")
    timestamp: datetime = Field(..., description="Event timestamp")
    data: Dict[str, Any] = Field(..., description="Event data")
    source: str = Field("openagent", description="Event source")


# Batch processing models
class BatchRequest(BaseRequest):
    """Batch request model."""

    requests: List[Dict[str, Any]] = Field(
        ..., description="List of requests to process"
    )
    parallel: bool = Field(True, description="Whether to process requests in parallel")
    max_workers: int = Field(5, description="Maximum number of parallel workers")


class BatchResponse(BaseResponse):
    """Batch response model."""

    results: List[Dict[str, Any]] = Field(..., description="List of results")
    success_count: int = Field(..., description="Number of successful operations")
    error_count: int = Field(..., description="Number of failed operations")
    total_processing_time: float = Field(..., description="Total processing time")


# Plugin models
class PluginInfo(BaseModel):
    """Plugin information model."""

    name: str = Field(..., description="Plugin name")
    version: str = Field(..., description="Plugin version")
    description: str = Field(..., description="Plugin description")
    author: str = Field(..., description="Plugin author")
    enabled: bool = Field(False, description="Whether the plugin is enabled")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Plugin configuration"
    )


# Health check model
class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service health status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")
    agents: int = Field(..., description="Number of active agents")
    uptime: float = Field(..., description="Service uptime in seconds")
    system_info: Dict[str, Any] = Field(
        default_factory=dict, description="System information"
    )
