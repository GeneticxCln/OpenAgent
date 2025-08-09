"""
Core module for OpenAgent framework.

This module contains the foundational classes and utilities that form
the backbone of the OpenAgent framework.
"""

from openagent.core.base import BaseAgent, BaseTool, BaseMessage, ToolResult
from openagent.core.exceptions import (
    OpenAgentError,
    AgentError,
    ToolError,
    ConfigError,
    ValidationError,
    AuthenticationError,
    RateLimitError,
    NetworkError,
    TimeoutError,
    ResourceError,
)

__all__ = [
    # Base classes
    "BaseAgent",
    "BaseTool", 
    "BaseMessage",
    "ToolResult",
    # Exceptions
    "OpenAgentError",
    "AgentError",
    "ToolError",
    "ConfigError",
    "ValidationError",
    "AuthenticationError",
    "RateLimitError",
    "NetworkError",
    "TimeoutError",
    "ResourceError",
]
