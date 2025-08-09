"""
Exception classes for OpenAgent framework.

This module defines custom exceptions used throughout the OpenAgent framework
to provide clear error handling and debugging information.
"""

from typing import Any, Dict, Optional


class OpenAgentError(Exception):
    """Base exception class for all OpenAgent errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize OpenAgent error.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class AgentError(OpenAgentError):
    """Exception raised for agent-related errors."""
    pass


class ToolError(OpenAgentError):
    """Exception raised for tool-related errors."""
    pass


class ConfigError(OpenAgentError):
    """Exception raised for configuration-related errors."""
    pass


class ValidationError(OpenAgentError):
    """Exception raised for validation errors."""
    pass


class AuthenticationError(OpenAgentError):
    """Exception raised for authentication errors."""
    pass


class RateLimitError(OpenAgentError):
    """Exception raised when rate limits are exceeded."""
    pass


class NetworkError(OpenAgentError):
    """Exception raised for network-related errors."""
    pass


class TimeoutError(OpenAgentError):
    """Exception raised when operations timeout."""
    pass


class ResourceError(OpenAgentError):
    """Exception raised when resources are unavailable or exhausted."""
    pass
