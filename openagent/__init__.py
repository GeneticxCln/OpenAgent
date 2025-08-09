"""
OpenAgent - A modern, extensible AI agent framework for building intelligent applications.

This package provides a comprehensive framework for creating, managing, and deploying
AI agents with tool integration, multi-agent coordination, and extensible architecture.
"""

from openagent.core.agent import Agent
from openagent.core.base import BaseAgent, BaseTool
from openagent.core.config import Config
from openagent.core.exceptions import OpenAgentError, ToolError, AgentError

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__description__ = "A modern, extensible AI agent framework"

# Main exports
__all__ = [
    "Agent",
    "BaseAgent",
    "BaseTool",
    "Config",
    "OpenAgentError",
    "ToolError", 
    "AgentError",
]

# Package metadata
__package_info__ = {
    "name": "openagent",
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "description": __description__,
    "url": "https://github.com/yourusername/OpenAgent",
}
