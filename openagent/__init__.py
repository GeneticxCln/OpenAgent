"""
OpenAgent - A powerful, production-ready AI agent framework.

Designed as an open-source alternative to Warp AI that can run entirely locally
with customizable AI models. Features Hugging Face models, Ollama integration,
security policy engine, and terminal assistance.

Core Components:
- Agent System: Main Agent implementation with LLM integration
- LLM Integration: Support for Hugging Face transformers and Ollama
- Tools System: CommandExecutor, FileManager, GitTool, SystemInfo
- Security Policy: Command validation and risk assessment
- Terminal Integration: Shell hooks and command interception
- Server: FastAPI web server with WebSocket support
- Plugins: Extensible plugin system

Architecture:
- Agent-Tool Pattern: Agents orchestrate multiple tools
- Policy-Driven Security: All operations go through security validation
- Multi-LLM Support: Unified interface for local and cloud models
- Async-First Design: Full async support for streaming and concurrency
"""

# Core components
from openagent.core.agent import Agent
from openagent.core.base import BaseAgent, BaseMessage, BaseTool, ToolResult
from openagent.core.config import Config
from openagent.core.exceptions import AgentError, OpenAgentError, ToolError
from openagent.core.llm import BaseLLM, HuggingFaceLLM, LLMResponse, get_llm
from openagent.core.observability import get_metrics_collector
from openagent.core.policy import PolicyEngine
from openagent.tools.git import GitTool, RepoGrep

# Tools
from openagent.tools.system import CommandExecutor, FileManager, SystemInfo

__version__ = "0.1.3"
__author__ = "GeneticxCln"
__email__ = "geneticxcln@example.com"
__description__ = (
    "A powerful, production-ready AI agent framework powered by Hugging Face models"
)

# Main exports
__all__ = [
    # Core
    "Agent",
    "BaseAgent",
    "BaseTool",
    "BaseMessage",
    "ToolResult",
    "Config",
    "PolicyEngine",
    # LLM
    "BaseLLM",
    "HuggingFaceLLM",
    "LLMResponse",
    "get_llm",
    # Tools
    "CommandExecutor",
    "FileManager",
    "SystemInfo",
    "GitTool",
    "RepoGrep",
    # Exceptions
    "OpenAgentError",
    "ToolError",
    "AgentError",
    # Observability
    "get_metrics_collector",
]

# Package metadata
__package_info__ = {
    "name": "openagent",
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "description": __description__,
    "url": "https://github.com/GeneticxCln/OpenAgent",
    "documentation": "https://geneticxcln.github.io/OpenAgent",
    "bug_tracker": "https://github.com/GeneticxCln/OpenAgent/issues",
}
