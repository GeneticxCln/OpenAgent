"""
OpenAgent Core Module

This module contains the core functionality of OpenAgent including:
- Agent implementation and base classes
- LLM integration (Hugging Face, Ollama)
- Security policy engine
- Configuration management
- Context and history management
- Observability and metrics
- Workflow management
"""

# Try to import core components, but don't fail if dependencies are missing
try:
    from .agent import Agent
except ImportError:
    Agent = None

try:
    from .base import BaseAgent, BaseMessage, BaseTool, ToolResult
except ImportError:
    BaseAgent = BaseTool = BaseMessage = ToolResult = None

try:
    from .config import Config
except ImportError:
    Config = None

try:
    from .context import SystemContext, gather_context
except ImportError:
    SystemContext = gather_context = None

try:
    from .exceptions import AgentError, ConfigError, OpenAgentError, ToolError
except ImportError:
    AgentError = OpenAgentError = ToolError = ConfigError = None

try:
    from .history import HistoryManager
except ImportError:
    HistoryManager = None

try:
    from .llm import BaseLLM, HuggingFaceLLM, LLMResponse, get_llm
except ImportError:
    BaseLLM = HuggingFaceLLM = LLMResponse = get_llm = None

try:
    from .llm_ollama import OllamaLLM
except ImportError:
    OllamaLLM = None

try:
    from .observability import MetricsCollector, get_metrics_collector
except ImportError:
    MetricsCollector = get_metrics_collector = None

try:
    from .policy import PolicyEngine
except ImportError:
    PolicyEngine = None

try:
    from .workflows import WorkflowManager
except ImportError:
    WorkflowManager = None

# Only export components that were successfully imported
__all__ = [
    name
    for name in [
        # Core components
        "Agent",
        "BaseAgent",
        "BaseTool",
        "BaseMessage",
        "ToolResult",
        "Config",
        # LLM
        "BaseLLM",
        "HuggingFaceLLM",
        "OllamaLLM",
        "LLMResponse",
        "get_llm",
        # Context and History
        "SystemContext",
        "gather_context",
        "HistoryManager",
        # Policy and Security
        "PolicyEngine",
        # Observability
        "MetricsCollector",
        "get_metrics_collector",
        # Workflows
        "WorkflowManager",
        # Exceptions
        "AgentError",
        "OpenAgentError",
        "ToolError",
        "ConfigError",
    ]
    if globals().get(name) is not None
]
