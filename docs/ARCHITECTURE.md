# OpenAgent Architecture

## Overview

OpenAgent is architected as a modular, production-ready AI agent framework that implements the **Agent-Tool Pattern** with **Policy-Driven Security**. The architecture prioritizes maintainability, extensibility, and production readiness.

## Core Principles

### 🏗️ **Agent-Tool Pattern**
- **Agents** orchestrate multiple tools to complete complex tasks
- **Tools** are composable and can be combined in different ways
- Clear separation of concerns between orchestration and execution

### 🔒 **Policy-Driven Security**
- All command execution goes through a centralized policy engine
- Commands can be ALLOWED, DENIED, EXPLAINED_ONLY, or REQUIRE_APPROVAL
- Risk assessment based on configurable security rules

### 🔄 **Multi-LLM Support**
- Unified interface supporting both Hugging Face transformers (local) and Ollama (local)
- Factory pattern for LLM creation with graceful fallbacks
- Protocol-based design for easy addition of new providers

### ⚡ **Async-First Design**
- All core operations are async to support streaming responses
- Concurrent tool execution and non-blocking operations
- Built for performance and scalability

## Directory Structure

```
openagent/
├── __init__.py                 # Main package exports
├── cli.py                      # CLI interface
├── py.typed                    # Type hint marker
│
├── core/                       # Core agent system
│   ├── agent.py               # Main Agent implementation
│   ├── base.py                # Abstract base classes (BaseAgent, BaseTool, etc.)
│   ├── llm.py                 # Unified LLM integration (HF + Ollama + protocols)
│   ├── llm_ollama.py          # Ollama-specific implementation
│   ├── policy.py              # Security policy engine
│   ├── observability.py       # Metrics collection, logging, request tracking
│   ├── workflows.py           # Workflow management for complex multi-step tasks
│   ├── config.py              # Configuration management
│   ├── exceptions.py          # Custom exceptions
│   ├── context.py             # Context management
│   ├── history.py             # History management
│   └── performance/           # Performance optimization
│       ├── memory_manager.py
│       ├── model_cache.py
│       └── optimization.py
│
├── tools/                     # Tools system
│   ├── system.py             # CommandExecutor, FileManager, SystemInfo
│   ├── git.py                # GitTool and RepoGrep
│   └── base.py               # Tool base classes
│
├── server/                    # Server components
│   ├── app.py                # FastAPI web server with REST API and streaming
│   ├── auth.py               # Authentication manager with token-based auth
│   ├── rate_limiter.py       # Request rate limiting and quota management
│   ├── models.py             # Pydantic models for API request/response schemas
│   └── versioning.py         # API versioning
│
├── terminal/                  # Terminal integration
│   ├── integration.py        # Shell integration for command interception
│   └── validator.py          # Command validation and security policy enforcement
│
├── websocket/                 # WebSocket support
│   ├── handler.py
│   ├── manager.py
│   ├── middleware.py
│   └── models.py
│
├── plugins/                   # Plugin system
│   ├── base.py
│   ├── loader.py
│   ├── manager.py
│   ├── registry.py
│   └── validator.py
│
├── ui/                        # User interface components
│   ├── blocks.py
│   ├── formatting.py
│   └── renderer.py
│
└── utils/                     # Utilities
    └── subprocess_utils.py
```

## Core Components

### 🤖 Agent System (`openagent/core/`)

**`agent.py`** - Main Agent Implementation
- Orchestrates LLM interactions and tool execution
- Manages conversation context and history
- Implements the core agent lifecycle

**`base.py`** - Abstract Base Classes
- `BaseAgent`: Protocol for all agent implementations
- `BaseTool`: Abstract base class for all tools
- `BaseMessage`: Standard message format
- `ToolResult`: Standardized tool execution results

**`llm.py`** - Unified LLM Integration
- Supports Hugging Face transformers (local execution)
- Supports Ollama (local execution)
- Protocol-based design for extensibility
- Graceful degradation when dependencies are missing

**`policy.py`** - Security Policy Engine
- Command validation and risk assessment
- Configurable security policies
- Audit trail for compliance

**`observability.py`** - Metrics and Monitoring
- Prometheus metrics collection
- Request tracking and performance monitoring
- Structured logging for production environments

### 🛠️ Tools System (`openagent/tools/`)

**`system.py`** - Core System Tools
- `CommandExecutor`: Safe command execution with policy validation
- `FileManager`: File operations with path restrictions
- `SystemInfo`: System information gathering

**`git.py`** - Version Control Tools
- `GitTool`: Git repository operations
- `RepoGrep`: Code search and analysis

All tools implement the `BaseTool` interface with:
- Async `execute()` method returning `ToolResult`
- Input validation via `validate_input()`
- Clear error handling and logging

### 🌐 Server Components (`openagent/server/`)

**`app.py`** - FastAPI Web Server
- REST API endpoints
- WebSocket support for real-time streaming
- Integration with the core agent system

**`auth.py`** - Authentication Manager
- Token-based authentication
- Role-based access control (RBAC)
- Session management

**`rate_limiter.py`** - Rate Limiting
- Request rate limiting per user/IP
- Quota management
- DoS protection

### 🔌 Plugin System (`openagent/plugins/`)

The plugin system allows for extensible functionality:

- **Dynamic Loading**: Plugins can be loaded at runtime
- **Validation**: All plugins are validated before execution
- **Registry**: Central registry for plugin management
- **Isolation**: Plugins run in isolated environments

## Design Patterns

### Factory Pattern - LLM Creation
```python
from openagent.core.llm import get_llm

# Factory creates appropriate LLM instance
llm = get_llm("codellama-7b")  # -> HuggingFaceLLM
llm = get_llm("ollama:llama3")  # -> OllamaLLM
```

### Strategy Pattern - Tool Selection
```python
from openagent.core.tool_selector import SmartToolSelector

selector = SmartToolSelector()
tools = selector.select_tools_for_task(task_description)
```

### Observer Pattern - Event System
```python
from openagent.core.observability import get_metrics_collector

metrics = get_metrics_collector()
metrics.record_agent_message(agent_id, message_type, success)
```

## Security Architecture

### Policy Engine
- **Risk Assessment**: Commands are classified by risk level
- **Policy Rules**: Configurable rules for different environments
- **Audit Trail**: All policy decisions are logged
- **Fallback Mechanisms**: Safe defaults when policies are unclear

### Input Validation
- **Sanitization**: All inputs are sanitized before processing
- **Type Checking**: Strong typing with Pydantic models
- **Bounds Checking**: Limits on input sizes and ranges

### Secret Management
- **Redaction**: Automatic redaction of secrets in logs
- **Environment Variables**: Secure storage of sensitive configuration
- **Token Rotation**: Support for token refresh and rotation

## Performance Optimizations

### Memory Management
- **Model Caching**: Intelligent caching of loaded models
- **Memory Monitoring**: Real-time memory usage tracking
- **Garbage Collection**: Proactive cleanup of unused resources

### Async Operations
- **Non-blocking I/O**: All I/O operations are async
- **Concurrent Execution**: Tools can run concurrently when safe
- **Streaming**: Support for streaming responses

### Fallback Mechanisms
- **Fast Responses**: Common queries have pre-computed responses
- **CPU Fallback**: Automatic fallback to CPU when GPU memory is insufficient
- **Graceful Degradation**: System remains functional when optional components fail

## Configuration Management

### Environment-Based Configuration
```bash
# Core settings
OPENAGENT_MODEL=codellama-7b
OPENAGENT_DEVICE=auto
OPENAGENT_EXPLAIN_ONLY=false

# Security settings
OPENAGENT_EXEC_STRICT=true
OPENAGENT_POLICY_FILE=~/.config/openagent/policy.yaml

# Performance settings
OPENAGENT_LOAD_IN_4BIT=true
OPENAGENT_MAX_MEMORY=8G
```

### Policy Configuration
```yaml
# ~/.config/openagent/policy.yaml
safe_paths:
  - /home/user/projects
  - /tmp

restricted_paths:
  - /etc
  - /root
  - /usr/bin

command_policies:
  - pattern: "rm -rf .*"
    action: DENY
    reason: "Dangerous recursive delete"
```

## Monitoring and Observability

### Metrics Collection
- **Request Metrics**: Latency, throughput, error rates
- **Model Metrics**: Inference time, memory usage, cache hits
- **Policy Metrics**: Policy decisions, blocked commands, approval requests

### Logging
- **Structured Logging**: JSON-formatted logs for easy parsing
- **Log Levels**: Configurable log levels for different environments
- **Audit Logs**: Security-focused audit trail

### Health Checks
- **Model Status**: Monitor model availability and performance
- **Memory Usage**: Track memory consumption and leaks
- **Response Time**: Monitor system responsiveness

## Testing Strategy

### Unit Tests
- **Core Components**: Comprehensive unit tests for all core modules
- **Mock Dependencies**: Mock external dependencies for isolated testing
- **Edge Cases**: Test error conditions and edge cases

### Integration Tests
- **End-to-End**: Full workflow testing from CLI to tool execution
- **API Testing**: REST API and WebSocket functionality
- **Performance Tests**: Load testing and performance benchmarks

### Security Tests
- **Policy Validation**: Test security policy enforcement
- **Input Validation**: Test input sanitization and validation
- **Authentication**: Test authentication and authorization

## Deployment Considerations

### Production Deployment
- **Container Ready**: Designed for containerized deployment
- **Horizontal Scaling**: Support for multiple instances
- **Load Balancing**: Compatible with standard load balancers

### Configuration Management
- **Environment Variables**: All configuration via environment variables
- **Secrets Management**: Integration with secret management systems
- **Feature Flags**: Support for feature toggles

### Monitoring Integration
- **Prometheus**: Native Prometheus metrics support
- **Health Endpoints**: Standard health check endpoints
- **Structured Logs**: ELK/EFK stack compatible logging

## Future Architecture Plans

### Plugin Ecosystem
- **Plugin Registry**: Central registry for community plugins
- **Plugin Marketplace**: Distribution platform for plugins
- **Plugin Templates**: Scaffolding for new plugin development

### Advanced Security
- **Sandboxing**: Improved isolation for command execution
- **Zero-Trust**: Zero-trust security model implementation
- **Compliance**: SOC2, GDPR compliance features

### Performance Enhancements
- **Distributed Processing**: Support for distributed agent execution
- **Edge Deployment**: Lightweight edge deployment options
- **Model Optimization**: Advanced model optimization techniques

This architecture provides a solid foundation for building powerful AI agents while maintaining security, performance, and maintainability.
