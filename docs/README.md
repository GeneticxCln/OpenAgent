# OpenAgent Documentation

Welcome to the OpenAgent documentation! This guide will help you understand, use, and contribute to OpenAgent - a powerful, production-ready AI agent framework.

## 📚 Documentation Overview

### Getting Started
- [Installation Guide](installation.md) - How to install and set up OpenAgent
- [Quick Start Guide](quickstart.md) - Get up and running in minutes
- [Configuration Guide](configuration.md) - Configure OpenAgent for your needs

### User Guides
- [CLI Usage](quickstart.md#cli-usage) - Using the command-line interface
- [API Reference](api.md) - REST API endpoints and usage
- [WebSocket Guide](quickstart_server_ws.md) - Real-time communication setup
- [Model Selection](../README.md#available-models) - Choosing the right model

### Architecture & Design
- [System Architecture](ARCHITECTURE.md) - High-level system overview
- [System Design](SYSTEM_DESIGN.md) - Detailed technical design
- [Architectural Decisions](adr/README.md) - ADRs documenting key decisions

### Development
- [Developer Guide](DEVELOPER_GUIDE.md) - Comprehensive development guide
- [Contributing Guidelines](../CONTRIBUTING.md) - How to contribute to OpenAgent
- [API Stability](API_STABILITY.md) - API compatibility and versioning

### Operations
- [Deployment Guide](DEPLOYMENT.md) - Production deployment strategies
- [Security Guide](../SECURITY.md) - Security best practices and features
- [Troubleshooting](DEPLOYMENT.md#troubleshooting) - Common issues and solutions

## 🏗️ Architecture Overview

OpenAgent implements a modular, security-first architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OpenAgent System                             │
├─────────────────────────────────────────────────────────────────────┤
│  CLI Interface    │  Web API       │  WebSocket      │  Terminal     │
│  (typer/rich)     │  (FastAPI)     │  (realtime)     │  Integration  │
├─────────────────┬─────────────────┬─────────────────┬─────────────────┤
│                 │                 │                 │                 │
│                 ▼                 ▼                 ▼                 │
│           ┌─────────────────────────────────────────────────────────┐ │
│           │              Agent Orchestration Layer               │ │
│           │  • Agent Management   • Tool Selection              │ │
│           │  • Context Management • Workflow Engine             │ │
│           └─────────────────────────────────────────────────────────┘ │
│                                   │                                   │
│                                   ▼                                   │
│           ┌─────────────────────────────────────────────────────────┐ │
│           │                  Policy Engine                         │ │
│           │  • Command Validation   • Risk Assessment             │ │
│           │  • Security Policies    • Audit Logging              │ │
│           └─────────────────────────────────────────────────────────┘ │
│                                   │                                   │
│                                   ▼                                   │
│           ┌─────────────────────────────────────────────────────────┐ │
│           │               LLM Abstraction Layer                    │ │
│           │  • HuggingFace Local  • Ollama Local                  │ │
│           │  • Model Management   • Response Streaming            │ │
│           └─────────────────────────────────────────────────────────┘ │
│                                   │                                   │
│                                   ▼                                   │
│           ┌─────────────────────────────────────────────────────────┐ │
│           │                  Tool Execution Layer                  │ │
│           │  • System Commands   • File Operations                 │ │
│           │  • Git Operations    • Custom Tools                   │ │
│           └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Core Features

### Agent-Tool Pattern
- **Flexible Architecture**: Agents orchestrate multiple tools to complete tasks
- **Extensible Design**: Easy to add new tools and agents
- **Composable**: Mix and match tools for different use cases

### Multi-LLM Support
- **HuggingFace Integration**: Local models with full privacy
- **Ollama Support**: Optimized local model serving
- **Unified Interface**: Consistent API across providers

### Security-First Design
- **Policy Engine**: Centralized security validation
- **Risk Assessment**: Intelligent command risk evaluation
- **Audit Logging**: Comprehensive audit trail

### Production Ready
- **High Performance**: Async-first with efficient resource management
- **Monitoring**: Built-in metrics and observability
- **Scalable**: Horizontal scaling support

## 🛠️ Development Workflow

### Setting Up Development Environment
```bash
# Clone and setup
git clone https://github.com/GeneticxCln/OpenAgent.git
cd OpenAgent

# Full development setup
make dev

# This runs:
# - make venv (creates virtual environment)
# - make editable (installs in development mode)
# - make shell-integration (configures shell)
# - make policy (enables security policy)
# - make path (adds to PATH)
```

### Testing
```bash
# Run all tests
make test
pytest

# Run with coverage
pytest --cov=openagent --cov-report=html

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Skip slow tests
```

### Code Quality
```bash
# Format code
make fmt
black openagent tests
isort openagent tests

# Lint code
make lint
flake8 openagent tests

# Type checking
make type
mypy openagent
```

## 📖 Key Concepts

### Agents
Agents are the core orchestrators in OpenAgent. They combine LLM capabilities with tool usage to complete tasks.

```python
from openagent import Agent
from openagent.tools.system import CommandExecutor, FileManager

# Create an agent with specific tools
agent = Agent(
    name="SystemAdmin",
    model_name="codellama-7b",
    tools=[CommandExecutor(), FileManager()]
)

# Use the agent
response = await agent.process_message("Show me system resource usage")
```

### Tools
Tools encapsulate specific functionality that agents can use.

```python
from openagent.core.base import BaseTool, ToolResult

class CustomTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="Does something useful"
        )
    
    async def execute(self, input_data):
        # Your tool logic here
        return ToolResult(
            success=True,
            content="Tool executed successfully"
        )
```

### Policy Engine
The policy engine provides security validation for all operations.

```python
from openagent.core.policy import PolicyEngine

# Create policy engine with custom policy file
policy_engine = PolicyEngine(policy_file="my-policy.yaml")

# Validate commands
decision = await policy_engine.validate_command("rm -rf /tmp/test")
if decision.action == PolicyAction.ALLOW:
    # Execute command
    pass
```

## 🔧 Configuration

### Environment Variables
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

## 📊 Performance & Monitoring

### Built-in Metrics
OpenAgent includes Prometheus-compatible metrics:

- Request rate and latency
- Model inference time
- Memory and CPU usage
- Policy decisions
- Error rates

### Health Monitoring
```bash
# Check system health
curl http://localhost:8000/health

# Get metrics
curl http://localhost:8000/metrics
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details on:

- Development setup
- Coding standards
- Testing requirements
- Pull request process
- Code review guidelines

### Quick Start for Contributors
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit pull request

## 📄 License

OpenAgent is released under the MIT License. See [LICENSE](../LICENSE) for details.

## 🆘 Support

- **Documentation**: You're reading it! 📖
- **Issues**: [GitHub Issues](https://github.com/GeneticxCln/OpenAgent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GeneticxCln/OpenAgent/discussions)
- **Security**: See [SECURITY.md](../SECURITY.md) for security policy

## 🗺️ Roadmap

See our [ROADMAP.md](../ROADMAP.md) for planned features and improvements.

---

**Built with ❤️ for developers who want AI assistance without compromise on privacy and control.**
