# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

OpenAgent is a powerful, production-ready AI agent framework powered by Hugging Face models, designed for terminal assistance, code generation, and system operations. It's designed as an open-source alternative to Warp AI that can run entirely locally with customizable AI models.

## Common Development Commands

### Development Environment Setup
```bash
# Full development setup (creates venv, installs dependencies, configures shell integration)
make dev

# Individual setup steps
make venv                    # Create virtual environment
make editable               # Install in development mode with pip install -e .
make shell-integration      # Apply zsh integration for shell hooks
make policy                 # Enable security policy with block_risky=True
make path                   # Add venv/bin to PATH in ~/.zshrc
```

### Testing
```bash
# Run all tests
make test
pytest

# Run tests with coverage
pytest --cov=openagent --cov-report=html

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Skip slow tests
pytest -m llm               # Tests requiring LLM functionality

# Run single test file
pytest tests/test_integration.py

# Run specific test
pytest tests/test_integration.py::TestHistoryManagement::test_create_history_block
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

# Security scanning
bandit -r openagent

# Install pre-commit hooks
./scripts/install_precommit.sh
pre-commit install
pre-commit run --all-files
```

### Running the Application
```bash
# Interactive chat with default model
openagent chat

# Use specific model
openagent chat --model codellama-7b
openagent chat --model ollama:llama3.2

# Single commands
openagent run "How do I list Python files recursively?"
openagent code "Create a function to calculate fibonacci"
openagent explain "find . -name '*.py' -exec grep -l 'import torch' {} +"

# Start API server
uvicorn openagent.server.app:app --host 0.0.0.0 --port 8000

# Export OpenAPI schema
make openapi
./scripts/export_openapi.sh 8042 openapi.json
```

### Building and Distribution
```bash
# Build standalone binary
./scripts/build_binary.sh --python venv/bin/python --name openagent --output dist

# Build with different parameters
python -m PyInstaller --onefile --name openagent openagent/cli.py
```

## Architecture Overview

### Core Components

**Agent System (`openagent/core/`)**
- `agent.py`: Main Agent implementation with LLM integration and tool orchestration
- `base.py`: Abstract base classes for agents and tools (BaseAgent, BaseTool, BaseMessage, ToolResult)
- `llm.py`: Hugging Face model integration with support for CodeLlama, Mistral, TinyLlama, etc.
- `llm_ollama.py`: Ollama local model integration for privacy-first usage
- `policy.py`: Security policy engine for command validation and risk assessment
- `observability.py`: Metrics collection, logging, and request tracking
- `workflows.py`: Workflow management for complex multi-step tasks

**Tools System (`openagent/tools/`)**
- `system.py`: CommandExecutor, FileManager, SystemInfo tools for terminal operations
- `git.py`: GitTool and RepoGrep for version control operations
- All tools implement the `BaseTool` interface with async `execute()` method

**Server Components (`openagent/server/`)**
- `app.py`: FastAPI web server with REST API, WebSocket support, and streaming
- `auth.py`: Authentication manager with token-based auth
- `rate_limiter.py`: Request rate limiting and quota management
- `models.py`: Pydantic models for API request/response schemas

**Terminal Integration (`openagent/terminal/`)**
- `integration.py`: Shell integration for command interception and assistance
- `validator.py`: Command validation and security policy enforcement

### Key Architecture Patterns

**Agent-Tool Pattern**: Agents orchestrate multiple tools to complete tasks. Tools are composable and can be combined in different ways.

**Policy-Driven Security**: All command execution goes through a policy engine that can ALLOW, DENY, EXPLAIN_ONLY, or REQUIRE_APPROVAL based on risk assessment.

**Multi-LLM Support**: Supports both Hugging Face transformers (local) and Ollama (local) with unified interface through `get_llm()` factory.

**Async-First Design**: All core operations are async to support streaming responses and concurrent tool execution.

**Observability Integration**: Built-in metrics, logging, and request tracking for production monitoring.

**Fallback Mechanisms**: Fast fallback responses for common queries to reduce latency before hitting the LLM.

### Model Selection Strategy

The system supports multiple model categories:
- **Code Models**: `codellama-7b`, `deepseek-coder`, `starcoder` for programming tasks
- **Chat Models**: `mistral-7b`, `zephyr-7b`, `openchat` for general conversation
- **Lightweight Models**: `tiny-llama`, `phi-2` for resource-constrained environments
- **Ollama Models**: Any local Ollama model via `ollama:model-name` prefix

### Configuration and Security

New environment flags:
- OPENAGENT_EXEC_STRICT: When set to 1/true, shell fallback in CommandExecutor is disabled. Only structured exec-style commands (safe subset) will run.
- OPENAGENT_SMART_SELECTOR: Enable/disable SmartToolSelector planning (1=on, 0=off). Default: on.

File operations policy:
- FileManager enforces safe_paths and restricted_paths from the PolicyEngine. Mutating ops (write/copy/move/delete) require:
  - Path within safe_paths (and not in restricted_paths)
  - Approval for medium/high risk operations unless confirm=true is provided
- Violations return clear errors and are logged to the audit trail.

**Environment Variables**:
- `HUGGINGFACE_TOKEN` or `HF_TOKEN`: For accessing private HF models
- `DEFAULT_MODEL`: Override default model selection
- `OPENAGENT_EXPLAIN`: Enable explain-only mode by default
- `OPENAGENT_WARN`: Enable command warnings

**Security Features**:
- Command filtering and risk assessment
- Safe mode vs. execution mode toggles
- Execution timeouts (30 seconds default)
- Input validation and sanitization
- Secret redaction in logs and responses

**Configuration Files**:
- `.env`: Environment variables and API tokens
- `~/.config/openagent/keys.env`: Global API keys
- Policy files for command validation rules

## Development Guidelines

### Testing Strategy
- Unit tests for individual components (`tests/test_*.py`)
- Integration tests for system-level functionality (`tests/test_integration.py`)
- CLI tests for command-line interface (`tests/test_cli_*.py`)
- Mock LLM usage in tests to avoid network dependencies
- Async test support with `pytest-asyncio`

### Adding New Tools
1. Inherit from `BaseTool` in `openagent.core.base`
2. Implement async `execute()` method returning `ToolResult`
3. Add tool validation in `validate_input()`
4. Register tool with agents in appropriate modules
5. Add tests for tool functionality

### Adding New Models
1. Add model configuration to `ModelConfig` class in `llm.py`
2. Test model compatibility with different quantization options
3. Update documentation and examples
4. Consider memory requirements and device compatibility

### API Development
- Follow FastAPI patterns with proper Pydantic models
- Add authentication and rate limiting as needed
- Support both REST and WebSocket interfaces
- Include proper error handling and logging

## Production Deployment

### Server Configuration
- Use `uvicorn` with multiple workers for production
- Configure CORS, trusted hosts, and security headers
- Set up proper logging and metrics collection
- Use environment variables for sensitive configuration

### Model Deployment
- Pre-cache models with `OPENAGENT_PRECACHE=1`
- Use 4-bit quantization for memory efficiency
- Consider CPU-only deployment for sensitive environments
- Monitor GPU memory usage and implement fallbacks

### Monitoring and Observability
- Built-in Prometheus metrics for requests, agent messages, policy decisions
- Structured logging with request tracing
- Command audit trails for security compliance
- Health checks and model status endpoints

## Important Files and Locations

- `openagent/cli.py`: Main CLI entry point
- `openagent/core/agent.py`: Core agent implementation
- `pyproject.toml`: Dependencies, build configuration, and development tools
- `Makefile`: Development workflow automation
- `requirements-dev.txt`: Development dependencies
- `scripts/`: Build and utility scripts
- `tests/`: Test suite with multiple test categories
- `.pre-commit-config.yaml`: Code quality automation
