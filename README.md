# OpenAgent

A powerful, production-ready AI agent framework powered by Hugging Face models, designed for terminal assistance, code generation, and system operations - just like Warp AI but open source and customizable.

**✨ Recently restructured for production readiness with clean architecture, 37% fewer files, and enhanced maintainability while preserving all features.**

## 🚀 Features

- **🤖 Hugging Face Integration** - Use any HF model (CodeLlama, Mistral, Llama2, etc.)
- **💻 Terminal Assistant** - Execute commands safely, explain operations, manage files
- **🔧 Code Generation** - Generate, analyze, and review code in multiple languages
- **⚡ High Performance** - Optimized with 4-bit quantization, CUDA support
- **🛠️ Advanced Tools** - System monitoring, file management, command execution
- **🎨 Rich CLI Interface** - Beautiful terminal interface with syntax highlighting
- **🔒 Security First** - Safe command execution with built-in security checks
- **📱 Multiple Interfaces** - CLI chat and single commands; API server and WebSocket streaming (alpha). See docs/API_STABILITY.md for endpoint contracts and docs/quickstart_server_ws.md for a quickstart.

## 🏁 Quick Start

### 1. Basic Installation

```bash
# Install core OpenAgent (lightweight)
pip install openagent
```

### 2. Install ML Features (for LLM integration)

```bash
# Install OpenAgent with ML capabilities
pip install openagent[ml]

# Or for full installation with all features
pip install openagent[all]
```

### 3. Development Setup

```bash
git clone https://github.com/yourusername/OpenAgent.git
cd OpenAgent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[all]"
```

### 4. Start Chatting

```bash
# Interactive chat with lightweight model
openagent chat --model tiny-llama

# Use a code-focused model for programming tasks
openagent chat --model codellama-7b

# CPU-only mode
openagent chat --model tiny-llama --device cpu
```

### 5. Run API Server (alpha)

The HTTP and WebSocket server are available for experimentation.

```bash
# Start FastAPI server
uvicorn openagent.server.app:app --host 0.0.0.0 --port 8000

# WebSocket endpoint: ws://localhost:8000/ws
# HTTP docs (OpenAPI): http://localhost:8000/docs
```

#### Streaming
- WebSocket: ws(s)://HOST/ws/chat (default path). Send a JSON payload and receive incremental JSON messages like {"content":"...","event":"chunk"} with a final {"event":"end"}.
- SSE: POST /chat/stream (server-sent events)

```bash
curl -N -H "Accept: text/event-stream" -H "Content-Type: application/json" \
  -d '{"message":"explain binary search in python","agent":"default"}' \
  http://localhost:8000/chat/stream
```

### CLI Streaming (SSE/WS) and Auth

The chat CLI can stream responses from the API server via SSE or WebSocket, and supports authenticated servers.

Examples:

- Use API server with SSE streaming (default):
  - OPENAGENT_API_URL=http://localhost:8000 openagent chat
  - or: openagent chat --api-url http://localhost:8000

- Force WebSocket streaming with optional custom path:
  - openagent chat --api-url http://localhost:8000 --ws
  - openagent chat --api-url https://api.example.com --ws --ws-path /ws/chat

- Provide an API token for Authorization headers (Bearer by default):
  - openagent chat --api-url https://api.example.com --api-token $TOKEN
  - OPENAGENT_API_TOKEN=$TOKEN openagent chat --api-url https://api.example.com

- Customize the auth scheme or pass raw token (no prefix):
  - openagent chat --api-url https://api.example.com --api-token $TOKEN --auth-scheme ""

- Include the token as a query parameter on the WS URL (if your server expects it):
  - openagent chat --api-url https://api.example.com --ws --api-token $TOKEN --ws-token-query-key token

- Disable streaming (use full-response HTTP):
  - openagent chat --api-url https://api.example.com --no-stream

Notes:
- If --ws is set and WebSocket connection fails (or the websockets package is not installed), the CLI automatically falls back to SSE; if SSE fails, it falls back to non-streaming.
- For WS, the URL is derived from the API base: http(s) -> ws(s), host preserved, path from --ws-path (default /ws/chat). Authorization header and/or token query param can be sent.
- For SSE and non-streaming HTTP, the Authorization header is included when an API token is provided.

Example clients:
- Python: examples/ws_client.py
- JavaScript (Node 18+): examples/ws_client.js

For security guidance and RBAC considerations, see SECURITY.md.

## 🧭 Terminal UI: Blocks, Folding, and Shortcuts

OpenAgent’s terminal UI renders commands and outputs as visual blocks with syntax highlighting, folding, and navigation.

Keyboard shortcuts (interactive renderer):
- j/k: navigate blocks down/up
- o or Enter: fold/unfold selected block
- l: toggle block list view (Enter returns to normal view)
- s / r: save session / reload session (.openagent_session.json in CWD)
- /: search blocks; n / p: next / previous match
- e: export blocks to openagent_history.md
- t: add a tag to the selected block
- g: go to block by ID or number

Tip: In the default chat prompt, type the shortcut key and press Enter (e.g., type `o` then Enter to fold/unfold the selected block).

Programmatic hooks (for non-interactive usage):
- save_session(path=None) → Path: persist current blocks/selection to JSON
- load_session(path=None) → int: restore a session; returns number of blocks
- search(query: str) → List[int]: record search results and return matching indices
- export(path=None, format="markdown") → Path: export blocks to a file (markdown or json)

Example:
```python path=null start=null
from openagent.ui.renderer import create_terminal_renderer

renderer = create_terminal_renderer()
# ... create and update blocks ...
session_path = renderer.save_session()  # ./.openagent_session.json
results = renderer.search("error")     # list of matching indices
md_path = renderer.export()             # ./openagent_history.md
```

Formatter improvements:
- Markdown: headers, lists, tables, and fenced code blocks rendered with a monokai theme.
- Diffs: colored with diff lexer (or fallback), including headers/index/hunks.
- Logs: timestamp and level highlighting (INFO/WARN/ERROR/etc.).
- Folding heuristics: better grouping for stack traces, test output, git, and logs on large outputs.

### Session persistence (save, load, export)

You can persist your session (blocks and selection) to resume later.

- At the prompt: press `s` then Enter to save; `r` to reload; `e` to export to markdown.
- Programmatically:

```python path=null start=null
from openagent.ui.renderer import create_terminal_renderer
r = create_terminal_renderer()
# ... create blocks during your run ...
r.save_session()                 # saves to ./.openagent_session.json
r.load_session()                 # reloads from that file
r.export()                       # writes ./openagent_history.md
r.export("history.json","json")  # export as JSON
```

## 🎯 Usage Examples

### Interactive Chat
```bash
# Start interactive session
openagent chat

# Inside the chat:
You: How do I list all Python files recursively?
Assistant: You can use the `find` command to recursively list all Python files...

You: /system  # Show system information
You: /help    # Show available commands
You: /quit    # Exit
```

### Single Commands
```bash
# Generate code
openagent code "Create a Python function to calculate fibonacci numbers" --language python

# Analyze existing code
openagent analyze myfile.py

# Explain shell commands
openagent explain "find . -name '*.py' -exec grep -l 'import torch' {} +"

# Quick AI assistance
openagent run "How do I optimize this Docker build process?"
```

### Available Models

**Code Models (Recommended for programming):**
- `tiny-llama` - Fast, lightweight (1.1B params)
- `codellama-7b` - Code-focused (7B params)  
- `codellama-13b` - More capable (13B params)
- `deepseek-coder` - Excellent for coding (6.7B params)
- `starcoder` - GitHub Copilot alternative

**Chat Models:**
- `mistral-7b` - General purpose
- `zephyr-7b` - Fine-tuned for helpfulness
- `openchat` - Conversation optimized

**Lightweight Models:**
- `phi-2` - Microsoft's efficient model (2.7B params)
- `stable-code` - Stability AI's code model (3B params)

## 📦 Build a standalone binary

You can package OpenAgent into a single-file executable for easy distribution (no Python required for end users).

Linux/macOS/Windows (via PyInstaller)
- Create and use a virtual environment
- Install dependencies (per README Quick Start), then run:

```bash
# Linux/macOS
chmod +x scripts/build_binary.sh
./scripts/build_binary.sh --python venv/bin/python --name openagent --output dist
# Windows (PowerShell)
# py -m pip install pyinstaller
# py -m PyInstaller --onefile --name openagent openagent/cli.py
```

The output binary will be in dist/. On Linux/macOS, rename it or distribute as-is (e.g., openagent or openagent.bin). On Windows, it will be openagent.exe.

Notes
- First run on a new machine will download your chosen model from Hugging Face (unless cached).
- For smaller footprint, ship without GPU torch and instruct end-users to use CPU mode or a tiny model.
- For Linux-only distribution, consider an AppImage wrapper; we can add this on request.

## 🛠️ Advanced Features

### System Operations
```python
from openagent import Agent
from openagent.tools.system import CommandExecutor, FileManager, SystemInfo

# Create agent with tools
agent = Agent(
    name="SystemAdmin",
    model_name="codellama-7b",
    tools=[CommandExecutor(), FileManager(), SystemInfo()]
)

# Use the agent
response = await agent.process_message("Show me system resource usage")
```

### Custom Tools
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

# Add to agent
agent.add_tool(CustomTool())
```

### Configuration
```bash
# Copy example config
cp .env.example .env

# Edit configuration
nano .env
```

Key configuration options:
- `HUGGINGFACE_TOKEN` - For private models
- `DEFAULT_MODEL` - Default model to use
- `DEFAULT_DEVICE` - cuda/cpu/auto
- `LOAD_IN_4BIT` - Enable 4-bit quantization
- `MAX_MEMORY_USAGE` - Memory limit

## 🏗️ Architecture

```
OpenAgent/
├── openagent/
│   ├── core/           # Core framework
│   │   ├── llm.py     # Hugging Face integration
│   │   ├── agent.py   # Main agent implementation
│   │   └── base.py    # Abstract base classes
│   ├── tools/         # Built-in tools
│   │   └── system.py  # Terminal/system tools
│   └── cli.py         # Command line interface
├── tests/             # Test suite (planned)
├── docs/              # Documentation (planned)
└── examples/          # Usage examples (planned)
```

# 🔧 Development

### Pre-commit hooks
To keep code quality consistent, install and enable pre-commit hooks:

```bash
pip install pre-commit
./scripts/install_precommit.sh
# or manually:
# pre-commit install
# pre-commit run --all-files
```

Note: This project remains CLI-first, but an API server and WebSocket interface are implemented in alpha. Expect breaking changes while server APIs stabilize.

### Setup Development Environment
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black openagent/
isort openagent/

# Type checking
mypy openagent/

# Security scan
bandit -r openagent/
```

### Creating Custom Agents
```python
from openagent.core.agent import Agent
from openagent.core.llm import HuggingFaceLLM

# Custom LLM configuration
llm_config = {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_length": 4096,
    "load_in_8bit": True,
}

# Create specialized agent
code_assistant = Agent(
    name="CodeAssistant",
    description="Expert Python developer and code reviewer",
    model_name="codellama-13b", 
    llm_config=llm_config
)
```

## 🚦 Performance Tips

### Memory Optimization
- Use `--load-in-4bit` for memory efficiency
- Start with `tiny-llama` for testing
- Monitor memory usage with `/system` command

### GPU Acceleration
- Install CUDA-enabled PyTorch
- Use `--device cuda` for GPU acceleration
- Check GPU memory with `nvidia-smi`

### Model Selection
- **Development/Testing**: `tiny-llama`, `phi-2`
- **Code Tasks**: `codellama-7b`, `deepseek-coder`
- **General Chat**: `mistral-7b`, `zephyr-7b`
- **Production**: `codellama-13b` (if you have enough memory)

## 🔐 Security

OpenAgent includes built-in security features:

- **Command Filtering** - Dangerous commands are blocked
- **Execution Timeouts** - Commands timeout after 30 seconds  
- **Safe Mode** - Option to explain commands without executing
- **Input Validation** - All inputs are validated before processing

See also: docs/HARDENING.md for production hardening tips (strict execution, sandboxing, safe paths, RBAC).

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Contribution Setup
```bash
# Fork the repo, then:
git clone https://github.com/yourusername/OpenAgent.git
cd OpenAgent
python setup.py dev
python setup.py test
```

## 🗺️ Roadmap

- Short term
  - Safer default command handling (explain-only by default) with opt-in execution
  - Environment-based configuration and Hugging Face token support
  - Better error handling and model fallback to CPU on OOM
- Medium term
  - Minimal API server (FastAPI) with auth/rate limiting
  - Tests, docs, and examples directory
  - Stronger sandboxing for command execution

## 📊 Comparison with Other Tools

| Feature | OpenAgent | Warp AI | GitHub Copilot | Cursor |
|---------|-----------|---------|----------------|--------|
| **Open Source** | ✅ | ❌ | ❌ | ❌ |
| **Local Models** | ✅ | ❌ | ❌ | Limited |
| **Terminal Integration** | ✅ | ✅ | ❌ | ❌ |
| **Code Generation** | ✅ | ✅ | ✅ | ✅ |
| **System Operations** | ✅ | ✅ | ❌ | ❌ |
| **Customizable** | ✅ | Limited | ❌ | Limited |
| **Privacy** | ✅ | ❌ | ❌ | ❌ |

## 📋 System Requirements

- **Python**: 3.9+
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for models
- **GPU**: Optional (CUDA-compatible for acceleration)
- **OS**: Linux, macOS, Windows (WSL recommended)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** - For the amazing model ecosystem
- **Meta AI** - For CodeLlama and Llama2 models
- **The Open Source Community** - For inspiration and feedback

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/OpenAgent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/OpenAgent/discussions)

---

**Built with ❤️ for developers who want AI assistance without compromise on privacy and control.**
