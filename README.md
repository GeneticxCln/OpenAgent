# OpenAgent

A powerful, production-ready AI agent framework powered by Hugging Face models, designed for terminal assistance, code generation, and system operations - just like Warp AI but open source and customizable.

## 🚀 Features

- **🤖 Hugging Face Integration** - Use any HF model (CodeLlama, Mistral, Llama2, etc.)
- **💻 Terminal Assistant** - Execute commands safely, explain operations, manage files
- **🔧 Code Generation** - Generate, analyze, and review code in multiple languages
- **⚡ High Performance** - Optimized with 4-bit quantization, CUDA support
- **🛠️ Advanced Tools** - System monitoring, file management, command execution
- **🎨 Rich CLI Interface** - Beautiful terminal interface with syntax highlighting
- **🔒 Security First** - Safe command execution with built-in security checks
- **📱 Multiple Interfaces** - CLI chat, single commands, API server

## 🏁 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/OpenAgent.git
cd OpenAgent

# Create virtual environment (already created)
# python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .
```

### 2. Install PyTorch (if not already installed)

```bash
# For CUDA (if you have a GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Start Chatting

```bash
# Interactive chat with lightweight model
openagent chat --model tiny-llama

# Use a code-focused model for programming tasks
openagent chat --model codellama-7b

# CPU-only mode
openagent chat --model tiny-llama --device cpu
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
├── tests/             # Test suite
├── docs/              # Documentation
└── examples/          # Usage examples
```

## 🔧 Development

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
