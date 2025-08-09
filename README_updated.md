# 🚀 OpenAgent - AI-Powered Terminal Assistant

[![CI/CD Pipeline](https://github.com/GeneticxCln/OpenAgent/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/GeneticxCln/OpenAgent/actions)
[![codecov](https://codecov.io/gh/GeneticxCln/OpenAgent/branch/main/graph/badge.svg)](https://codecov.io/gh/GeneticxCln/OpenAgent)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://hub.docker.com/r/openagent/openagent)

OpenAgent is a sophisticated AI-powered terminal assistant that combines the power of large language models with a comprehensive suite of system tools. It provides an intuitive conversational interface for complex tasks, file operations, system management, and code generation - all while maintaining security and extensibility at its core.

## ✨ Features

### 🧠 Core AI Capabilities
- **Multi-Model Support**: Choose from lightweight to powerful LLMs
- **Contextual Understanding**: Maintains conversation history and context
- **Smart Tool Selection**: Automatically selects appropriate tools for tasks
- **Code Generation**: Generate, analyze, and review code in multiple languages
- **Natural Language Interface**: Conversational interaction with complex systems

### 🛠️ Powerful Tool Ecosystem
- **System Tools**: Process management, system monitoring, diagnostics
- **File Operations**: Read, write, search, and manipulate files safely
- **Command Execution**: Secure shell command execution with safeguards
- **Web Integration**: HTTP requests, API interactions, web scraping
- **Development Tools**: Git operations, package management, testing

### 🌐 Web API & Server
- **FastAPI Backend**: Production-ready REST API with automatic documentation
- **Authentication**: JWT-based secure authentication system
- **Rate Limiting**: Configurable rate limiting with multiple strategies
- **Real-time Monitoring**: Health checks, metrics, and observability
- **Docker Support**: Containerized deployment with orchestration

### 🔒 Security & Safety
- **Sandboxed Execution**: Safe command execution with configurable restrictions
- **Permission System**: Fine-grained access control for sensitive operations
- **Audit Logging**: Comprehensive logging of all actions and decisions
- **Input Validation**: Robust validation and sanitization of all inputs

### 🧪 Quality Assurance
- **Comprehensive Testing**: Unit, integration, and end-to-end tests
- **Code Quality**: Automated linting, formatting, and type checking
- **Continuous Integration**: Automated testing and deployment pipelines
- **Documentation**: Extensive documentation with examples

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Docker and Docker Compose (optional, for containerized deployment)
- Git

### Installation

#### Option 1: Development Setup
```bash
# Clone the repository
git clone https://github.com/GeneticxCln/OpenAgent.git
cd OpenAgent

# Set up development environment
./scripts/dev.sh setup

# Activate virtual environment
source venv/bin/activate

# Start development server
./scripts/dev.sh start
```

#### Option 2: Docker Deployment
```bash
# Clone and start with Docker
git clone https://github.com/GeneticxCln/OpenAgent.git
cd OpenAgent

# Start production services
docker-compose up -d

# Or start development environment
docker-compose --profile dev up -d
```

#### Option 3: PyPI Installation
```bash
# Install from PyPI (when available)
pip install openagent

# Start the server
openagent serve
```

### Quick Test
```bash
# Test the CLI
openagent chat "What's the current system status?"

# Test the API
curl http://localhost:8000/health

# Open API documentation
open http://localhost:8000/docs
```

## 💻 Usage Examples

### Command Line Interface
```bash
# Basic conversation
openagent chat "Help me organize my project files"

# Code generation
openagent generate-code "Create a Python REST API with FastAPI" --language python

# System analysis
openagent system-info --detailed

# File operations
openagent file-ops "Search for TODO comments in my Python files"
```

### Web API
```python
import httpx

# Chat with the agent
response = httpx.post("http://localhost:8000/chat", json={
    "message": "Help me debug this Python script",
    "context": {"file_path": "script.py"}
})

# Generate code
response = httpx.post("http://localhost:8000/generate-code", json={
    "description": "Create a data validation function",
    "language": "python",
    "include_tests": True
})

# Analyze code
response = httpx.post("http://localhost:8000/analyze-code", json={
    "code": "def hello(): print('world')",
    "language": "python",
    "focus": ["security", "performance"]
})
```

### Python Library
```python
from openagent import Agent, SystemTool, FileTool

# Create an agent
agent = Agent(
    name="my-assistant",
    model_name="tiny-llama",
    tools=[SystemTool(), FileTool()]
)

# Use the agent
response = await agent.process_message(
    "What Python files are in the current directory?"
)
print(response)
```

## 📡 API Reference

### Authentication
```bash
# Login (if auth enabled)
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Use token in subsequent requests
curl -H "Authorization: Bearer <token>" http://localhost:8000/agents
```

### Core Endpoints
- `POST /chat` - Chat with an agent
- `POST /generate-code` - Generate code
- `POST /analyze-code` - Analyze code
- `GET /system-info` - Get system information
- `GET /agents` - List agents
- `POST /agents` - Create new agent
- `GET /models` - List available models
- `GET /health` - Health check

### WebSocket Support (Coming Soon)
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.send(JSON.stringify({
    type: 'chat',
    message: 'Hello, agent!',
    stream: true
}));
```

## 🔧 Configuration

### Environment Variables
```bash
# Authentication
AUTH_ENABLED=true
JWT_SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Rate Limiting
RATE_LIMIT_ENABLED=true
REQUESTS_PER_MINUTE=60
REQUESTS_PER_HOUR=1000

# Models
DEFAULT_MODEL=tiny-llama
MODEL_CACHE_DIR=/app/models

# Security
SAFE_MODE=true
ALLOWED_COMMANDS=["ls", "cat", "grep"]
RESTRICTED_PATHS=["/etc", "/sys"]

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
ENABLE_ACCESS_LOGS=true
```

### Configuration File
```yaml
# config.yaml
agent:
  name: "OpenAgent"
  description: "AI-powered assistant"
  model: "tiny-llama"
  
tools:
  system:
    enabled: true
    safe_mode: true
    timeout: 30
  
  file:
    enabled: true
    max_file_size: "10MB"
    allowed_extensions: [".txt", ".py", ".js"]
  
  command:
    enabled: true
    whitelist: ["git", "npm", "python"]
    
security:
  sandbox_mode: true
  audit_logs: true
  max_execution_time: 60
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        OpenAgent                            │
├─────────────────────────────────────────────────────────────┤
│  Web API (FastAPI)          │  CLI Interface               │
│  ├── Authentication         │  ├── Chat Commands          │
│  ├── Rate Limiting          │  ├── File Operations        │
│  ├── Request Validation     │  └── System Tools           │
│  └── Response Formatting    │                              │
├─────────────────────────────────────────────────────────────┤
│                     Core Agent Engine                       │
│  ├── Message Processing     │  ├── Context Management     │
│  ├── Tool Orchestration     │  ├── Memory System          │
│  ├── Response Generation    │  └── Error Handling         │
├─────────────────────────────────────────────────────────────┤
│                      Tool Ecosystem                         │
│  ├── System Tools           │  ├── File Tools             │
│  ├── Command Tools          │  ├── Web Tools              │
│  ├── Development Tools      │  └── Custom Tools           │
├─────────────────────────────────────────────────────────────┤
│                     LLM Integration                         │
│  ├── Model Registry         │  ├── Prompt Templates       │
│  ├── Response Processing    │  ├── Context Injection      │
│  ├── Error Recovery         │  └── Fallback Mechanisms    │
└─────────────────────────────────────────────────────────────┘
```

## 🧪 Development

### Setting Up Development Environment
```bash
# Setup everything
./scripts/dev.sh setup

# Start development services
./scripts/dev.sh start --monitoring

# Run tests
./scripts/dev.sh test

# Run linting
./scripts/dev.sh lint

# Build Docker images
./scripts/dev.sh build
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/unit/ -v           # Unit tests
pytest tests/integration/ -v    # Integration tests
pytest tests/api/ -v            # API tests

# Run with coverage
pytest tests/ --cov=openagent --cov-report=html
```

### Code Quality
```bash
# Format code
black openagent/ tests/
isort openagent/ tests/

# Lint code
flake8 openagent/ tests/
pylint openagent/

# Type checking
mypy openagent/ --ignore-missing-imports

# Security scanning
bandit -r openagent/
safety check
```

## 🚢 Deployment

### Docker Deployment
```bash
# Production deployment
docker-compose --profile production up -d

# With monitoring
docker-compose --profile production --profile monitoring up -d

# Scale services
docker-compose up --scale openagent-api=3
```

### Kubernetes Deployment
```bash
# Apply manifests
kubectl apply -f k8s/

# Check status
kubectl get pods -l app=openagent

# View logs
kubectl logs -l app=openagent -f
```

### Cloud Deployment

#### AWS ECS
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin
docker build -t openagent .
docker tag openagent:latest $ECR_URI:latest
docker push $ECR_URI:latest

# Deploy with ECS
aws ecs update-service --cluster openagent --service openagent-api --force-new-deployment
```

#### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/openagent
gcloud run deploy openagent --image gcr.io/PROJECT-ID/openagent --platform managed
```

## 🛡️ Security Considerations

### Best Practices
1. **Authentication**: Always enable authentication in production
2. **Rate Limiting**: Configure appropriate rate limits
3. **Input Validation**: All inputs are validated and sanitized
4. **Command Restrictions**: Use whitelisting for allowed commands
5. **File System Access**: Restrict file operations to safe directories
6. **Logging**: Enable comprehensive audit logging
7. **Updates**: Keep dependencies and models up to date

### Security Configuration
```yaml
security:
  # Enable sandbox mode for safe execution
  sandbox_mode: true
  
  # Restrict file system access
  allowed_directories:
    - "/home/user/projects"
    - "/tmp"
  
  forbidden_directories:
    - "/etc"
    - "/sys"
    - "/proc"
  
  # Command whitelist
  allowed_commands:
    - "git"
    - "npm"
    - "python"
    - "ls"
    - "cat"
  
  # Maximum execution time
  max_execution_time: 30
  
  # Enable audit logging
  audit_logs: true
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Contribution Setup
```bash
# Fork and clone your fork
git clone https://github.com/YOUR-USERNAME/OpenAgent.git
cd OpenAgent

# Set up development environment
./scripts/dev.sh setup

# Create a feature branch
git checkout -b feature/amazing-feature

# Make changes and run tests
./scripts/dev.sh test
./scripts/dev.sh lint

# Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature

# Create a pull request
```

### Development Guidelines
- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation
- Use conventional commit messages
- Ensure all CI checks pass

## 📚 Documentation

- [API Documentation](https://openagent.github.io/api/)
- [User Guide](https://openagent.github.io/guide/)
- [Developer Guide](https://openagent.github.io/dev/)
- [Tool Development](https://openagent.github.io/tools/)
- [Deployment Guide](https://openagent.github.io/deploy/)

## 📊 Monitoring & Observability

### Metrics & Monitoring
```bash
# Start with monitoring stack
./scripts/dev.sh start --monitoring

# Access monitoring dashboards
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus
```

### Available Metrics
- Request rate and latency
- Agent response times
- Tool execution statistics
- Error rates and types
- System resource usage
- Model performance metrics

### Logging
```bash
# View logs
./scripts/dev.sh logs

# Structured logging with ELK stack
./scripts/dev.sh start --logging
open http://localhost:5601  # Kibana
```

## 🗺️ Roadmap

### Phase 1: Foundation ✅
- [x] Core agent architecture
- [x] Basic tool system
- [x] CLI interface
- [x] Comprehensive testing
- [x] API server with authentication
- [x] Rate limiting and security

### Phase 2: Advanced Features 🚧
- [ ] Plugin system for custom tools
- [ ] WebSocket real-time communication
- [ ] Multi-agent orchestration
- [ ] Advanced memory management
- [ ] Integration marketplace

### Phase 3: Enterprise Features 🔮
- [ ] RBAC (Role-Based Access Control)
- [ ] Advanced monitoring and analytics
- [ ] Custom model fine-tuning
- [ ] Enterprise SSO integration
- [ ] Compliance and governance tools

### Phase 4: AI Enhancement 🌟
- [ ] Multi-modal capabilities (vision, speech)
- [ ] Advanced reasoning and planning
- [ ] Self-improvement mechanisms
- [ ] Federated learning support
- [ ] Custom model hosting

## 🐛 Troubleshooting

### Common Issues

#### Installation Issues
```bash
# Python version conflicts
pyenv install 3.11.0
pyenv local 3.11.0

# Permission errors
sudo chown -R $(whoami) /path/to/openagent
chmod +x scripts/dev.sh
```

#### Model Loading Issues
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/transformers/

# Check disk space
df -h

# Update transformers
pip install --upgrade transformers torch
```

#### API Connection Issues
```bash
# Check service status
./scripts/dev.sh logs openagent-api

# Test connectivity
curl http://localhost:8000/health

# Check authentication
export TOKEN=$(curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | jq -r .access_token)
```

### Getting Help
- 📖 Check the [documentation](https://openagent.github.io/)
- 🐛 Search [existing issues](https://github.com/GeneticxCln/OpenAgent/issues)
- 💬 Join our [Discord community](https://discord.gg/openagent)
- 📧 Contact support: support@openagent.dev

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for model infrastructure
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [LangChain](https://langchain.com/) for AI application patterns
- [Rich](https://rich.readthedocs.io/) for beautiful terminal interfaces
- The open-source community for continuous inspiration

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeneticxCln/OpenAgent&type=Date)](https://star-history.com/#GeneticxCln/OpenAgent&Date)

---

Made with ❤️ by the OpenAgent team. 

**Ready to transform your terminal experience? [Get Started Now!](https://github.com/GeneticxCln/OpenAgent/blob/main/README.md#quick-start)**
