# Installation Guide

## Basic Installation

For basic functionality without ML features:

```bash
pip install openagent
```

## Optional Dependencies

OpenAgent offers several optional dependency groups that can be installed based on your needs:

### Machine Learning Features (ml)

For LLM integration, text processing, and ML capabilities:

```bash
pip install openagent[ml]
```

This includes:
- PyYAML for configuration files
- Jinja2 for templating
- Transformers and tokenizers
- PyTorch
- Hugging Face Hub
- Datasets and accelerate
- NumPy

### Server Components (server)

For running OpenAgent as a web service:

```bash
pip install openagent[server]
```

This includes:
- FastAPI
- Uvicorn with standard extensions
- Python multipart support

### Database Support (db)

For database connectivity and job queues:

```bash
pip install openagent[db]
```

This includes:
- SQLAlchemy
- Alembic for migrations
- Redis
- Celery

### Development Tools (dev)

For contributing to OpenAgent:

```bash
pip install openagent[dev]
```

This includes testing, linting, and code quality tools.

### Documentation (docs)

For building documentation:

```bash
pip install openagent[docs]
```

### LLM Provider Integrations

For specific LLM providers:

```bash
# OpenAI integration
pip install openagent[openai]

# Anthropic integration
pip install openagent[anthropic]
```

### Complete Installation

To install all optional dependencies:

```bash
pip install openagent[all]
```

## Usage Examples

### Basic Usage (Core Dependencies Only)

```python
from openagent.core.agent import Agent

# Create a basic agent without LLM capabilities
agent = Agent(name="basic_agent")
```

### With ML Features

```python
# First install: pip install openagent[ml]
from openagent.core.agent import Agent
from openagent.core.llm import get_llm

# Create an agent with LLM capabilities
llm = get_llm("huggingface", model_name="gpt2")
agent = Agent(name="ml_agent", llm=llm)
```

### With Server Components

```python
# First install: pip install openagent[server,ml]
from openagent.server import create_app

app = create_app()
# Run with: uvicorn main:app
```

## Troubleshooting

If you encounter import errors for optional dependencies, make sure you've installed the appropriate extras:

- For ML features: `pip install openagent[ml]`
- For server features: `pip install openagent[server]` 
- For full functionality: `pip install openagent[all]`
