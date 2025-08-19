# OpenAgent Tutorial: From Zero to AI Agent

Welcome to the comprehensive OpenAgent tutorial! This guide will take you from installation to building your own custom AI agents with OpenAgent.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding Core Concepts](#understanding-core-concepts)
3. [Building Your First Agent](#building-your-first-agent)
4. [Working with Tools](#working-with-tools)
5. [Security and Policies](#security-and-policies)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

```bash
# Basic installation
pip install openagent

# Full installation with all features
pip install openagent[all]

# Development installation
git clone https://github.com/GeneticxCln/OpenAgent.git
cd OpenAgent
pip install -e ".[dev]"
```

### Your First Chat

```bash
# Start interactive chat with a lightweight model
openagent chat --model tiny-llama

# Use CPU-only mode for compatibility
openagent chat --model tiny-llama --device cpu
```

## Understanding Core Concepts

### 1. Agents

Agents are the core orchestrators in OpenAgent. They combine LLM capabilities with tool usage to complete tasks.

```python
from openagent.core.agent import Agent
from openagent.tools.system import SystemInfo

# Create a simple agent
agent = Agent(
    name="MyAssistant",
    model_name="tiny-llama",
    tools=[SystemInfo()]
)

# Use the agent
response = await agent.process_message("What's my system information?")
print(response.content)
```

### 2. Tools

Tools encapsulate specific functionality that agents can use. They follow a consistent interface:

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
            content="Tool executed successfully",
            metadata={"execution_time": 0.1}
        )
```

### 3. Models

OpenAgent supports multiple LLM providers:

- **HuggingFace**: Local models with full privacy
- **Ollama**: Optimized local model serving
- **Future**: OpenAI, Anthropic (configurable)

```python
# Different model configurations
agent_hf = Agent(name="HF", model_name="codellama-7b")
agent_ollama = Agent(name="Ollama", model_name="ollama:llama3.2")
```

## Building Your First Agent

Let's build a practical agent step by step.

### Step 1: Define Your Agent's Purpose

```python
"""
Goal: Create a developer assistant that can:
- Execute shell commands safely
- Manage files
- Provide system information
- Help with Git operations
"""
```

### Step 2: Choose and Configure Tools

```python
from openagent.tools.system import CommandExecutor, FileManager, SystemInfo
from openagent.tools.git import GitTool

# Configure tools with appropriate safety settings
tools = [
    CommandExecutor(default_explain_only=True),  # Safe mode by default
    FileManager(safe_paths=["/home/user/projects", "/tmp"]),
    SystemInfo(),
    GitTool()
]
```

### Step 3: Create the Agent

```python
from openagent.core.agent import Agent
from openagent.core.config import Config

# Create configuration
config = Config({
    "model": "codellama-7b",
    "device": "cpu",
    "load_in_4bit": True,  # Memory optimization
    "max_memory": "4G"
})

# Create the agent
dev_assistant = Agent(
    name="DevAssistant",
    description="A helpful development assistant",
    model_name="codellama-7b",
    tools=tools,
    config=config
)
```

### Step 4: Use Your Agent

```python
async def main():
    # Example interactions
    responses = await asyncio.gather(
        dev_assistant.process_message("What files are in my current directory?"),
        dev_assistant.process_message("Show me the git status"),
        dev_assistant.process_message("What's my system's CPU info?")
    )
    
    for response in responses:
        print(f"🤖 {response.content}\n")

# Run the agent
asyncio.run(main())
```

## Working with Tools

### Built-in Tools

#### CommandExecutor
Safely execute shell commands with policy validation.

```python
from openagent.tools.system import CommandExecutor

# Safe mode (explains commands without executing)
safe_executor = CommandExecutor(default_explain_only=True)

# Production mode (executes commands after policy check)
prod_executor = CommandExecutor(default_explain_only=False)

# Usage in agent
agent = Agent(
    name="ShellAssistant",
    model_name="codellama-7b",
    tools=[safe_executor]
)

# Examples
await agent.process_message("List all Python files in this directory")
await agent.process_message("Check if Docker is running")
```

#### FileManager
Manage files with path restrictions for security.

```python
from openagent.tools.system import FileManager

# Configure safe paths
file_manager = FileManager(
    safe_paths=[
        "/home/user/documents",
        "/home/user/projects",
        "/tmp"
    ],
    restricted_paths=[
        "/etc",
        "/root",
        "/sys"
    ]
)

# Usage examples
await agent.process_message("Create a new Python script called hello.py")
await agent.process_message("Read the contents of README.md")
await agent.process_message("List files in my Documents folder")
```

#### SystemInfo
Get system information and monitor resources.

```python
from openagent.tools.system import SystemInfo

system_tool = SystemInfo()

# Usage examples
await agent.process_message("What's my system's memory usage?")
await agent.process_message("Show me CPU information")
await agent.process_message("What operating system am I running?")
```

#### GitTool
Perform Git operations on repositories.

```python
from openagent.tools.git import GitTool

git_tool = GitTool()

# Usage examples
await agent.process_message("What's the git status of this repository?")
await agent.process_message("Show me the latest 5 commits")
await agent.process_message("What branch am I on?")
```

### Creating Custom Tools

#### Simple Custom Tool

```python
from openagent.core.base import BaseTool, ToolResult
import requests

class WeatherTool(BaseTool):
    def __init__(self, api_key: str):
        super().__init__(
            name="weather",
            description="Get current weather information for a location"
        )
        self.api_key = api_key
    
    async def execute(self, input_data):
        location = input_data.get("location")
        if not location:
            return ToolResult(
                success=False,
                content="Location is required",
                error="Missing location parameter"
            )
        
        try:
            # Call weather API (example)
            response = requests.get(
                f"http://api.weather.com/current",
                params={"location": location, "key": self.api_key}
            )
            
            if response.status_code == 200:
                weather_data = response.json()
                return ToolResult(
                    success=True,
                    content=f"Weather in {location}: {weather_data['description']}",
                    metadata=weather_data
                )
            else:
                return ToolResult(
                    success=False,
                    content="Failed to fetch weather data",
                    error=f"API returned {response.status_code}"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                content="Error fetching weather",
                error=str(e)
            )
```

#### Advanced Custom Tool with Validation

```python
from openagent.core.base import BaseTool, ToolResult
from pydantic import BaseModel, validator
from typing import Optional

class DatabaseQuery(BaseModel):
    """Input model for database queries."""
    query: str
    database: str = "default"
    limit: Optional[int] = 100
    
    @validator('query')
    def validate_query(cls, v):
        # Basic SQL injection protection
        dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE']
        if any(keyword in v.upper() for keyword in dangerous_keywords):
            raise ValueError("Only SELECT queries are allowed")
        return v

class DatabaseTool(BaseTool):
    def __init__(self, connection_string: str):
        super().__init__(
            name="database",
            description="Execute safe database queries"
        )
        self.connection_string = connection_string
    
    def validate_input(self, input_data):
        try:
            DatabaseQuery(**input_data)
            return True
        except Exception as e:
            return False
    
    async def execute(self, input_data):
        if not self.validate_input(input_data):
            return ToolResult(
                success=False,
                content="Invalid query parameters",
                error="Validation failed"
            )
        
        query_params = DatabaseQuery(**input_data)
        
        # Execute query safely
        try:
            # Your database execution logic here
            results = await self._execute_query(query_params.query)
            
            return ToolResult(
                success=True,
                content=f"Query returned {len(results)} rows",
                metadata={
                    "results": results[:query_params.limit],
                    "total_rows": len(results)
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content="Database query failed",
                error=str(e)
            )
    
    async def _execute_query(self, query: str):
        # Implement your database query logic
        pass
```

## Security and Policies

OpenAgent includes a comprehensive security system to safely execute commands and operations.

### Policy Engine

The policy engine validates all operations before execution:

```python
from openagent.core.policy import PolicyEngine, PolicyAction

# Create policy engine with custom rules
policy_engine = PolicyEngine(policy_file="my-policy.yaml")

# Manual policy validation
decision = await policy_engine.validate_command("rm -rf /tmp/test")

if decision.action == PolicyAction.ALLOW:
    # Safe to execute
    pass
elif decision.action == PolicyAction.DENY:
    print(f"Command blocked: {decision.reason}")
elif decision.action == PolicyAction.REQUIRE_APPROVAL:
    # Ask user for confirmation
    user_approval = input(f"Allow '{command}'? (y/N): ")
    if user_approval.lower() == 'y':
        # Execute with approval
        pass
```

### Policy Configuration

Create a `policy.yaml` file to define security rules:

```yaml
# Security policy configuration
safe_paths:
  - /home/user/projects
  - /tmp
  - /home/user/documents

restricted_paths:
  - /etc
  - /root
  - /usr/bin
  - /sys
  - /proc

command_policies:
  # Dangerous commands
  - pattern: "rm -rf .*"
    action: DENY
    reason: "Recursive delete operations are not allowed"
  
  - pattern: "sudo .*"
    action: REQUIRE_APPROVAL
    reason: "Privileged operations require approval"
  
  # Safe commands
  - pattern: "ls .*"
    action: ALLOW
    reason: "Directory listing is safe"
  
  - pattern: "git .*"
    action: ALLOW
    reason: "Git operations are generally safe"

# Network policies
network_policies:
  allowed_domains:
    - github.com
    - api.github.com
    - pypi.org
  
  blocked_ips:
    - 169.254.0.0/16  # Link-local addresses

# Resource limits
resource_limits:
  max_memory_mb: 1024
  max_cpu_percent: 50
  max_execution_time: 30
  max_file_size_mb: 100
```

### Safe Mode

Use safe mode for development and testing:

```python
# Enable safe mode globally
import os
os.environ["OPENAGENT_EXPLAIN_ONLY"] = "true"

# Or configure per tool
safe_executor = CommandExecutor(default_explain_only=True)

# Toggle safe mode at runtime
agent = Agent(name="SafeAgent", model_name="tiny-llama", tools=[safe_executor])
await agent.process_message("rm important_file.txt")  # Will explain, not execute
```

## Advanced Features

### Conversation History

OpenAgent automatically manages conversation history:

```python
from openagent.core.history import HistoryManager

history = HistoryManager()

# Create history block
block = HistoryManager.new_block(
    input_text="Hello, world!",
    response="Hello! How can I help you?",
    metadata={"model": "tiny-llama"}
)

# Save to history
history.append(block)

# Retrieve history
recent_blocks = history.list_blocks(limit=10)
search_results = history.search("hello", limit=5)

# Export history
markdown_export = history.export(block.id, format="md")
json_export = history.export(block.id, format="json")
```

### Context Management

Manage conversation context across interactions:

```python
# Set context for the agent
agent.update_context({
    "user_id": "user123",
    "session_id": "session456",
    "preferences": {
        "language": "en",
        "expertise_level": "intermediate"
    }
})

# Context is automatically included in LLM prompts
response = await agent.process_message("Help me with Python")

# Clear context when needed
agent.clear_context()
```

### Streaming Responses

For real-time applications, use streaming:

```python
async for chunk in agent.stream_message("Generate a long explanation"):
    print(chunk, end="", flush=True)
```

### Workflow Management

Create complex multi-step workflows:

```python
from openagent.core.workflows import Workflow, WorkflowStep

# Define workflow steps
steps = [
    WorkflowStep(
        name="analyze_code",
        tool="file_manager",
        parameters={"action": "read", "path": "main.py"}
    ),
    WorkflowStep(
        name="run_tests",
        tool="command_executor",
        parameters={"command": "python -m pytest"}
    ),
    WorkflowStep(
        name="generate_report",
        tool="file_manager",
        parameters={"action": "write", "path": "report.md"}
    )
]

# Create and execute workflow
workflow = Workflow(name="code_analysis", steps=steps)
result = await agent.execute_workflow(workflow)
```

## Best Practices

### 1. Agent Design

**Start Simple**
```python
# Good: Start with basic functionality
agent = Agent(
    name="SimpleAssistant",
    model_name="tiny-llama",
    tools=[SystemInfo()]
)

# Avoid: Over-engineering from the start
```

**Choose Appropriate Models**
```python
# For development/testing
dev_agent = Agent(name="Dev", model_name="tiny-llama")

# For code tasks
code_agent = Agent(name="Coder", model_name="codellama-7b")

# For production
prod_agent = Agent(name="Prod", model_name="codellama-13b")
```

### 2. Security Best Practices

**Always Use Safe Mode Initially**
```python
# Good: Start with safety
executor = CommandExecutor(default_explain_only=True)

# Test thoroughly before enabling execution
executor = CommandExecutor(default_explain_only=False)
```

**Implement Proper Path Restrictions**
```python
# Good: Restrict file access
file_manager = FileManager(
    safe_paths=["/home/user/projects"],
    restricted_paths=["/etc", "/root"]
)

# Avoid: Unrestricted access
file_manager = FileManager()  # No restrictions
```

**Use Policy Files**
```python
# Good: Define clear policies
policy_engine = PolicyEngine(policy_file="production-policy.yaml")

# Avoid: No policy validation
```

### 3. Error Handling

**Graceful Error Handling**
```python
async def safe_agent_call(agent, message):
    try:
        response = await agent.process_message(message)
        return response.content
    except AgentError as e:
        logger.error(f"Agent error: {e}")
        return "I apologize, but I encountered an error processing your request."
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "An unexpected error occurred. Please try again."
```

**Tool Error Handling**
```python
class RobustTool(BaseTool):
    async def execute(self, input_data):
        try:
            # Tool logic here
            result = self._do_work(input_data)
            return ToolResult(success=True, content=result)
        except SpecificError as e:
            logger.warning(f"Tool warning: {e}")
            return ToolResult(
                success=False,
                content="Operation partially failed",
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Tool error: {e}")
            return ToolResult(
                success=False,
                content="Tool execution failed",
                error="Internal error"
            )
```

### 4. Performance Optimization

**Model Selection**
```python
# Memory-constrained environments
light_agent = Agent(
    name="Light",
    model_name="tiny-llama",
    config=Config({"load_in_4bit": True, "max_memory": "2G"})
)

# High-performance requirements
power_agent = Agent(
    name="Power",
    model_name="codellama-13b",
    config=Config({"device": "cuda", "load_in_4bit": False})
)
```

**Async Best Practices**
```python
# Good: Process multiple requests concurrently
responses = await asyncio.gather(
    agent.process_message("Task 1"),
    agent.process_message("Task 2"),
    agent.process_message("Task 3")
)

# Avoid: Sequential processing when not necessary
responses = []
for message in messages:
    response = await agent.process_message(message)
    responses.append(response)
```

## Troubleshooting

### Common Issues

#### Model Loading Errors

```python
# Problem: Model not found
try:
    agent = Agent(name="Test", model_name="nonexistent-model")
except ModelLoadError as e:
    print(f"Model error: {e}")
    # Solution: Use available model
    agent = Agent(name="Test", model_name="tiny-llama")
```

#### Memory Issues

```python
# Problem: Out of memory
# Solution: Use quantization and CPU mode
config = Config({
    "device": "cpu",
    "load_in_4bit": True,
    "max_memory": "2G"
})
agent = Agent(name="Efficient", model_name="codellama-7b", config=config)
```

#### Tool Execution Errors

```python
# Problem: Tool fails to execute
try:
    result = await tool.execute({"invalid": "params"})
except ToolExecutionError as e:
    print(f"Tool error: {e}")
    # Solution: Validate input parameters
    if tool.validate_input(input_data):
        result = await tool.execute(input_data)
```

### Debugging Tips

**Enable Debug Logging**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
import os
os.environ["OPENAGENT_LOG_LEVEL"] = "DEBUG"
```

**Use Test Mode**
```python
# Set test environment
os.environ["OPENAGENT_ENV"] = "test"
os.environ["OPENAGENT_EXPLAIN_ONLY"] = "true"
```

**Monitor Resource Usage**
```python
import psutil
import os

def monitor_resources():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent()
    print(f"Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%")

# Call periodically during development
monitor_resources()
```

## Next Steps

Congratulations! You now have a solid understanding of OpenAgent. Here are suggested next steps:

1. **Build Your First Custom Agent**: Use the patterns from this tutorial
2. **Create Custom Tools**: Implement tools specific to your use case
3. **Explore the Examples**: Check out the examples directory for more ideas
4. **Read the API Documentation**: Dive deeper into specific components
5. **Join the Community**: Contribute to the project or ask questions

### Additional Resources

- [API Reference](api.md) - Detailed API documentation
- [Architecture Guide](ARCHITECTURE.md) - System architecture overview
- [Developer Guide](DEVELOPER_GUIDE.md) - For contributors
- [Examples Directory](../examples/) - Working examples
- [GitHub Issues](https://github.com/GeneticxCln/OpenAgent/issues) - Report bugs or request features

Happy building with OpenAgent! 🚀
