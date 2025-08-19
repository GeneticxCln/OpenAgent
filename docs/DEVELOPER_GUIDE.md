# OpenAgent Developer Guide

## Overview

This guide provides comprehensive information for developers working on the OpenAgent project, including development setup, coding standards, architectural decisions, and contribution workflows.

## Table of Contents

- [Development Environment](#development-environment)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Coding Standards](#coding-standards)
- [Testing Strategy](#testing-strategy)
- [Plugin Development](#plugin-development)
- [Performance Guidelines](#performance-guidelines)
- [Security Best Practices](#security-best-practices)
- [Debugging and Troubleshooting](#debugging-and-troubleshooting)

## Development Environment

### Initial Setup

```bash
# 1. Clone the repository
git clone https://github.com/GeneticxCln/OpenAgent.git
cd OpenAgent

# 2. Set up development environment using Make
make dev

# This runs:
# - make venv (creates virtual environment)
# - make editable (installs in development mode)
# - make shell-integration (configures zsh integration)
# - make policy (enables security policy)
# - make path (adds to PATH)
```

### Manual Setup (Alternative)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install development dependencies
pip install -e ".[dev,ml,server]"

# 3. Install pre-commit hooks
pre-commit install

# 4. Verify installation
openagent --help
pytest --version
```

### Development Tools

#### Required Tools
- **Python 3.9+**: Primary language
- **Git**: Version control
- **Make**: Build automation
- **pytest**: Testing framework

#### Recommended Tools
- **VS Code** with Python extension
- **PyCharm** Professional/Community
- **Docker** for containerized development
- **Redis** for caching (optional)
- **PostgreSQL** for persistence (optional)

#### VS Code Configuration

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".coverage": true,
        "htmlcov": true
    }
}
```

### Environment Variables

```bash
# Development environment (.env.dev)
OPENAGENT_ENV=development
OPENAGENT_DEBUG=true
OPENAGENT_LOG_LEVEL=DEBUG

# Model settings for development
OPENAGENT_MODEL=tiny-llama
OPENAGENT_DEVICE=cpu
OPENAGENT_LOAD_IN_4BIT=false

# Development database
OPENAGENT_DATABASE_URL=sqlite:///dev.db
OPENAGENT_REDIS_URL=redis://localhost:6379/1

# Security (relaxed for development)
OPENAGENT_EXEC_STRICT=false
OPENAGENT_POLICY_FILE=config/dev-policy.yaml

# Testing
PYTEST_CURRENT_TEST=true
OPENAGENT_TEST_MODE=true
```

## Architecture Deep Dive

### Core Components

#### 1. Agent System (`openagent/core/`)

**Agent Class Design**
```python
class Agent:
    """Main agent orchestrating LLM and tools."""
    
    def __init__(
        self,
        name: str,
        model_name: str,
        tools: Optional[List[BaseTool]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        policy_engine: Optional[PolicyEngine] = None
    ):
        self.name = name
        self.model_name = model_name
        self.tools = tools or []
        self.llm = self._initialize_llm(model_name, llm_config)
        self.policy_engine = policy_engine or PolicyEngine()
        self.context_manager = ContextManager()
        self.metrics = get_metrics_collector()
    
    async def process_message(
        self, 
        message: str, 
        context: Optional[ConversationContext] = None
    ) -> AgentResponse:
        """Process a user message and return response."""
        start_time = time.time()
        
        try:
            # 1. Validate input
            validated_message = await self._validate_input(message)
            
            # 2. Update context
            if context:
                self.context_manager.update_context(context)
            
            # 3. Tool selection
            selected_tools = await self._select_tools(validated_message)
            
            # 4. Generate response
            response = await self.llm.generate_response(
                prompt=self._build_prompt(validated_message, selected_tools),
                tools=selected_tools
            )
            
            # 5. Execute tools if needed
            if response.tool_calls:
                tool_results = await self._execute_tools(response.tool_calls)
                response = await self._incorporate_tool_results(response, tool_results)
            
            # 6. Record metrics
            self.metrics.record_agent_message(
                agent_id=self.name,
                message_type="success",
                duration=time.time() - start_time
            )
            
            return response
            
        except Exception as e:
            self.metrics.record_agent_message(
                agent_id=self.name,
                message_type="error",
                duration=time.time() - start_time
            )
            raise
```

**Tool Selection Strategy**
```python
class SmartToolSelector:
    """Intelligent tool selection based on context and intent."""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.tool_embeddings = {}
        self.usage_patterns = defaultdict(int)
    
    async def select_tools(
        self, 
        message: str, 
        available_tools: List[BaseTool],
        context: Optional[ConversationContext] = None
    ) -> List[BaseTool]:
        """Select most relevant tools for the given message."""
        
        # 1. Extract intent from message
        intent = await self._extract_intent(message)
        
        # 2. Score tools based on relevance
        tool_scores = {}
        for tool in available_tools:
            score = await self._score_tool_relevance(tool, intent, message)
            if context:
                score *= self._apply_context_boost(tool, context)
            tool_scores[tool] = score
        
        # 3. Select top tools (max 3 to avoid overwhelming)
        selected_tools = sorted(
            tool_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        return [tool for tool, score in selected_tools if score > 0.5]
    
    async def _extract_intent(self, message: str) -> Dict[str, float]:
        """Extract user intent using LLM classification."""
        prompt = f"""
        Classify the user intent in this message: "{message}"
        
        Possible intents:
        - code_generation: Writing new code
        - code_analysis: Analyzing existing code
        - system_operation: System commands or information
        - file_management: File operations
        - git_operation: Version control operations
        - question_answering: General questions
        
        Return JSON with intent probabilities.
        """
        
        response = await self.llm.generate_response(prompt)
        return json.loads(response)
```

#### 2. LLM Abstraction Layer

**Provider Factory Pattern**
```python
def get_llm(model_name: str, **kwargs) -> BaseLLM:
    """Factory function to create appropriate LLM instance."""
    
    if model_name.startswith("ollama:"):
        return OllamaLLM(model_name.replace("ollama:", ""), **kwargs)
    elif model_name in HUGGINGFACE_MODELS:
        return HuggingFaceLLM(model_name, **kwargs)
    elif model_name.startswith("openai:"):
        return OpenAILLM(model_name.replace("openai:", ""), **kwargs)
    else:
        # Default to HuggingFace
        return HuggingFaceLLM(model_name, **kwargs)

class HuggingFaceLLM(BaseLLM):
    """HuggingFace transformer implementation."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        load_in_4bit: bool = True,
        torch_dtype: str = "auto",
        max_memory: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.torch_dtype = torch_dtype
        self.max_memory = max_memory
        
        self._model = None
        self._tokenizer = None
        self._pipeline = None
    
    async def generate_response(
        self, 
        prompt: str, 
        max_length: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate response using HuggingFace model."""
        
        if not self._pipeline:
            await self._load_model()
        
        # Prepare generation parameters
        generation_kwargs = {
            "max_length": max_length,
            "temperature": temperature,
            "do_sample": True,
            "pad_token_id": self._tokenizer.eos_token_id,
            **kwargs
        }
        
        # Generate response
        with torch.no_grad():
            response = self._pipeline(
                prompt,
                **generation_kwargs
            )[0]["generated_text"]
        
        # Clean up response (remove prompt)
        return response[len(prompt):].strip()
    
    async def stream_response(
        self, 
        prompt: str, 
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream response generation."""
        
        if not self._pipeline:
            await self._load_model()
        
        # Use text generation pipeline with streaming
        streamer = TextStreamer(self._tokenizer, skip_prompt=True)
        
        generation_kwargs = {
            "streamer": streamer,
            "return_full_text": False,
            **kwargs
        }
        
        # This is a simplified streaming implementation
        # In practice, you'd need more sophisticated streaming
        response = await self.generate_response(prompt, **generation_kwargs)
        
        # Simulate streaming by yielding chunks
        words = response.split()
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield f" {word}"
            await asyncio.sleep(0.01)  # Small delay for streaming effect
```

#### 3. Policy Engine Design

**Security Policy Architecture**
```python
class PolicyEngine:
    """Centralized security policy enforcement."""
    
    def __init__(self, policy_file: Optional[str] = None):
        self.policies = self._load_policies(policy_file)
        self.audit_logger = AuditLogger()
        self.risk_assessor = RiskAssessor()
    
    async def validate_command(
        self, 
        command: str, 
        context: Optional[ExecutionContext] = None
    ) -> PolicyDecision:
        """Validate command against security policies."""
        
        # 1. Check command patterns
        for policy in self.policies.command_policies:
            if re.match(policy.pattern, command):
                decision = PolicyDecision(
                    action=policy.action,
                    reason=policy.reason,
                    confidence=1.0
                )
                await self._log_policy_decision(command, decision, context)
                return decision
        
        # 2. Risk assessment
        risk_level = await self.risk_assessor.assess_command(command)
        
        if risk_level >= RiskLevel.HIGH:
            decision = PolicyDecision(
                action=PolicyAction.REQUIRE_APPROVAL,
                reason=f"High risk operation detected: {risk_level.name}",
                confidence=0.8
            )
        elif risk_level >= RiskLevel.MEDIUM:
            decision = PolicyDecision(
                action=PolicyAction.ALLOW,
                reason=f"Medium risk operation: {risk_level.name}",
                confidence=0.6
            )
        else:
            decision = PolicyDecision(
                action=PolicyAction.ALLOW,
                reason="Low risk operation",
                confidence=0.9
            )
        
        await self._log_policy_decision(command, decision, context)
        return decision

class RiskAssessor:
    """Assess risk level of operations."""
    
    RISK_PATTERNS = {
        RiskLevel.CRITICAL: [
            r"rm\s+-rf\s+/",
            r"dd\s+if=.*of=/dev/",
            r"format\s+c:",
            r"mkfs\.",
        ],
        RiskLevel.HIGH: [
            r"sudo\s+",
            r"chmod\s+777",
            r"chown\s+root",
            r"iptables\s+",
            r"systemctl\s+(stop|disable)",
        ],
        RiskLevel.MEDIUM: [
            r"rm\s+.*\.(py|js|ts|go|rs)$",
            r"git\s+reset\s+--hard",
            r"docker\s+run\s+.*--privileged",
            r"pip\s+install.*--force",
        ]
    }
    
    async def assess_command(self, command: str) -> RiskLevel:
        """Assess risk level of a command."""
        
        command_lower = command.lower().strip()
        
        # Check against known patterns
        for risk_level, patterns in self.RISK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, command_lower):
                    return risk_level
        
        # Additional heuristics
        risk_score = 0
        
        # File system operations
        if any(word in command_lower for word in ["rm", "delete", "del"]):
            risk_score += 20
        
        # Network operations
        if any(word in command_lower for word in ["curl", "wget", "nc", "netcat"]):
            risk_score += 10
        
        # Privileged operations
        if any(word in command_lower for word in ["sudo", "su", "admin"]):
            risk_score += 30
        
        # Determine risk level based on score
        if risk_score >= 50:
            return RiskLevel.HIGH
        elif risk_score >= 25:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
```

### Plugin Architecture

#### Plugin Interface
```python
class BasePlugin(ABC):
    """Abstract base class for all plugins."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.name = config.get("name", self.__class__.__name__)
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the plugin."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup plugin resources."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        pass

class PluginManager:
    """Manage plugin lifecycle and registration."""
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_registry = PluginRegistry()
    
    async def load_plugin(self, plugin_path: str) -> None:
        """Load a plugin from file path."""
        
        # 1. Validate plugin
        if not await self._validate_plugin(plugin_path):
            raise PluginValidationError(f"Invalid plugin: {plugin_path}")
        
        # 2. Load plugin module
        spec = importlib.util.spec_from_file_location("plugin", plugin_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 3. Find plugin class
        plugin_class = None
        for item in dir(module):
            obj = getattr(module, item)
            if (isinstance(obj, type) and 
                issubclass(obj, BasePlugin) and 
                obj != BasePlugin):
                plugin_class = obj
                break
        
        if not plugin_class:
            raise PluginLoadError("No plugin class found")
        
        # 4. Initialize plugin
        plugin_config = await self._load_plugin_config(plugin_path)
        plugin = plugin_class(plugin_config)
        await plugin.initialize()
        
        # 5. Register plugin
        self.plugins[plugin.name] = plugin
        await self.plugin_registry.register(plugin)
```

## Coding Standards

### Python Style Guide

#### Type Hints
```python
# Always use type hints for function signatures
async def process_message(
    message: str,
    context: Optional[ConversationContext] = None,
    timeout: float = 30.0
) -> AgentResponse:
    """Process a user message."""
    pass

# Use proper generics
from typing import List, Dict, Optional, Union, TypeVar, Generic

T = TypeVar('T')

class Repository(Generic[T]):
    def find_by_id(self, id: str) -> Optional[T]:
        pass

# Use Protocol for structural typing
from typing import Protocol

class Executable(Protocol):
    async def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        ...
```

#### Error Handling
```python
# Custom exceptions with proper inheritance
class OpenAgentError(Exception):
    """Base exception for OpenAgent."""
    pass

class ModelLoadError(OpenAgentError):
    """Error loading AI model."""
    pass

class PolicyViolationError(OpenAgentError):
    """Security policy violation."""
    
    def __init__(self, command: str, policy: str, reason: str):
        self.command = command
        self.policy = policy
        self.reason = reason
        super().__init__(f"Policy violation: {reason}")

# Proper error handling with context
async def load_model(model_name: str) -> BaseLLM:
    """Load AI model with proper error handling."""
    try:
        logger.info(f"Loading model: {model_name}")
        
        if model_name.startswith("ollama:"):
            llm = OllamaLLM(model_name.replace("ollama:", ""))
        else:
            llm = HuggingFaceLLM(model_name)
        
        await llm.initialize()
        logger.info(f"Successfully loaded model: {model_name}")
        return llm
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"GPU memory error loading {model_name}: {e}")
        raise ModelLoadError(f"Insufficient GPU memory for {model_name}")
    
    except Exception as e:
        logger.error(f"Unexpected error loading {model_name}: {e}")
        raise ModelLoadError(f"Failed to load model {model_name}: {e}")
```

#### Logging
```python
import logging
import structlog

# Configure structured logging
logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage examples
async def process_command(command: str) -> CommandResult:
    """Process command with structured logging."""
    
    logger.info(
        "Processing command",
        command=command,
        user_id=get_current_user_id(),
        timestamp=datetime.utcnow().isoformat()
    )
    
    try:
        result = await execute_command(command)
        
        logger.info(
            "Command executed successfully",
            command=command,
            result_type=type(result).__name__,
            execution_time=result.execution_time
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "Command execution failed",
            command=command,
            error=str(e),
            error_type=type(e).__name__,
            traceback=traceback.format_exc()
        )
        raise
```

### Documentation Standards

#### Docstring Format
```python
def complex_function(
    param1: str,
    param2: Optional[int] = None,
    param3: List[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    One-line summary of the function.
    
    Longer description explaining the purpose and behavior of the function.
    This can span multiple lines and should provide enough context for
    developers to understand how to use the function.
    
    Args:
        param1: Description of the first parameter.
        param2: Optional parameter with default value. Defaults to None.
        param3: List of strings for processing. Defaults to empty list.
    
    Returns:
        Tuple containing:
            - bool: Success status of the operation
            - Dict[str, Any]: Result data with operation details
    
    Raises:
        ValueError: If param1 is empty or invalid format.
        ProcessingError: If internal processing fails.
    
    Example:
        >>> success, data = complex_function("test", param2=42)
        >>> print(f"Success: {success}, Data: {data}")
        Success: True, Data: {'processed': True, 'count': 1}
    
    Note:
        This function is CPU intensive for large inputs. Consider using
        the async version for better performance.
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    
    # Implementation here
    return True, {"processed": True, "count": 1}
```

## Testing Strategy

### Test Structure
```
tests/
├── unit/                    # Unit tests
│   ├── core/
│   │   ├── test_agent.py
│   │   ├── test_llm.py
│   │   └── test_policy.py
│   ├── tools/
│   │   ├── test_system.py
│   │   └── test_git.py
│   └── server/
│       ├── test_app.py
│       └── test_auth.py
├── integration/             # Integration tests
│   ├── test_agent_workflow.py
│   ├── test_api_endpoints.py
│   └── test_websocket.py
├── fixtures/               # Test data and fixtures
│   ├── models/
│   ├── policies/
│   └── sample_data/
├── conftest.py             # Pytest configuration
└── test_utils.py           # Test utilities
```

### Testing Patterns

#### Unit Tests
```python
import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, AsyncMock
from openagent.core.agent import Agent
from openagent.core.base import ToolResult

class TestAgent:
    """Test suite for Agent class."""
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        llm = AsyncMock()
        llm.generate_response = AsyncMock(return_value="Test response")
        llm.stream_response = AsyncMock()
        return llm
    
    @pytest.fixture
    def mock_tool(self):
        """Mock tool for testing."""
        tool = Mock()
        tool.name = "test_tool"
        tool.description = "Test tool"
        tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            content="Tool executed",
            metadata={}
        ))
        return tool
    
    @pytest.fixture
    def agent(self, mock_llm, mock_tool):
        """Create agent instance for testing."""
        with patch('openagent.core.agent.get_llm', return_value=mock_llm):
            agent = Agent(
                name="TestAgent",
                model_name="test-model",
                tools=[mock_tool]
            )
            return agent
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.name == "TestAgent"
        assert agent.model_name == "test-model"
        assert len(agent.tools) == 1
    
    @pytest.mark.asyncio
    async def test_process_message_success(self, agent, mock_llm):
        """Test successful message processing."""
        mock_llm.generate_response.return_value = "Success response"
        
        response = await agent.process_message("Hello, world!")
        
        assert response.content == "Success response"
        assert response.success is True
        mock_llm.generate_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_message_with_tool_execution(self, agent, mock_tool):
        """Test message processing with tool execution."""
        # Mock LLM to return tool call
        agent.llm.generate_response.return_value = AgentResponse(
            content="I'll help you with that",
            tool_calls=[ToolCall(name="test_tool", args={"input": "test"})]
        )
        
        response = await agent.process_message("Execute test tool")
        
        # Verify tool was called
        mock_tool.execute.assert_called_once_with({"input": "test"})
        assert response.success is True
    
    @pytest.mark.asyncio
    async def test_process_message_error_handling(self, agent):
        """Test error handling in message processing."""
        # Mock LLM to raise exception
        agent.llm.generate_response.side_effect = Exception("LLM error")
        
        with pytest.raises(Exception, match="LLM error"):
            await agent.process_message("Cause error")
```

#### Integration Tests
```python
import pytest
import httpx
from openagent.server.app import app

@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.fixture
    async def client(self):
        """HTTP client for testing."""
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = await client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    @pytest.mark.asyncio
    async def test_chat_endpoint(self, client):
        """Test chat endpoint."""
        payload = {
            "message": "Hello, how are you?",
            "agent": "default"
        }
        
        response = await client.post("/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["success"] is True
    
    @pytest.mark.asyncio
    async def test_chat_streaming(self, client):
        """Test streaming chat endpoint."""
        payload = {
            "message": "Generate a Python function",
            "agent": "default"
        }
        
        async with client.stream(
            "POST", 
            "/chat/stream", 
            json=payload,
            headers={"Accept": "text/event-stream"}
        ) as response:
            assert response.status_code == 200
            
            chunks = []
            async for chunk in response.aiter_text():
                if chunk.strip():
                    chunks.append(chunk)
            
            assert len(chunks) > 0
            # Verify SSE format
            assert any("data:" in chunk for chunk in chunks)
```

#### Performance Tests
```python
import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

@pytest.mark.performance
class TestPerformance:
    """Performance tests for critical paths."""
    
    @pytest.mark.asyncio
    async def test_agent_response_time(self, agent):
        """Test agent response time under load."""
        messages = ["Hello"] * 10
        
        start_time = time.time()
        
        # Process messages concurrently
        tasks = [agent.process_message(msg) for msg in messages]
        responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # Assert all responses successful
        assert all(r.success for r in responses)
        
        # Assert reasonable response time (adjust based on hardware)
        avg_time = (end_time - start_time) / len(messages)
        assert avg_time < 2.0, f"Average response time too slow: {avg_time}s"
    
    @pytest.mark.asyncio
    async def test_concurrent_users(self, client):
        """Test system under concurrent user load."""
        
        async def simulate_user():
            """Simulate a user session."""
            responses = []
            for i in range(5):
                response = await client.post("/chat", json={
                    "message": f"Message {i}",
                    "agent": "default"
                })
                responses.append(response.status_code)
            return responses
        
        # Simulate 10 concurrent users
        tasks = [simulate_user() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Verify all requests successful
        for user_responses in results:
            assert all(status == 200 for status in user_responses)
```

### Testing Configuration

#### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --disable-warnings
    --cov=openagent
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    slow: Slow tests (can be skipped)
    llm: Tests requiring LLM functionality
asyncio_mode = auto
```

#### conftest.py
```python
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = AsyncMock()
    llm.generate_response = AsyncMock(return_value="Mock response")
    llm.stream_response = AsyncMock()
    llm.model_name = "mock-model"
    return llm

@pytest.fixture
def sample_config():
    """Sample configuration for tests."""
    return {
        "model": "test-model",
        "device": "cpu",
        "load_in_4bit": False,
        "max_memory": "1G"
    }

# Test data fixtures
@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        "Hello, world!",
        "Generate a Python function",
        "Explain how to use Git",
        "List files in current directory"
    ]

@pytest.fixture
def mock_policy_engine():
    """Mock policy engine for testing."""
    engine = Mock()
    engine.validate_command = AsyncMock(return_value=PolicyDecision(
        action=PolicyAction.ALLOW,
        reason="Test allowed",
        confidence=1.0
    ))
    return engine
```

This developer guide provides comprehensive information for working with the OpenAgent codebase while maintaining high standards of code quality, testing, and documentation.
