# ADR-001: Agent-Tool Pattern Architecture

## Status
Accepted

## Context

OpenAgent needs a flexible and extensible architecture to support various AI agents that can perform different tasks. The system must:

- Support multiple types of agents (code generation, system operations, general chat)
- Allow agents to use different combinations of tools
- Enable easy addition of new tools and agents
- Maintain clear separation of concerns
- Support both simple and complex workflows

The key challenge is designing an architecture that is both powerful enough for complex tasks and simple enough for basic operations.

## Decision

We adopt the **Agent-Tool Pattern** as the core architectural pattern for OpenAgent:

### Core Components

1. **Agents**: Orchestrate tasks by combining LLM capabilities with tool usage
2. **Tools**: Encapsulate specific functionality (file operations, system commands, etc.)
3. **Tool Selector**: Intelligently choose appropriate tools for each task
4. **Base Abstractions**: Provide consistent interfaces for agents and tools

### Pattern Structure

```python
class BaseAgent:
    """Abstract base for all agents."""
    async def process_message(message: str) -> AgentResponse
    
class BaseTool:
    """Abstract base for all tools."""
    async def execute(input_data: Dict[str, Any]) -> ToolResult

class Agent(BaseAgent):
    """Concrete agent implementation."""
    - llm: BaseLLM
    - tools: List[BaseTool]
    - tool_selector: ToolSelector
    
    async def process_message(message: str) -> AgentResponse:
        1. Analyze message intent
        2. Select appropriate tools
        3. Generate LLM response with tool calls
        4. Execute tools if needed
        5. Synthesize final response
```

### Key Principles

- **Composition over Inheritance**: Agents compose tools rather than inheriting behavior
- **Single Responsibility**: Each tool has one clear purpose
- **Interface Segregation**: Tools implement only the interfaces they need
- **Dependency Injection**: Tools are injected into agents, enabling easy testing and configuration

## Consequences

### Positive
- **Flexibility**: Easy to create new agents by combining existing tools
- **Extensibility**: New tools can be added without modifying existing agents
- **Testability**: Tools and agents can be tested independently
- **Reusability**: Tools can be shared across different agents
- **Maintainability**: Clear separation of concerns makes code easier to understand and modify

### Negative
- **Complexity**: More abstraction layers than a monolithic approach
- **Indirection**: Tool calls require additional abstraction overhead
- **Learning Curve**: Developers need to understand the pattern to contribute effectively

### Neutral
- **Performance**: Slight overhead from abstraction, but negligible for AI workloads
- **Memory Usage**: Additional objects for tools and selectors

## Alternatives Considered

### 1. Monolithic Agent
**Description**: Single agent class with all functionality built-in.

**Pros**:
- Simple to understand and implement
- Direct method calls, no abstraction overhead
- Fewer files and classes

**Cons**:
- Difficult to extend with new functionality
- Hard to test individual components
- Code duplication across different agent types
- Violates single responsibility principle

**Why Rejected**: Doesn't scale well for a framework intended to support diverse use cases.

### 2. Plugin-Based Architecture
**Description**: Core agent with functionality added via plugins.

**Pros**:
- Very flexible and extensible
- Clear separation between core and extensions
- Can load/unload functionality dynamically

**Cons**:
- More complex to implement and understand
- Requires plugin discovery and lifecycle management
- Potential security concerns with dynamic loading
- Overkill for current requirements

**Why Rejected**: Too complex for initial implementation, though we may add plugin support later.

### 3. Service-Oriented Architecture
**Description**: Each tool as a separate service with RPC/HTTP communication.

**Pros**:
- Maximum separation and scalability
- Language-agnostic tool implementation
- Independent deployment and scaling

**Cons**:
- Network latency and reliability concerns
- Complex service discovery and management
- Overkill for single-node deployment
- Debugging and testing complexity

**Why Rejected**: Adds unnecessary complexity for a framework that primarily runs on single nodes.

### 4. Functional Composition
**Description**: Tools as pure functions composed together.

**Pros**:
- Simple and predictable
- Easy to test and reason about
- No state management concerns

**Cons**:
- Limited support for stateful operations
- Difficult to handle async operations cleanly
- No natural place for tool configuration
- Harder to implement complex tool behaviors

**Why Rejected**: Too restrictive for the variety of operations OpenAgent needs to support.

## Implementation Details

### Tool Interface
```python
class BaseTool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    async def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given input."""
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate tool input before execution."""
        pass
```

### Agent Implementation
```python
class Agent:
    def __init__(self, name: str, model_name: str, tools: List[BaseTool]):
        self.name = name
        self.llm = get_llm(model_name)
        self.tools = {tool.name: tool for tool in tools}
        self.tool_selector = SmartToolSelector(self.llm)
    
    async def process_message(self, message: str) -> AgentResponse:
        # Select relevant tools
        selected_tools = await self.tool_selector.select_tools(
            message, list(self.tools.values())
        )
        
        # Generate response with tool context
        response = await self.llm.generate_response(
            message, available_tools=selected_tools
        )
        
        # Execute tools if requested
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool = self.tools[tool_call.name]
                result = await tool.execute(tool_call.args)
                response.tool_results.append(result)
        
        return response
```

## Migration Strategy

1. **Phase 1**: Implement base abstractions and core agent
2. **Phase 2**: Migrate existing functionality to tool pattern
3. **Phase 3**: Add tool selector and smart tool selection
4. **Phase 4**: Optimize performance and add advanced features

## Success Metrics

- **Developer Experience**: Time to implement new tools < 30 minutes
- **Test Coverage**: Each tool and agent independently testable
- **Performance**: < 10% overhead compared to direct implementation
- **Adoption**: New contributors can understand and extend the pattern

## Related ADRs

- [ADR-002: Multi-LLM Provider Support](ADR-002-multi-llm-support.md) - Complements this pattern
- [ADR-003: Policy-Driven Security](ADR-003-policy-driven-security.md) - Security layer integration
- [ADR-005: Plugin Architecture](ADR-005-plugin-architecture.md) - Future extension of this pattern
