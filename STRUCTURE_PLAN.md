# OpenAgent Project Structure Reorganization Plan

## Current Issues
1. **Duplicate LLM modules**: Multiple overlapping LLM implementations
2. **Inconsistent organization**: Policy and core functionality scattered
3. **Missing architecture alignment**: Structure doesn't match WARP.md specification

## Target Structure (Based on WARP.md Architecture)

```
openagent/
├── __init__.py                 # Main package exports
├── cli.py                      # CLI interface
├── py.typed                    # Type hint marker
│
├── core/                       # Core agent system
│   ├── __init__.py
│   ├── agent.py               # Main Agent implementation
│   ├── base.py                # Abstract base classes (BaseAgent, BaseTool, etc.)
│   ├── llm.py                 # Unified LLM integration (HF + Ollama)
│   ├── llm_ollama.py          # Ollama-specific implementation
│   ├── policy.py              # Security policy engine
│   ├── observability.py       # Metrics, logging, request tracking
│   ├── workflows.py           # Workflow management
│   ├── config.py              # Configuration management
│   ├── exceptions.py          # Custom exceptions
│   ├── context.py             # Context management
│   ├── history.py             # History management
│   └── performance/           # Performance optimization
│       ├── __init__.py
│       ├── memory_manager.py
│       ├── model_cache.py
│       └── optimization.py
│
├── tools/                     # Tools system
│   ├── __init__.py
│   ├── system.py             # CommandExecutor, FileManager, SystemInfo
│   ├── git.py                # GitTool and RepoGrep
│   └── base.py               # Tool base classes (if needed)
│
├── server/                    # Server components
│   ├── __init__.py
│   ├── app.py                # FastAPI web server
│   ├── auth.py               # Authentication manager
│   ├── rate_limiter.py       # Rate limiting
│   ├── models.py             # Pydantic models
│   └── versioning.py         # API versioning
│
├── terminal/                  # Terminal integration
│   ├── __init__.py
│   ├── integration.py        # Shell integration
│   └── validator.py          # Command validation
│
├── websocket/                 # WebSocket support
│   ├── __init__.py
│   ├── handler.py
│   ├── manager.py
│   ├── middleware.py
│   └── models.py
│
├── plugins/                   # Plugin system
│   ├── __init__.py
│   ├── base.py
│   ├── loader.py
│   ├── manager.py
│   ├── registry.py
│   └── validator.py
│
├── ui/                        # User interface components
│   ├── __init__.py
│   ├── blocks.py
│   ├── formatting.py
│   └── renderer.py
│
└── utils/                     # Utilities
    ├── __init__.py
    └── subprocess_utils.py
```

## Consolidation Plan

### 1. LLM Module Consolidation
- **Keep**: `core/llm.py` (main HF integration)
- **Keep**: `core/llm_ollama.py` (Ollama integration)
- **Remove**: `core/llm_base.py` (merge into llm.py)
- **Remove**: `llms/` directory (redundant)

### 2. Policy Module Consolidation
- **Keep**: `core/policy.py` (main policy engine)
- **Remove**: `policy/engine.py` (merge into core/policy.py)

### 3. Core Module Cleanup
- Remove duplicate/experimental modules
- Consolidate context management
- Merge performance modules properly

### 4. Update Imports
- Update all __init__.py files
- Fix cross-module imports
- Ensure CLI works with new structure

## Implementation Steps
1. Create new structure in parallel
2. Move and consolidate modules
3. Update imports and __init__.py files
4. Test functionality
5. Remove old structure
6. Update documentation
