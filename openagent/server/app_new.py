"""
FastAPI web server for OpenAgent.

Provides REST API access to OpenAgent functionality with authentication,
rate limiting, and comprehensive error handling.
"""

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .auth import AuthConfig, AuthManager

# Import server components
from .models import (
    AgentCreateRequest,
    AgentStatus,
    AnalysisRequest,
    AnalysisResponse,
    ChatRequest,
    ChatResponse,
    CodeRequest,
    CodeResponse,
    ErrorResponse,
    HealthResponse,
    LoginRequest,
    LoginResponse,
    ModelInfo,
    SystemInfoResponse,
    User,
)
from .rate_limiter import RateLimitConfig, RateLimiter, rate_limit_dependency

try:
    from ..agent import Agent
    from ..models.llm import HuggingFaceLLM
    from ..models.registry import MODEL_REGISTRY
    from ..tools.command import CommandTool
    from ..tools.file import FileTool
    from ..tools.system import SystemTool
except ImportError:
    # Fallback imports for running as standalone module
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent import Agent
    from models.llm import HuggingFaceLLM
    from models.registry import MODEL_REGISTRY
    from tools.command import CommandTool
    from tools.file import FileTool
    from tools.system import SystemTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize authentication and rate limiting
auth_manager = AuthManager()
rate_limiter = RateLimiter()

# Global state
agents: Dict[str, Agent] = {}
server_start_time = time.time()


# Helper functions for error handling
def create_error_response(
    error_type: str, message: str, details: Optional[Dict[str, Any]] = None
) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        error=error_type,
        message=message,
        timestamp=datetime.utcnow().isoformat(),
        details=details,
    )


# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting OpenAgent server...")

    # Initialize default agent
    try:
        default_llm = HuggingFaceLLM(model_name="tiny-llama")
        default_agent = Agent(
            name="default",
            description="Default OpenAgent assistant",
            llm=default_llm,
            tools=[SystemTool(), FileTool(), CommandTool()],
        )
        agents["default"] = default_agent
        logger.info("Default agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize default agent: {e}")

    logger.info("OpenAgent server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down OpenAgent server...")

    # Clean up agents
    for agent_name, agent in agents.items():
        try:
            if hasattr(agent, "llm") and agent.llm:
                agent.llm.unload_model()
            logger.info(f"Agent '{agent_name}' shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down agent '{agent_name}': {e}")

    agents.clear()
    logger.info("OpenAgent server shut down")


# FastAPI app configuration
app = FastAPI(
    title="OpenAgent API",
    description="REST API for OpenAgent - AI Assistant with Tool Integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"]  # Configure appropriately for production
)


# Add authentication dependency
async def get_current_user(request: Request) -> User:
    """Get the current authenticated user."""
    return await auth_manager.get_current_user()


# Add rate limiting dependency
async def check_rate_limit(
    request: Request, user: User = Depends(get_current_user)
) -> None:
    """Check rate limits for the current user."""
    await rate_limiter.check_request(request, user.id if user else None)


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with consistent error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            "HTTP Error", str(exc.detail), {"status_code": exc.status_code}
        ).dict(),
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions with error logging."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=create_error_response(
            "Internal Server Error",
            "An unexpected error occurred",
            {"exception_type": type(exc).__name__},
        ).dict(),
    )


# Authentication endpoints
@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest) -> LoginResponse:
    """Authenticate user and return access token."""
    try:
        return auth_manager.login(request.username, request.password)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error",
        )


@app.post("/auth/register", response_model=User)
async def register(
    request: LoginRequest, current_user: User = Depends(get_current_user)
) -> User:
    """Register a new user (admin only)."""
    # For now, only allow admin to create users
    if current_user.username != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Only admin can create users"
        )

    try:
        return auth_manager.create_user(request.username, request.password)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User creation failed",
        )


@app.get("/auth/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)) -> User:
    """Get current user information."""
    return current_user


# Rate limiting info endpoint
@app.get("/rate-limits")
async def get_rate_limit_info(
    request: Request, current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current rate limit information."""
    client_ip = rate_limiter.get_client_ip(request)
    return rate_limiter.get_rate_limit_info(current_user.id, client_ip)


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        agents=len(agents),
        uptime=time.time() - server_start_time,
        system_info={"python_version": sys.version, "platform": sys.platform},
    )


# Chat endpoint
@app.post(
    "/chat", response_model=ChatResponse, dependencies=[Depends(check_rate_limit)]
)
async def chat(
    request: ChatRequest, current_user: User = Depends(get_current_user)
) -> ChatResponse:
    """Process a chat message with the specified agent."""
    agent_name = request.agent or "default"

    if agent_name not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    agent = agents[agent_name]

    try:
        start_time = time.time()
        response = await agent.process_message(request.message, request.context or {})
        processing_time = time.time() - start_time

        return ChatResponse(
            message=response,
            metadata={"user_id": current_user.id, "context": request.context},
            processing_time=processing_time,
            agent=agent_name,
            tools_used=getattr(agent, "last_tools_used", []),
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing message: {str(e)}"
        )


# Code generation endpoint
@app.post(
    "/generate-code",
    response_model=CodeResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def generate_code(
    request: CodeRequest, current_user: User = Depends(get_current_user)
) -> CodeResponse:
    """Generate code based on description."""
    agent_name = request.agent or "default"

    if agent_name not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    agent = agents[agent_name]

    try:
        start_time = time.time()

        # Create code generation prompt
        prompt = f"""Generate {request.language} code for the following request:
        
Description: {request.description}
Language: {request.language}
Style: {request.style or 'clean and readable'}
Include tests: {request.include_tests}
Include documentation: {request.include_docs}

Please provide clean, well-documented code that follows best practices."""

        response = await agent.process_message(prompt)
        processing_time = time.time() - start_time

        return CodeResponse(
            code=response,
            language=request.language,
            description=request.description,
            processing_time=processing_time,
            suggestions=[
                "Consider adding error handling",
                "Add unit tests for better coverage",
                "Document public interfaces",
            ],
        )

    except Exception as e:
        logger.error(f"Code generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating code: {str(e)}")


# Code analysis endpoint
@app.post(
    "/analyze-code",
    response_model=AnalysisResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def analyze_code(
    request: AnalysisRequest, current_user: User = Depends(get_current_user)
) -> AnalysisResponse:
    """Analyze code and provide insights."""
    agent_name = request.agent or "default"

    if agent_name not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    agent = agents[agent_name]

    try:
        start_time = time.time()

        # Create code analysis prompt
        focus_areas = request.focus or ["quality", "security", "performance"]
        prompt = f"""Analyze the following {request.language} code:

```{request.language}
{request.code}
```

Focus areas: {', '.join(focus_areas)}

Please provide:
1. Overall code quality assessment
2. Potential issues or bugs
3. Security considerations
4. Performance optimization suggestions
5. Best practice recommendations"""

        response = await agent.process_message(prompt)
        processing_time = time.time() - start_time

        return AnalysisResponse(
            analysis=response,
            language=request.language,
            processing_time=processing_time,
            issues=[],  # Would be populated by actual analysis
            suggestions=[
                "Consider adding type hints",
                "Add proper error handling",
                "Improve variable naming",
            ],
            metrics={
                "complexity": "medium",
                "maintainability": "good",
                "test_coverage": "unknown",
            },
        )

    except Exception as e:
        logger.error(f"Code analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing code: {str(e)}")


# System information endpoint
@app.get(
    "/system-info",
    response_model=SystemInfoResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def get_system_info(
    current_user: User = Depends(get_current_user),
) -> SystemInfoResponse:
    """Get system information."""
    if "default" not in agents:
        raise HTTPException(status_code=500, detail="Default agent not available")

    agent = agents["default"]

    try:
        # Find system tool
        system_tool = None
        for tool in agent.tools:
            if hasattr(tool, "name") and "system" in tool.name.lower():
                system_tool = tool
                break

        if system_tool:
            response = await agent.process_message("Get system information")
        else:
            # Fallback system info
            import platform

            import psutil

            response = f"""System Information:
- Platform: {platform.platform()}
- Python Version: {sys.version}
- CPU Count: {psutil.cpu_count()}
- Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB
- Disk Usage: {psutil.disk_usage('/').free / (1024**3):.2f} GB free"""

        return SystemInfoResponse(
            content=response,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": current_user.id,
            },
        )

    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting system info: {str(e)}"
        )


# Agent management endpoints
@app.get("/agents", response_model=List[AgentStatus])
async def list_agents(
    current_user: User = Depends(get_current_user),
) -> List[AgentStatus]:
    """List all available agents."""
    agent_statuses = []

    for name, agent in agents.items():
        # Get basic agent information
        model_name = "unknown"
        if hasattr(agent, "llm") and agent.llm:
            model_name = getattr(agent.llm, "model_name", "unknown")

        tools = []
        if hasattr(agent, "tools") and agent.tools:
            tools = [getattr(tool, "name", str(tool)) for tool in agent.tools]

        agent_statuses.append(
            AgentStatus(
                name=name,
                description=getattr(agent, "description", "No description"),
                model=model_name,
                is_processing=False,  # Would need to track this in agent
                tools=tools,
                message_count=0,  # Would need to track this in agent
                uptime=time.time() - server_start_time,
            )
        )

    return agent_statuses


@app.post("/agents", response_model=AgentStatus, status_code=201)
async def create_agent(
    request: AgentCreateRequest, current_user: User = Depends(get_current_user)
) -> AgentStatus:
    """Create a new agent."""
    if request.name in agents:
        raise HTTPException(
            status_code=400, detail=f"Agent '{request.name}' already exists"
        )

    try:
        # Initialize LLM
        llm = HuggingFaceLLM(model_name=request.model_name)

        # Initialize tools
        tools = []
        for tool_name in request.tools:
            if tool_name == "system":
                tools.append(SystemTool())
            elif tool_name == "file":
                tools.append(FileTool())
            elif tool_name == "command":
                tools.append(CommandTool())

        # Create agent
        agent = Agent(
            name=request.name, description=request.description, llm=llm, tools=tools
        )

        agents[request.name] = agent

        return AgentStatus(
            name=request.name,
            description=request.description,
            model=request.model_name,
            is_processing=False,
            tools=request.tools,
            message_count=0,
            uptime=0.0,
        )

    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating agent: {str(e)}")


@app.get("/agents/{agent_name}", response_model=AgentStatus)
async def get_agent(
    agent_name: str, current_user: User = Depends(get_current_user)
) -> AgentStatus:
    """Get information about a specific agent."""
    if agent_name not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    agent = agents[agent_name]

    model_name = "unknown"
    if hasattr(agent, "llm") and agent.llm:
        model_name = getattr(agent.llm, "model_name", "unknown")

    tools = []
    if hasattr(agent, "tools") and agent.tools:
        tools = [getattr(tool, "name", str(tool)) for tool in agent.tools]

    return AgentStatus(
        name=agent_name,
        description=getattr(agent, "description", "No description"),
        model=model_name,
        is_processing=False,
        tools=tools,
        message_count=0,
        uptime=time.time() - server_start_time,
    )


@app.delete("/agents/{agent_name}", status_code=204)
async def delete_agent(agent_name: str, current_user: User = Depends(get_current_user)):
    """Delete an agent."""
    if agent_name not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    if agent_name == "default":
        raise HTTPException(status_code=400, detail="Cannot delete the default agent")

    try:
        agent = agents[agent_name]
        if hasattr(agent, "llm") and agent.llm:
            agent.llm.unload_model()

        del agents[agent_name]
        logger.info(f"Agent '{agent_name}' deleted successfully")

    except Exception as e:
        logger.error(f"Error deleting agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {str(e)}")


# Model management endpoints
@app.get("/models", response_model=List[ModelInfo])
async def list_models(
    current_user: User = Depends(get_current_user),
) -> List[ModelInfo]:
    """List all available models."""
    models = []

    # Add models from registry
    for category, category_models in MODEL_REGISTRY.items():
        for name, config in category_models.items():
            models.append(
                ModelInfo(
                    name=name,
                    path=config.get("path", ""),
                    category=category,
                    description=config.get("description", f"{category} model: {name}"),
                    size=config.get("size"),
                    parameters=config.get("parameters"),
                    memory_required=config.get("memory_required"),
                    is_loaded=False,  # Would need to check if loaded
                )
            )

    return models


if __name__ == "__main__":
    import time

    # Add server start time tracking
    server_start_time = time.time()

    # Get command line arguments
    host = "127.0.0.1"
    port = 8000

    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}. Using default port 8000.")

    if len(sys.argv) > 2:
        host = sys.argv[2]

    print(f"Starting OpenAgent server on {host}:{port}")
    print(f"API documentation available at: http://{host}:{port}/docs")
    print(
        f"Authentication: {'Enabled' if auth_manager.config.auth_enabled else 'Disabled'}"
    )
    print(f"Rate limiting: {'Enabled' if rate_limiter.config.enabled else 'Disabled'}")

    if not auth_manager.config.auth_enabled:
        print("\nDefault credentials (when auth is enabled):")
        print("  Admin: admin / admin123")
        print("  User:  user / user123")

    uvicorn.run("app:app", host=host, port=port, reload=True, log_level="info")
