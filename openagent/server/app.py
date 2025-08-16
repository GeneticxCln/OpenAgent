"""
FastAPI web server for OpenAgent.

Provides REST API access to OpenAgent functionality with authentication,
rate limiting, and comprehensive error handling.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from openagent.core.agent import Agent
from openagent.core.exceptions import AgentError, ToolError
from openagent.tools.system import CommandExecutor, FileManager, SystemInfo
from openagent.server.auth import AuthManager
from openagent.server.rate_limiter import RateLimiter
from openagent.server.models import (
    ChatRequest, ChatResponse, AgentStatus, ModelInfo,
    CodeRequest, CodeResponse, AnalysisRequest, AnalysisResponse,
    SystemInfoResponse, ErrorResponse, User, LoginRequest, LoginResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Observability
from openagent.core.observability import (
    configure_observability,
    get_logger,
    get_metrics_collector,
    get_request_tracker,
)
obs_logger = get_logger(__name__)
metrics = get_metrics_collector()
tracker = get_request_tracker()

# Global instances
auth_manager = AuthManager()
rate_limiter = RateLimiter()
agents: Dict[str, Agent] = {}

# Security
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting OpenAgent server...")
    
    # Initialize default agent (prefer Gemini if key present)
    import os
    def _default_model():
        if os.getenv("GEMINI_API_KEY"):
            return "gemini-1.5-flash"
        return os.getenv("DEFAULT_MODEL", "tiny-llama")

    default_agent = Agent(
        name="WebAgent",
        description="OpenAgent web interface assistant",
        model_name=_default_model(),
        tools=[CommandExecutor(), FileManager(), SystemInfo()]
    )
    agents["default"] = default_agent
    
    logger.info("OpenAgent server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down OpenAgent server...")
    for agent in agents.values():
        if hasattr(agent, 'llm') and agent.llm:
            await agent.llm.unload_model()
    agents.clear()
    logger.info("OpenAgent server shut down")


# Create FastAPI app
app = FastAPI(
    title="OpenAgent API",
    description="REST API for OpenAgent - AI-powered terminal assistant",
    version="0.1.1",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.localhost"]
)

# Request ID and metrics middleware
from fastapi.responses import Response, PlainTextResponse
import uuid as _uuid

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    # Configure observability on first request if needed
    configure_observability()

    request_id = request.headers.get("X-Request-ID") or str(_uuid.uuid4())
    path = request.scope.get("path", request.url.path)
    method = request.method
    start = time.time()

    # Set context for logs
    tracker.start_request(request_id=request_id)
    obs_logger.set_context(request_id=request_id)

    try:
        response: Response = await call_next(request)
        status = response.status_code
    except Exception as e:
        status = 500
        obs_logger.error("Unhandled exception during request", error=e, metadata={"path": path, "method": method})
        raise
    finally:
        duration = time.time() - start
        metrics.record_request(method, path, status, duration)
        # Clear context
        tracker.end_request()
        obs_logger.clear_context()

    # Add request id header
    response.headers["X-Request-ID"] = request_id
    return response

# WebSocket support
from fastapi import WebSocket, WebSocketDisconnect
from openagent.websocket import WebSocketManager, WebSocketHandler, WebSocketMessage, MessageType

ws_manager = WebSocketManager()

# Helper to lookup agents for handler

def _lookup_agent(name: str):
    return agents.get(name)

# Optional auth: use existing auth_manager if tokens are enabled
async def _verify_token(token: str):
    try:
        payload = auth_manager.verify_token(token)
        return payload
    except Exception:
        return None

ws_handler = WebSocketHandler(
    send=lambda cid, msg: ws_manager.send_message(cid, msg),
    lookup_agent=_lookup_agent,
    authenticate_token=_verify_token,
)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Handle WebSocket connections for real-time communication."""
    info = ConnectionInfo()
    info.client_ip = ws.client.host if ws.client else None
    info.user_agent = ws.headers.get("user-agent")

    connection_id = await ws_manager.accept(ws, info)

    try:
        while True:
            raw_text = await ws.receive_text()
            await ws_handler.handle(connection_id, ws_manager.get_client(connection_id).info, raw_text)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        # Send error then close
        try:
            await ws_manager.send_message(connection_id, WebSocketMessage(type=MessageType.ERROR, data={"error": str(e)}))
        except Exception:
            pass
    finally:
        await ws_manager.disconnect(connection_id)


# Authentication dependency
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """Get current authenticated user."""
    if not credentials or not credentials.credentials:
        return None
    try:
        payload = auth_manager.verify_token(credentials.credentials)
        if not payload:
            return None
        user_id = payload.get("sub")
        if not user_id:
            return None
        user = auth_manager.get_user_by_id(user_id)
        return user
    except Exception as e:
        logger.warning(f"Authentication failed: {e}")
        return None


# Rate limiting dependency
async def check_rate_limit(request: Request, user: Optional[User] = Depends(get_current_user)):
    """Check rate limits for the request."""
    user_id = user.id if user else None
    await rate_limiter.check_request(request, user_id)


# Error handlers
@app.exception_handler(AgentError)
async def agent_error_handler(request: Request, exc: AgentError):
    """Handle agent errors."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="agent_error",
            message=str(exc),
timestamp=datetime.now(timezone.utc).isoformat()
        ).dict()
    )


@app.exception_handler(ToolError)
async def tool_error_handler(request: Request, exc: ToolError):
    """Handle tool errors."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="tool_error", 
            message=str(exc),
timestamp=datetime.now(timezone.utc).isoformat()
        ).dict()
    )


# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint (deprecated; use /healthz)."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.1",
        "agents": len(agents)
    }

@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.1",
        "agents": len(agents)
    }

@app.get("/readyz")
async def readyz():
    # Simple readiness: default agent exists
    ready = "default" in agents
    return {
        "status": "ok" if ready else "starting",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.1",
        "agents": len(agents)
    }

# Metrics endpoint
@app.get("/metrics")
async def metrics_endpoint():
    data = metrics.get_metrics()
    # Use Prometheus content type if available
    content_type = "text/plain; version=0.0.4; charset=utf-8"
    try:
        from prometheus_client import CONTENT_TYPE_LATEST as _CTL
        content_type = _CTL
    except Exception:
        pass
    return Response(content=data, media_type=content_type)


# Authentication endpoints
@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest) -> LoginResponse:
    """Authenticate user and return access token."""
    user = auth_manager.authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = auth_manager.create_access_token({"sub": user.id})
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        user=user,
        expires_in=auth_manager.config.access_token_expire_minutes * 60
    )


# Chat endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    user: Optional[User] = Depends(get_current_user),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Process a chat message with the agent."""
    agent_name = request.agent or "default"
    
    if agent_name not in agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found"
        )
    
    agent = agents[agent_name]
    
    try:
        # Process message
        start_time = time.time()
        response = await agent.process_message(request.message)
        processing_time = time.time() - start_time
        
        # Log usage
        if user:
            background_tasks.add_task(
                log_usage,
                user.id,
                "chat",
                {"agent": agent_name, "processing_time": processing_time}
            )
        
        return ChatResponse(
            message=response.content,
            metadata=response.metadata,
            processing_time=processing_time,
            agent=agent_name
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )


# Code generation endpoints
@app.post("/code/generate", response_model=CodeResponse)
async def generate_code(
    request: CodeRequest,
    background_tasks: BackgroundTasks,
    user: Optional[User] = Depends(get_current_user),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Generate code based on description."""
    agent_name = request.agent or "default"
    
    if agent_name not in agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found"
        )
    
    agent = agents[agent_name]
    
    try:
        start_time = time.time()
        code = await agent.llm.generate_code(
            request.description,
            request.language
        )
        processing_time = time.time() - start_time
        
        # Log usage
        if user:
            background_tasks.add_task(
                log_usage,
                user.id,
                "code_generation",
                {"language": request.language, "processing_time": processing_time}
            )
        
        return CodeResponse(
            code=code,
            language=request.language,
            description=request.description,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating code: {str(e)}"
        )


@app.post("/code/analyze", response_model=AnalysisResponse) 
async def analyze_code(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    user: Optional[User] = Depends(get_current_user),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Analyze code and provide insights."""
    agent_name = request.agent or "default"
    
    if agent_name not in agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found"
        )
    
    agent = agents[agent_name]
    
    try:
        start_time = time.time()
        analysis = await agent.llm.analyze_code(
            request.code,
            request.language
        )
        processing_time = time.time() - start_time
        
        # Log usage
        if user:
            background_tasks.add_task(
                log_usage,
                user.id,
                "code_analysis", 
                {"language": request.language, "processing_time": processing_time}
            )
        
        return AnalysisResponse(
            analysis=analysis,
            language=request.language,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Code analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing code: {str(e)}"
        )


# System endpoints
@app.get("/system/info", response_model=SystemInfoResponse)
async def get_system_info(
    user: Optional[User] = Depends(get_current_user),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Get system information."""
    if "default" not in agents:
        raise HTTPException(
            status_code=500,
            detail="Default agent not available"
        )
    
    agent = agents["default"]
    system_tool = None
    
    # Find system info tool
    for tool in agent.tools:
        if tool.name == "system_info":
            system_tool = tool
            break
    
    if not system_tool:
        raise HTTPException(
            status_code=500,
            detail="System info tool not available"
        )
    
    try:
        result = await system_tool.execute("overview")
        return SystemInfoResponse(
            content=result.content,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting system info: {str(e)}"
        )


# Agent management endpoints
@app.get("/agents", response_model=List[AgentStatus])
async def list_agents(
    user: Optional[User] = Depends(get_current_user)
):
    """List all available agents."""
    agent_list = []
    
    for name, agent in agents.items():
        status = agent.get_status()
        model_info = agent.llm.get_model_info() if hasattr(agent, 'llm') and agent.llm else {}
        
        agent_list.append(AgentStatus(
            name=name,
            description=status["description"],
            model=model_info.get("model_name", "unknown"),
            is_processing=status["is_processing"],
            tools=status["tools"],
            message_count=status["message_history_length"]
        ))
    
    return agent_list


@app.get("/agents/{agent_name}/status", response_model=AgentStatus)
async def get_agent_status(
    agent_name: str,
    user: Optional[User] = Depends(get_current_user)
):
    """Get status of a specific agent."""
    if agent_name not in agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found"
        )
    
    agent = agents[agent_name]
    status = agent.get_status()
    model_info = agent.llm.get_model_info() if hasattr(agent, 'llm') and agent.llm else {}
    
    return AgentStatus(
        name=agent_name,
        description=status["description"],
        model=model_info.get("model_name", "unknown"),
        is_processing=status["is_processing"],
        tools=status["tools"],
        message_count=status["message_history_length"]
    )


@app.post("/agents/{agent_name}/reset")
async def reset_agent(
    agent_name: str,
    user: Optional[User] = Depends(get_current_user),
    _rate_limit: None = Depends(check_rate_limit)
):
    """Reset an agent's conversation history."""
    if agent_name not in agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found"
        )
    
    agent = agents[agent_name]
    agent.reset()
    
    return {"message": f"Agent '{agent_name}' has been reset"}


# Models endpoint
@app.get("/models", response_model=List[ModelInfo])
async def list_models(
    user: Optional[User] = Depends(get_current_user)
):
    """List all available models."""
    from openagent.core.llm import ModelConfig
    
    models = []
    
    # Add code models
    for name, path in ModelConfig.CODE_MODELS.items():
        models.append(ModelInfo(
            name=name,
            path=path,
            category="code",
            description=f"Code-focused model: {name}"
        ))
    
    # Add chat models
    for name, path in ModelConfig.CHAT_MODELS.items():
        models.append(ModelInfo(
            name=name,
            path=path,
            category="chat",
            description=f"Chat model: {name}"
        ))
    
    # Add lightweight models
    for name, path in ModelConfig.LIGHTWEIGHT_MODELS.items():
        models.append(ModelInfo(
            name=name,
            path=path,
            category="lightweight",
            description=f"Lightweight model: {name}"
        ))
    
    return models


# Background task functions
async def log_usage(user_id: str, operation: str, metadata: Dict[str, Any]):
    """Log API usage for analytics."""
    # In a real implementation, this would write to a database
    logger.info(f"Usage: user={user_id}, operation={operation}, metadata={metadata}")


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "openagent.server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
