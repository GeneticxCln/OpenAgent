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
from datetime import datetime, timedelta

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
from openagent.server.auth import AuthManager, User
from openagent.server.rate_limit import RateLimiter
from openagent.server.models import (
    ChatRequest, ChatResponse, AgentStatus, ModelInfo,
    CodeRequest, CodeResponse, AnalysisRequest, AnalysisResponse,
    SystemInfoResponse, ErrorResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    # Initialize default agent
    default_agent = Agent(
        name="WebAgent",
        description="OpenAgent web interface assistant",
        model_name="tiny-llama",
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
    version="0.1.0",
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


# Authentication dependency
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """Get current authenticated user."""
    if not credentials:
        return None
    
    try:
        user = await auth_manager.verify_token(credentials.credentials)
        return user
    except Exception as e:
        logger.warning(f"Authentication failed: {e}")
        return None


# Rate limiting dependency
async def check_rate_limit(request: Request, user: Optional[User] = Depends(get_current_user)):
    """Check rate limits for the request."""
    client_ip = request.client.host
    user_id = user.id if user else None
    
    if not await rate_limiter.check_limit(client_ip, user_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )


# Error handlers
@app.exception_handler(AgentError)
async def agent_error_handler(request: Request, exc: AgentError):
    """Handle agent errors."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="agent_error",
            message=str(exc),
            timestamp=datetime.utcnow().isoformat()
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
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "agents": len(agents)
    }


# Authentication endpoints
@app.post("/auth/login")
async def login(username: str, password: str):
    """Authenticate user and return access token."""
    user = await auth_manager.authenticate(username, password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )
    
    token = await auth_manager.create_token(user)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user.dict()
    }


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
