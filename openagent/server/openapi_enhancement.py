"""
OpenAPI enhancement module for comprehensive API documentation.

This module extends the default FastAPI OpenAPI generation with detailed
examples, better descriptions, and comprehensive error responses.
"""

from typing import Dict, Any, List, Optional
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def enhance_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """Generate enhanced OpenAPI schema with comprehensive documentation."""
    
    if app.openapi_schema:
        return app.openapi_schema
    
    # Generate base schema
    openapi_schema = get_openapi(
        title="OpenAgent API",
        version="0.1.3", 
        description="""
# OpenAgent API

A powerful, production-ready AI agent framework API providing terminal assistance, 
code generation, and system operations capabilities.

## Features

- **🤖 AI-Powered Chat**: Interactive conversations with intelligent agents
- **💻 Code Generation**: Generate code in multiple programming languages
- **🔍 Code Analysis**: Analyze and review existing code
- **⚡ Real-time Streaming**: WebSocket and Server-Sent Events support
- **🔒 Security First**: Built-in authentication and rate limiting
- **📊 Observability**: Comprehensive metrics and structured logging

## Authentication

Most endpoints require authentication using Bearer tokens:

```
Authorization: Bearer <your-token>
```

Get a token by calling `/auth/login` with valid credentials.

## API Versioning

This API supports versioning through multiple methods:

1. **Accept Header** (Recommended): `Accept: application/json; version=1.0`
2. **Custom Header**: `X-API-Version: 1.0` 
3. **Path Prefix**: `/v1.0/chat` (future)

Current version: **1.0** | Supported versions: **1.0**

## Rate Limiting

API calls are rate limited per user/IP:
- **Chat endpoints**: 100 requests per minute
- **Code generation**: 20 requests per minute
- **System info**: 30 requests per minute

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset timestamp

## Streaming

The API supports real-time streaming for chat responses:

### Server-Sent Events (SSE)
```bash
curl -N -H "Accept: text/event-stream" -H "Content-Type: application/json" \\
  -H "Authorization: Bearer <token>" \\
  -d '{"message":"explain binary search"}' \\
  http://localhost:8000/chat/stream
```

### WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat');
ws.send(JSON.stringify({message: 'hello'}));
```

## Error Handling

All errors follow a consistent format:

```json
{
  "error": "error_code",
  "message": "Human readable error message",
  "details": {},
  "timestamp": "2023-01-01T00:00:00Z",
  "request_id": "uuid-here"
}
```

Common error codes:
- `authentication_required`: Missing or invalid token
- `rate_limit_exceeded`: Too many requests
- `agent_not_found`: Specified agent doesn't exist
- `processing_failed`: Internal processing error
- `unsupported_api_version`: Invalid API version

## SDKs and Examples

- **Python**: See `examples/python_client.py`
- **JavaScript**: See `examples/js_client.js`
- **CLI**: Use `openagent chat --api-url http://localhost:8000`
        """,
        routes=app.routes,
    )
    
    # Add comprehensive examples and enhanced schemas
    openapi_schema = add_comprehensive_examples(openapi_schema)
    openapi_schema = add_error_responses(openapi_schema)
    openapi_schema = add_security_schemes(openapi_schema)
    openapi_schema = enhance_operation_documentation(openapi_schema)
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def add_comprehensive_examples(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Add comprehensive examples to the OpenAPI schema."""
    
    examples = {
        "ChatRequest": {
            "simple_question": {
                "summary": "Simple question",
                "description": "A basic question to the AI assistant",
                "value": {
                    "message": "What is Python?",
                    "agent": "default"
                }
            },
            "code_help": {
                "summary": "Code help request",
                "description": "Ask for help with coding problems",
                "value": {
                    "message": "How do I implement a binary search in Python?",
                    "agent": "default",
                    "context": {
                        "language": "python",
                        "difficulty": "intermediate"
                    }
                }
            },
            "system_command": {
                "summary": "System command",
                "description": "Request help with system administration",
                "value": {
                    "message": "How do I check disk usage on Linux?",
                    "agent": "default"
                }
            }
        },
        "CodeRequest": {
            "simple_function": {
                "summary": "Simple function",
                "description": "Generate a simple utility function",
                "value": {
                    "description": "Create a function to calculate the factorial of a number",
                    "language": "python",
                    "include_tests": True,
                    "include_docs": True
                }
            },
            "web_scraper": {
                "summary": "Web scraper",
                "description": "Generate a web scraping script",
                "value": {
                    "description": "Create a web scraper to extract product prices from an e-commerce site",
                    "language": "python",
                    "style": "object-oriented",
                    "include_tests": True,
                    "include_docs": True
                }
            }
        },
        "AnalysisRequest": {
            "code_review": {
                "summary": "Code review",
                "description": "Analyze code for potential improvements",
                "value": {
                    "code": "def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    return fibonacci(n-1) + fibonacci(n-2)",
                    "language": "python",
                    "focus": ["performance", "readability"]
                }
            },
            "security_audit": {
                "summary": "Security audit",
                "description": "Analyze code for security vulnerabilities",
                "value": {
                    "code": "import os\\ncommand = input('Enter command: ')\\nos.system(command)",
                    "language": "python", 
                    "focus": ["security"]
                }
            }
        }
    }
    
    # Add examples to components
    if "components" not in schema:
        schema["components"] = {}
    if "schemas" not in schema["components"]:
        schema["components"]["schemas"] = {}
        
    for schema_name, schema_examples in examples.items():
        if schema_name in schema["components"]["schemas"]:
            schema["components"]["schemas"][schema_name]["examples"] = schema_examples
    
    return schema


def add_error_responses(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Add comprehensive error response schemas."""
    
    error_responses = {
        "400": {
            "description": "Bad Request - Invalid input or API version",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                    "examples": {
                        "invalid_input": {
                            "summary": "Invalid input",
                            "value": {
                                "error": "validation_failed",
                                "message": "Invalid request format",
                                "details": {"field": "message", "issue": "required"},
                                "timestamp": "2023-01-01T00:00:00Z",
                                "request_id": "12345678-1234-1234-1234-123456789012"
                            }
                        },
                        "unsupported_version": {
                            "summary": "Unsupported API version", 
                            "value": {
                                "error": "unsupported_api_version",
                                "message": "API version '2.0' is not supported",
                                "details": {
                                    "supported_versions": ["1.0"],
                                    "current_version": "1.0"
                                },
                                "timestamp": "2023-01-01T00:00:00Z",
                                "request_id": "12345678-1234-1234-1234-123456789012"
                            }
                        }
                    }
                }
            }
        },
        "401": {
            "description": "Unauthorized - Authentication required",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                    "example": {
                        "error": "authentication_required",
                        "message": "Valid authentication token required",
                        "timestamp": "2023-01-01T00:00:00Z",
                        "request_id": "12345678-1234-1234-1234-123456789012"
                    }
                }
            }
        },
        "403": {
            "description": "Forbidden - Insufficient permissions",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                    "example": {
                        "error": "insufficient_permissions",
                        "message": "User does not have permission to access this resource",
                        "timestamp": "2023-01-01T00:00:00Z",
                        "request_id": "12345678-1234-1234-1234-123456789012"
                    }
                }
            }
        },
        "404": {
            "description": "Not Found - Resource not found",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                    "example": {
                        "error": "agent_not_found",
                        "message": "Agent 'nonexistent' not found",
                        "timestamp": "2023-01-01T00:00:00Z",
                        "request_id": "12345678-1234-1234-1234-123456789012"
                    }
                }
            }
        },
        "429": {
            "description": "Too Many Requests - Rate limit exceeded",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                    "example": {
                        "error": "rate_limit_exceeded",
                        "message": "Rate limit exceeded. Try again in 60 seconds.",
                        "details": {
                            "limit": 100,
                            "reset_time": "2023-01-01T00:01:00Z"
                        },
                        "timestamp": "2023-01-01T00:00:00Z",
                        "request_id": "12345678-1234-1234-1234-123456789012"
                    }
                }
            }
        },
        "500": {
            "description": "Internal Server Error - Processing failed",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                    "example": {
                        "error": "processing_failed",
                        "message": "Internal server error occurred while processing request",
                        "timestamp": "2023-01-01T00:00:00Z",
                        "request_id": "12345678-1234-1234-1234-123456789012"
                    }
                }
            }
        }
    }
    
    # Add ErrorResponse schema if not exists
    if "components" not in schema:
        schema["components"] = {}
    if "schemas" not in schema["components"]:
        schema["components"]["schemas"] = {}
        
    schema["components"]["schemas"]["ErrorResponse"] = {
        "type": "object",
        "properties": {
            "error": {
                "type": "string",
                "description": "Error code identifying the type of error",
                "example": "authentication_required"
            },
            "message": {
                "type": "string", 
                "description": "Human-readable error message",
                "example": "Valid authentication token required"
            },
            "details": {
                "type": "object",
                "description": "Additional error details",
                "additionalProperties": True
            },
            "timestamp": {
                "type": "string",
                "format": "date-time",
                "description": "Timestamp when the error occurred"
            },
            "request_id": {
                "type": "string",
                "description": "Unique request identifier for tracking",
                "example": "12345678-1234-1234-1234-123456789012"
            }
        },
        "required": ["error", "message", "timestamp", "request_id"]
    }
    
    # Add error responses to all paths
    if "paths" in schema:
        for path_info in schema["paths"].values():
            for operation in path_info.values():
                if isinstance(operation, dict) and "responses" in operation:
                    operation["responses"].update(error_responses)
    
    return schema


def add_security_schemes(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Add comprehensive security schemes documentation."""
    
    if "components" not in schema:
        schema["components"] = {}
    if "securitySchemes" not in schema["components"]:
        schema["components"]["securitySchemes"] = {}
    
    schema["components"]["securitySchemes"].update({
        "HTTPBearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": """
JWT Bearer token authentication. 

To get a token, call `/auth/login` with your credentials:

```bash
curl -X POST http://localhost:8000/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{"username":"user","password":"pass"}'
```

Then use the returned token in the Authorization header:

```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/chat
```
            """
        },
        "ApiKeyQuery": {
            "type": "apiKey",
            "in": "query",
            "name": "token",
            "description": "API key as query parameter (for WebSocket connections only)"
        },
        "ApiKeyHeader": {
            "type": "apiKey", 
            "in": "header",
            "name": "X-API-Key",
            "description": "API key in header (alternative to Bearer token)"
        }
    })
    
    return schema


def enhance_operation_documentation(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance individual operation documentation."""
    
    if "paths" not in schema:
        return schema
    
    # Enhanced documentation for specific endpoints
    enhancements = {
        "/chat": {
            "post": {
                "summary": "Chat with AI Agent",
                "description": """
Send a message to an AI agent and receive a response.

This endpoint processes natural language requests and can:
- Answer questions about programming, system administration, etc.
- Generate code snippets and explanations
- Provide step-by-step guides for technical tasks
- Execute safe system commands (when configured)

**Usage Tips:**
- Be specific in your requests for better responses
- Mention the programming language when asking code questions
- Use the `context` field to provide additional information
- Set `stream: true` for real-time response streaming

**Processing Time:**
- Simple questions: < 2 seconds
- Code generation: 2-10 seconds
- Complex analysis: 10-30 seconds
                """,
                "tags": ["Chat"],
                "operationId": "chat_with_agent"
            }
        },
        "/chat/stream": {
            "post": {
                "summary": "Stream Chat Response", 
                "description": """
Stream a chat response using Server-Sent Events (SSE).

This endpoint provides real-time streaming of AI responses, allowing
clients to display content as it's generated rather than waiting for
completion.

**Event Format:**
- `event: start` - Indicates response generation has begun
- `data: {"content": "..."}` - Incremental content chunks
- `event: end` - Indicates response is complete

**Client Example:**
```javascript
const eventSource = new EventSource('/chat/stream', {
  headers: { 'Authorization': 'Bearer ' + token }
});

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.content) {
    console.log(data.content);
  }
};
```
                """,
                "tags": ["Chat", "Streaming"]
            }
        },
        "/code/generate": {
            "post": {
                "summary": "Generate Code",
                "description": """
Generate code based on a natural language description.

Supports multiple programming languages and can generate:
- Functions and classes
- Complete applications
- Code snippets and utilities
- Unit tests and documentation

**Supported Languages:**
- Python, JavaScript, TypeScript
- Java, C#, C++, Go, Rust
- HTML, CSS, SQL
- Shell scripts (Bash, PowerShell)

**Best Practices:**
- Provide clear, specific descriptions
- Mention any libraries/frameworks to use
- Specify code style preferences
- Request tests and documentation when needed
                """,
                "tags": ["Code Generation"]
            }
        },
        "/code/analyze": {
            "post": {
                "summary": "Analyze Code",
                "description": """
Analyze code and provide insights on quality, security, and performance.

**Analysis Types:**
- **Security**: Identify potential vulnerabilities
- **Performance**: Suggest optimization opportunities  
- **Style**: Code formatting and best practices
- **Maintainability**: Code complexity and structure
- **Testing**: Test coverage and quality suggestions

**Focus Areas:**
Use the `focus` parameter to target specific aspects:
- `["security"]` - Security vulnerability scan
- `["performance"]` - Performance optimization
- `["style", "readability"]` - Code quality review
- `["testing"]` - Test coverage analysis
                """,
                "tags": ["Code Analysis"]
            }
        }
    }
    
    # Apply enhancements
    for path, methods in enhancements.items():
        if path in schema["paths"]:
            for method, enhancement in methods.items():
                if method in schema["paths"][path]:
                    schema["paths"][path][method].update(enhancement)
    
    return schema
