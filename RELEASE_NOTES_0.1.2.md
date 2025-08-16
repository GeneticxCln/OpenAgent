# OpenAgent 0.1.2 Release Notes

Date: 2025-08-16

Highlights
- CLI WebSocket streaming for chat with graceful fallback to SSE and non-streaming
- Authentication support in CLI for API/WS (Authorization header and optional WS query token)
- New FastAPI WebSocket endpoint /ws/chat with optional auth (header or query)
- Documentation updates and new example clients (Python & JavaScript)
- Comprehensive tests for CLI streaming and server WS authentication

Changes
- CLI
  - Added flags: --ws, --ws-path, --api-token, --auth-scheme, --ws-token-query-key
  - Implemented SSE and WS streaming with fallbacks
  - Added helpers to build WS URLs and HTTP headers
- Server
  - Added /ws/chat WebSocket endpoint supporting Authorization header or token query parameters
  - Enforces authentication when AUTH_ENABLED=true
- Docs
  - README updated with streaming and auth usage
  - Example WS clients added under examples/
- Tests
  - CLI helper tests for WS/HTTP header construction
  - Chat loop streaming tests with WS->SSE fallback and auth headers
  - Server /ws/chat tests for auth disabled/enabled scenarios

Upgrade Notes
- No breaking changes to existing CLI commands
- If using WebSocket streaming, ensure the websockets package is installed (already in dependencies)
- To enable server auth, set AUTH_ENABLED=true and provide JWT_SECRET_KEY

Thanks to all contributors and users for feedback!

