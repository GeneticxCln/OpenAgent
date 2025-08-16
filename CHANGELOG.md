# Changelog

## 0.1.3 - 2025-08-16

Added
- Unit tests for CLI helpers constructing WS URLs and HTTP headers.
- Release notes for 0.1.3.

Changed
- WebSocket models migrated to Pydantic v2 APIs to eliminate deprecation warnings.
- README streaming/auth section polished.

Fixed
- Syntax error in legacy /ws endpoint; inlined streaming logic and ensured consistent start/end events.

## 0.1.2 - 2025-08-16

Added
- CLI WebSocket streaming for chat with graceful fallback to SSE/non-streaming.
- CLI authentication flags for API/WS: --api-token, --auth-scheme, --ws-token-query-key.
- New helpers in CLI to construct WS URLs and HTTP headers.
- FastAPI /ws/chat endpoint that supports Authorization header and token query params.
- Unit/integration tests for CLI streaming and server WebSocket auth.
- README updates documenting streaming and authentication usage.

Changed
- Prefer local lightweight model by default during tests to avoid external dependencies.

