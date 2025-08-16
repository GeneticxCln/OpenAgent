# API Stability and Endpoint Contracts

This document describes the supported API surface for the 0.1.x series.

Supported Endpoints
- POST /chat
  - Request: { message: string, agent?: string }
  - Response: { message: string, metadata?: object, processing_time?: number, agent?: string }
- POST /chat/stream (SSE)
  - Request: { message: string, agent?: string }
  - Events:
    - event: start; data: { agent: string }
    - data: { content: string } (multiple)
    - event: end; data: {}
- WS /ws/chat (JSON frames)
  - Client sends: { message: string, agent?: string }
  - Server sends:
    - { event: "start", agent: string }
    - { content: string } (multiple)
    - { event: "end" }

Authentication
- Optional token-based auth using Authorization: Bearer <token> for HTTP and WS.
- WS may also accept a query param token key (e.g., ?token=...). Prefer headers.

Compatibility Policy (0.1.x)
- The payload structures above are stable. Adding optional fields is allowed.
- The legacy /ws endpoint is deprecated and may be removed in a future minor release.
- Breaking changes will be called out in the changelog and a migration guide.

Versioning
- The FastAPI app exposes version via OpenAPI metadata and health endpoints.

Notes
- Servers should avoid leaking tokens in logs. Prefer Authorization headers.
- SSE and WS streaming semantics match the CLI expectations.

