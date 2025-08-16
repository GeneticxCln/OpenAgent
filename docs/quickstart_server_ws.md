# Server and WebSocket Quickstart

This guide walks you through starting the API server and using SSE and WebSocket streaming, with and without auth.

Prerequisites
- Python 3.9+
- Optional: websockets (for WS client examples)

1) Start the FastAPI server

bash:
  uvicorn openagent.server.app:app --host 0.0.0.0 --port 8000

Endpoints:
- OpenAPI docs: http://localhost:8000/docs
- WS (legacy): ws://localhost:8000/ws
- WS (preferred): ws://localhost:8000/ws/chat

2) Stream via SSE (no auth)

bash:
  curl -N -H "Accept: text/event-stream" -H "Content-Type: application/json" \
    -d '{"message":"explain binary search","agent":"default"}' \
    http://localhost:8000/chat/stream

Expect a start event, a series of `data: {"content":"..."}` lines, then an end event.

3) Stream via WebSocket (no auth)

- Python (requires: pip install websockets)

bash:
  python examples/ws_client.py --url ws://localhost:8000/ws/chat

- Node.js (requires: npm i ws)

bash:
  node examples/ws_client.js ws://localhost:8000/ws/chat

4) Enable auth (optional)

If your server enforces auth, supply a token using an Authorization header (Bearer recommended).

- SSE with Bearer token:

bash:
  TOKEN=your_token_here
  curl -N -H "Accept: text/event-stream" -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d '{"message":"secure test"}' \
    http://localhost:8000/chat/stream

- WS with Bearer token:

bash:
  python examples/ws_client.py --url ws://localhost:8000/ws/chat --token "$TOKEN"

- WS with token in query (not recommended, but supported):

bash:
  node examples/ws_client.js "ws://localhost:8000/ws/chat?token=$TOKEN"

5) CLI streaming to the server

- SSE (default):

bash:
  OPENAGENT_API_URL=http://localhost:8000 openagent chat

- WebSocket:

bash:
  openagent chat --api-url http://localhost:8000 --ws --ws-path /ws/chat

- With auth:

bash:
  OPENAGENT_API_URL=https://api.example.com \
  OPENAGENT_API_TOKEN=$TOKEN \
  openagent chat --ws --ws-path /ws/chat

6) Security notes
- Prefer Authorization headers to avoid token leakage in URLs.
- See SECURITY.md for RBAC, rate limiting, and hardening guidance.

Troubleshooting
- If WS fails, the CLI falls back to SSE, then non-streaming.
- Ensure the `websockets` package is installed for Python WS examples.
- Check server logs for errors and verify that auth is configured as expected.

