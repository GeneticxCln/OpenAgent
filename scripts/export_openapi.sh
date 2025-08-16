#!/usr/bin/env bash
set -euo pipefail

# Export OpenAPI schema from FastAPI app and write to openapi.json
# Requires uvicorn, curl, and a free port.

PORT=${1:-8042}
HOST=127.0.0.1
APP="openagent.server.app:app"

# Start server in background
uvicorn "$APP" --host "$HOST" --port "$PORT" --log-level error > /dev/null 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT

# Wait for readiness
for i in $(seq 1 30); do
  if curl -fsS "http://$HOST:$PORT/healthz" > /dev/null; then
    break
  fi
  sleep 0.3
done

# Fetch OpenAPI JSON
curl -fsS "http://$HOST:$PORT/openapi.json" > openapi.json

# Stop server
kill $PID 2>/dev/null || true
wait $PID 2>/dev/null || true

echo "Wrote openapi.json"

