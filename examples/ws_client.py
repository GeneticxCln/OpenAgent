# Minimal Python WebSocket client for OpenAgent /ws/chat
# Requires: websockets
# Usage:
#   python ws_client.py --url ws://localhost:8000/ws/chat --token YOUR_TOKEN

import asyncio
import argparse
import json
from typing import Optional

try:
    import websockets  # type: ignore
except Exception:
    raise SystemExit("Please install websockets: pip install websockets")


async def main(url: str, token: Optional[str]):
    headers = [("Authorization", f"Bearer {token}")] if token else None
    async with websockets.connect(url, extra_headers=headers) as ws:
        # Send a simple chat message payload
        await ws.send(json.dumps({"message": "Hello from Python WS client"}))
        while True:
            msg = await ws.recv()
            try:
                data = json.loads(msg)
                if data.get("content"):
                    print(data["content"], end="", flush=True)
                if data.get("event") == "end":
                    print()
                    break
            except Exception:
                print(str(msg), end="")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8000/ws/chat")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()
    asyncio.run(main(args.url, args.token))

