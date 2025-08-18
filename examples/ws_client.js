// Minimal JavaScript WebSocket client for OpenAgent /ws/chat
// Run with: node ws_client.js (Node 18+)
// Usage:
//   node ws_client.js ws://localhost:8000/ws/chat YOUR_TOKEN

const url = process.argv[2] || 'ws://localhost:8000/ws/chat';
const token = process.argv[3] || null;

const headers = token ? { Authorization: `Bearer ${token}` } : {};

// Use 'ws' package for Node.js
// npm install ws
const WebSocket = require('ws');
const ws = new WebSocket(url, { headers });

ws.addEventListener('open', () => {
  ws.send(JSON.stringify({ message: 'Hello from JS WS client' }));
});

ws.addEventListener('message', (ev) => {
  try {
    const data = JSON.parse(ev.data);
    if (data.content) process.stdout.write(data.content);
    if (data.event === 'end') {
      process.stdout.write('\n');
      ws.close();
    }
  } catch (e) {
    process.stdout.write(String(ev.data));
  }
});

ws.addEventListener('error', (err) => {
  console.error('WS error:', err.message || err);
});

