# Security and RBAC Guidance

This document outlines security considerations and recommendations for deploying OpenAgent.

Authentication and Authorization
- Optional token-based auth is supported in the API server. Enable and configure via environment and server settings.
- WebSocket /ws/chat supports Authorization header (e.g., Bearer <token>) and token query parameters.
- Prefer Authorization header to avoid token leakage in URLs and logs.

RBAC (Role-Based Access Control)
- If auth is enabled, restrict sensitive endpoints (e.g., /system/info) to admin roles.
- The example server includes a basic check: when auth is enabled, /system/info requires an admin role.
- Consider implementing a policy for agents and tools that may execute commands or read files.

Transport Security
- For public deployments, terminate TLS (HTTPS/WSS). Avoid plain HTTP/WS over untrusted networks.
- Rotate tokens regularly and expire them on a short schedule.

Command Execution Safety
- Default CLI behavior can be explain-only (safe mode) or require approval for risky commands.
- Use policy features and validator to block unsafe commands.
- Prefer sandboxing for tool execution in production.

Secret Management
- Never log secrets.
- Use environment variables or a secret manager to store API keys (e.g., HUGGINGFACE_TOKEN).

Rate Limiting and Abuse Prevention
- Enable rate limiting to mitigate abuse.
- Consider IP-based throttling and per-user quotas.

Audit and Observability
- Enable audit logging for commands and tool usage.
- Integrate with metrics and tracing to observe errors and latency.

Deployment Hardening Checklist
- [ ] TLS enabled (HTTPS/WSS)
- [ ] Auth enabled and tokens scoped/rotating
- [ ] RBAC enforced for sensitive endpoints
- [ ] Rate limiting configured
- [ ] Command policy configured; sandbox enabled where appropriate
- [ ] Secrets stored via environment or secret manager
- [ ] Logging and metrics enabled (no sensitive data in logs)
- [ ] Regular dependency and image scans (e.g., safety, bandit)

