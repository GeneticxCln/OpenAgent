# OpenAgent Hardening Guide

This guide summarizes recommended settings and practices to harden OpenAgent for production or security-sensitive environments.

Audience: Operators and developers deploying OpenAgent locally or as an API server.

1. Execution strictness and shell fallback
- Default strict mode (server): The API server now defaults OPENAGENT_EXEC_STRICT=1, which disables shell fallback in CommandExecutor. Only the safe structured exec subset is used.
- CLI hardening: For CLI runs, export OPENAGENT_EXEC_STRICT=1 to match server defaults.
- When to relax: Only if you understand the risks and need complex shell constructs (pipes/redirects). Prefer sandboxing instead.

2. Policy presets and modes
- Modes:
  - explain_only (recommended default): Always explain, never execute unless explicitly approved.
  - approve: Require approval for medium/high risk; allows safe commands.
  - execute: Allow safe-by-default (not recommended in production).
- High-level tips:
  - Keep block_high_risk=True
  - require_approval_for_medium=True for unknown scripts or admin tasks
  - allow_admin_override=False for shared environments
- Example (CLI):
  - openagent policy strict  # block-by-default + block risky
  - openagent policy relaxed # warn-by-default + block risky

3. Safe paths and restricted paths for FileManager
- Safe paths (where mutating ops are allowed): e.g., ~/Documents, ~/Downloads, /tmp
- Restricted paths (never allow writes): /etc, /boot, /sys, /proc, /usr/bin, /usr/sbin, /bin, /sbin, ~/.ssh, ~/.gnupg
- Strategy:
  - Expand safe_paths minimally to your project workspace(s)
  - Do NOT include system directories in safe_paths
  - Review and tighten restricted_paths if running on shared hosts

4. Sandboxing
- Enable sandbox_mode=True in the policy to execute risky commands inside an isolated environment with resource limits.
- Backends (auto-selected):
  - bubblewrap (bwrap) – preferred when available
  - firejail – good alternative
  - unshare – baseline isolation with namespaces
  - container (optional, explicit): Requires container_image and podman|docker installed
- All backends apply ulimit resource limits inside the sandbox shell.
- For container backend, set:
  - policy.sandbox_backend="container"
  - policy.container_image="docker.io/library/alpine:latest"

5. Authentication, RBAC, and server
- Keep auth enabled for the API server; use bearer tokens via Authorization headers.
- Use rate limiting (enabled by default) and trusted hosts middleware.
- For administrative endpoints (/system/info), restrict to admin role when auth is enabled.
- Network exposure: Bind to localhost or put behind a reverse proxy with TLS for remote access.

6. Audit logs and integrity
- Audit logs are stored under ~/.openagent/audit/ as JSONL with tamper-evident hash chaining.
- Regularly verify integrity:
  - From code: PolicyEngine.verify_audit_integrity() or verify_audit_integrity_report()
  - Export reports with export_audit_report(output_format="summary"|"json")
- Rotate and secure these logs; they contain sensitive operational metadata.

7. Environment variables and secrets
- HUGGINGFACE_TOKEN / HF_TOKEN: Set only if needed for private models.
- OPENAGENT_PRECACHE=1: Pre-download model weights during maintenance windows.
- Do not print or log secret values; rely on built-in redaction and avoid echoing secrets.

8. Model selection and resource control
- Prefer local-only models (Ollama/tiny code models) for privacy-sensitive deployments.
- Use 4-bit quantization where supported (load_in_4bit=True).
- Monitor memory and latency (Prometheus metrics endpoint /metrics).

9. Command allow/deny lists
- Maintain allowlist_patterns for safe commands (ls, pwd, git status, etc.).
- Keep denylist_patterns for dangerous patterns (rm -rf /, curl|sh, chmod 777, etc.).
- Add organization-specific patterns as needed.

10. Operational checklist
- Server: OPENAGENT_EXEC_STRICT=1, auth enabled, rate limiting on, trusted hosts set
- Policy: explain_only or approve, block_high_risk=True, require_approval_for_medium=True
- File operations: safe_paths limited to workspaces; restricted_paths intact
- Sandboxing: sandbox_mode=True; backend auto (bwrap/firejail/unshare) or container (explicit)
- Observability: metrics endpoint protected; audit logs stored securely and verified
- Models: prefer local (ollama/tiny) when possible; pin versions; pre-cache images/models

Notes
- Defaults are conservative. Review and override via environment variables or policy configuration where appropriate.
- Changes in this guide correspond to the 0.1.3+ hardening improvements (server strict mode, sandbox backend selection, LLM cache pool).

