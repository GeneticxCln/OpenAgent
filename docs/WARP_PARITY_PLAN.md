# OpenAgent — Warp Parity Improvement Plan

This document captures a concrete, actionable plan to close gaps vs. Warp Agent while strengthening OpenAgent’s core strengths (privacy, local models, extensibility).

Scope
- Immediate: Re-enable command intelligence, integrate the block UI into the CLI, align safety defaults, and harden secrets redaction.
- Near term: Add a safe patch-editing tool, expand VCS host integrations, improve docs/tests.
- Medium term: Collaboration/sharing, optional cloud sync/team features, analytics.

Tracking
- Use the checklists below to manage execution. Each task includes acceptance criteria.
- Keep updates in this file until spun out into individual RFCs under docs/design/.

---

1) Current gaps (validated in code)

- Command intelligence not wired in:
  - openagent/terminal/integration.py falls back to dummy shims because the modules it imports (command_intelligence, command_templates, context_v2) exist only in openagent_backup.
- Block UI not in interactive flow:
  - CLI prints to console directly; TerminalRenderer/BlockManager not used, so no visual command blocks or folding UX.
- Collaboration features missing:
  - No session sharing, notebooks, or team workflows.
- Cloud/team features limited:
  - FastAPI server exists with streaming + token auth, but no device sync, org settings, or multi-tenant RBAC.
- Patch-level editing primitive absent:
  - FileManager supports read/write/copy/move/delete with safe paths, but no atomic, uniquely-scoped search/replace tool for safe refactors.
- Advanced VCS host integration absent:
  - No GitHub/GitLab PR/issue integration.
- Secrets UX not enforced globally in CLI printing:
  - Redaction utilities exist (core/redact.py) but not universally applied to CLI outputs.
- Safety defaults slightly inconsistent:
  - create_agent docstring vs. CLI default for unsafe_exec.
- Docs claim features beyond current wired code:
  - Some capabilities listed as IMPLEMENTED are in openagent_backup or not used in runtime paths.

---

2) Milestone plan

Milestone 1 (1–2 weeks)
- Re-enable command intelligence
  - Promote modules from openagent_backup/core into openagent/core:
    - command_intelligence.py
    - command_templates.py
    - context_v2/history_intelligence.py
    - context_v2/project_analyzer.py
  - Update imports in openagent/terminal/integration.py (remove shims).
  - Acceptance criteria:
    - Typo autocorrect, flag/arg completion, and project-aware suggestions work in interactive sessions.
    - New unit tests cover suggestions for git/docker/kubectl examples.

- Integrate block UI into CLI
  - Route interactive chat/streamed output through TerminalRenderer/BlockManager.
  - Auto-fold long output; add keyboard shortcuts (already defined in UI) to the live view.
  - Provide a CLI flag to disable block UI if needed (e.g., --no-ui-blocks).
  - Acceptance criteria:
    - Commands render as blocks with status, output folding on long output, and navigation shortcuts.
    - Streaming shows incremental chunks in an AI response block.

- Align safety defaults
  - Ensure --unsafe-exec defaults to False and update the create_agent() docstring to reflect safe-by-default behavior.
  - Acceptance criteria:
    - CLI help and README are consistent; tests confirm default is explain-only unless explicitly enabled.

- Enforce secrets redaction in CLI outputs
  - Pipe any user-visible output through redact_text().
  - For shell integration snippets, add guidance about secret-safe printing and avoiding echo of secrets.
  - Acceptance criteria:
    - Any detected secret-like values are displayed as {{REDACTED_*}} in CLI outputs.

Milestone 2 (2–4 weeks)
- Safe patch-editing tool
  - Implement openagent/tools/patch.py with a uniquely-scoped search/replace primitive:
    - Reject ambiguous matches; support multi-hunk edits; support dry-run/preview; provide rollback on failure.
  - Acceptance criteria:
    - Unit tests for unique-match enforcement, dry-run, multi-hunk behavior, and rollback.

- VCS host integration (GitHub first)
  - Add a tool that reads PRs/issues/comments safely via GitHub API (or gh CLI if present) without paging.
  - Support environment variable-based auth (never echo secrets; only reference $GITHUB_TOKEN-like envs).
  - Acceptance criteria:
    - Can fetch PR list/details, comments, and diff metadata read-only.

- Documentation and tests alignment
  - Update WARP_COMPARISON.md to reflect what is actually wired.
  - Expand tests for command intelligence + block UI.

Milestone 3 (4–8 weeks)
- Collaboration & sharing
  - Session export/import (JSONL or markdown transcript with blocks).
  - Optional local sharing endpoint with signed tokens; later org-managed sharing and RBAC.

- Optional cloud sync/team features
  - Encrypted settings sync and org-distributed policy.

- Analytics (privacy-first)
  - Opt-in usage stats with local aggregation; lightweight dashboard or metrics exporter.

---

3) Workstreams and detailed tasks (checklists)

A. Command intelligence (restore and enable)
- [ ] Move modules from openagent_backup/core to openagent/core
- [ ] Fix imports in openagent/terminal/integration.py
- [ ] Add unit tests for completions, flag/args, typo autocorrect
- [ ] Add integration test for interactive CLI completions
- [ ] Update README usage to reference completions/templates

Acceptance criteria
- [ ] Typo fixes (e.g., gti -> git) surfaced inline
- [ ] Flag/arg completion works for git/docker examples
- [ ] Project-type detection drives better suggestions

B. Block UI integration (CLI)
- [ ] Create a renderer instance for interactive chat
- [ ] Wrap streamed AI output in AI_RESPONSE block
- [ ] Wrap executed commands in COMMAND block; attach output/error
- [ ] Auto-fold long outputs; hotkeys functional
- [ ] Add --no-ui-blocks to disable block UI

Acceptance criteria
- [ ] Streaming is incremental and scrolls within a block
- [ ] Folding/expand works for long outputs
- [ ] Block navigation shortcuts operate as documented

C. Safety defaults and docs
- [ ] Standardize default: explain-only (unsafe_exec=False)
- [ ] Fix create_agent docstring and CLI help text
- [ ] Confirm tests checking default behavior

D. Secrets redaction
- [ ] Apply redact_text to all user-visible responses in CLI
- [ ] Add tests for redaction patterns and env-driven redaction
- [ ] Update shell snippet docs to avoid echoing secrets

E. Patch-editing tool
- [ ] Implement openagent/tools/patch.py with unique search/replace semantics
- [ ] Support dry-run, multi-hunk edits, rollback on failure
- [ ] Add tests (happy-path, ambiguity, rollback)

F. VCS host integration (GitHub)
- [ ] Tool to list PRs/issues and fetch PR details/diffs
- [ ] Auth via $GITHUB_TOKEN in env only; never print token
- [ ] No pager, non-interactive; resilient to rate-limits
- [ ] Add read-only tests using recorded fixtures/mocks

G. Docs/tests alignment
- [ ] Update WARP_COMPARISON.md to match wired features
- [ ] Expand README quickstart with block UI and completions
- [ ] Ensure tests cover new paths (CLI + UI + intelligence)

---

4) Design notes and interfaces

4.1 Patch tool early interface
```text path=null start=null
name: patch_editor
input:
  {
    "files": [
      { "path": "path/to/file", "hunks": [ {"search": "exact string", "replace": "new string"} ] }
    ],
    "require_unique": true,
    "dry_run": false,
    "rollback_on_failure": true
  }
output:
  {
    "success": true/false,
    "summary": "...",
    "diff_preview": "..." (when dry_run=true),
    "errors": []
  }
```

Semantics
- search must match exactly once per hunk when require_unique=true; otherwise fail.
- Multi-hunk edits applied atomically; failure rolls back all changes.

4.2 GitHub integration
- Read-only operations first (PR list/details/comments/diffs).
- Auth from env var (e.g., GITHUB_TOKEN). Never print token; only pass header using process env.
- Prefer HTTP API; optionally gh CLI if present, with --no-pager behavior.

4.3 Secrets redaction pipeline
- All CLI prints pass through redact_text().
- For shell integration docs: discourage printing secrets and suggest environment variable usage.

---

5) Risks and mitigations

- Performance regressions (block UI, intelligence)
  - Mitigate with lazy initialization, background warmup, and fallbacks.
- Security regressions
  - Keep policy engine as the gate for all executions; maintain strict mode (OPENAGENT_EXEC_STRICT) and safe_paths.
- UX complexity
  - Provide --no-ui-blocks flag and clear help text.
- API stability
  - Stabilize CLI flags and document any breaking changes ahead of minor version bumps.

---

6) Metrics and success criteria

- Startup time unchanged (within +/-10%) after Milestone 1.
- Command intelligence suggestion hit-rate in tests >= 80% for covered commands.
- 0 regressions in safety tests (policy, file policy, timeouts).
- UI adoption: block UI enabled by default and covered by tests.

---

7) Execution guide (quick commands)

- Run tests
```bash path=null start=null
pytest -q
```

- Lint/format/type-check
```bash path=null start=null
make fmt && make lint && make type
```

- Local server for manual testing
```bash path=null start=null
uvicorn openagent.server.app:app --host 127.0.0.1 --port 8000
```

- Interactive chat (safe by default)
```bash path=null start=null
openagent chat
```

- Interactive chat with block UI disabled (once implemented)
```bash path=null start=null
openagent chat --no-ui-blocks
```

---

8) Ownership and sequencing

Suggested order of work within Milestone 1:
1) Command intelligence (restore and wire) — unlocks terminal UX.
2) Block UI integration — visible UX uplift.
3) Safety defaults alignment — remove confusion.
4) Secrets redaction — improve default hygiene.

Each subtask should PR separately with tests and doc updates. Merge once green on CI.

---

9) Post-merge cleanups
- Remove any unused backup modules after migration.
- Align WARP_COMPARISON.md with reality.
- Update README sections and examples.

---

Appendix A — Cross-references
- WARP_COMPARISON.md: baseline feature mapping.
- WARP.md: developer guide for local workflows.
- README.md and ROADMAP.md: keep in sync with this plan as milestones land.

