# Release Notes for v0.1.3 (2025-08-16)

Highlights
- Fixed syntax error in legacy /ws endpoint and unified streaming logic.
- WebSocket models migrated to Pydantic v2 APIs (no more deprecation warnings).
- CLI helpers for WS URL and HTTP headers covered by unit tests.
- README updates for CLI auth and streaming, and minor cleanup.

Details
- Server
  - Removed invalid walrus operator usage in /ws legacy handler and inlined streaming.
  - Ensured /ws/chat supports both Authorization headers and token query parameters.
  - App version exposed as 0.1.3 in FastAPI metadata and health endpoints.
- WebSocket models
  - Replaced Config with model_config = ConfigDict(use_enum_values=True).
  - Replaced json()/parse_raw() with model_dump_json()/model_validate_json().
- CLI
  - Helpers _build_ws_url_and_headers and _build_http_headers added previously now have tests.
  - Streaming behavior: prefer WS when requested, fallback to SSE, then non-streaming.
- Docs
  - README: expanded CLI streaming/auth section and fixed arrow rendering.
- Tests
  - Added tests/cli/test_cli_helpers.py.
  - All tests: 191 passed.

Upgrade Notes
- No breaking changes. If you relied on deprecated Pydantic v1-style methods on WebSocket models, adjust to the new methods.

Thanks to contributors and testers!

