"""
Lightweight routing heuristics for fast, Warp-like responsiveness.

This module provides a small, dependency-free classifier to decide whether
we should use tools, answer directly, or explain-only, without invoking
heavy LLM calls.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal


class Route(Enum):
    DIRECT = "direct"  # answer directly via LLM (no tools)
    TOOL = "tool"  # use tools (system, file, git, grep)
    EXPLAIN_ONLY = "explain"  # explain the command; don't run
    CODEGEN = "codegen"  # generate code


# Precompile basic regexes for speed
_CMD_LIKE = re.compile(
    r"(^|\b)(ls|cat|grep|find|git|docker|kubectl|curl|wget|python|pip|npm|node)\b"
)
_GIT_LIKE = re.compile(r"\bgit\b|pull request|diff|commit|branch", re.I)
_FILE_LIKE = re.compile(
    r"\b(read|write|append|create|delete|move|rename)\b.*\b(file|folder|directory|path)\b",
    re.I,
)
_SYSTEM_LIKE = re.compile(
    r"\b(cpu|memory|ram|disk|usage|processes|system info|uptime|load)\b", re.I
)
_SEARCH_LIKE = re.compile(r"\b(search|grep|ripgrep|rg|scan|find in)\b", re.I)
_EXPLAIN_LIKE = re.compile(r"\b(explain|what does|is this safe|meaning of)\b", re.I)
_CODEGEN_LIKE = re.compile(
    r"\b(write|generate|implement|create)\b.*\b(code|function|class|script|api)\b", re.I
)
_DANGEROUS = re.compile(
    r"\b(rm\s+-rf|chmod\s+777|chown\s+/.+|curl\s+.*\|\s*sh|wget\s+.*\|\s*sh)\b", re.I
)


def classify(text: str) -> Route:
    t = text.strip()
    tl = t.lower()

    # Explain-only if explicitly asked
    if _EXPLAIN_LIKE.search(tl) or tl.startswith("explain "):
        return Route.EXPLAIN_ONLY

    # Code generation
    if _CODEGEN_LIKE.search(tl):
        return Route.CODEGEN

    # Obvious git/file/system/search intents use tools
    if (
        _GIT_LIKE.search(tl)
        or _FILE_LIKE.search(tl)
        or _SYSTEM_LIKE.search(tl)
        or _SEARCH_LIKE.search(tl)
    ):
        return Route.TOOL

    # Command-like text: suggest tools, but prefer explain-only if dangerous
    if _CMD_LIKE.search(t):
        return Route.EXPLAIN_ONLY if _DANGEROUS.search(tl) else Route.TOOL

    # Very short questions likely fine as direct answer
    if len(t) <= 60 and tl.endswith("?"):
        return Route.DIRECT

    # Default to direct to minimize latency unless tool keywords are present
    return Route.DIRECT
