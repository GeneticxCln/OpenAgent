"""
Redaction utilities to remove secrets from exported content.
"""
from __future__ import annotations

import os
import re
from typing import Dict
from pathlib import Path

from dotenv import dotenv_values

COMMON_SECRET_KEYS = {
    "API_KEY", "API_TOKEN", "SECRET", "PASSWORD", "PASS", "TOKEN",
    "OPENAI_API_KEY", "HF_TOKEN", "HUGGINGFACE_TOKEN",
}

# Simple patterns for common token formats (best-effort)
TOKEN_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # generic secret-like
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),  # GitHub PAT
]


def load_env_secrets() -> Dict[str, str]:
    """Load secrets from current environment and .env file without printing them."""
    secrets: Dict[str, str] = {}
    # Environment
    for k, v in os.environ.items():
        if any(key in k.upper() for key in COMMON_SECRET_KEYS):
            secrets[k] = v
    # .env
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        try:
            vals = dotenv_values(str(env_path))
            for k, v in (vals or {}).items():
                if v and any(key in k.upper() for key in COMMON_SECRET_KEYS):
                    secrets[k] = v
        except Exception:
            pass
    return secrets


def redact_text(text: str) -> str:
    """Redact secret-like values from text using env/.env values and regex patterns."""
    if not text:
        return text
    redacted = text
    # Replace known values
    secrets = load_env_secrets()
    for k, v in secrets.items():
        if not v:
            continue
        try:
            redacted = redacted.replace(v, f"{{{{REDACTED_{k}}}}}")
        except Exception:
            continue
    # Replace token patterns
    for pat in TOKEN_PATTERNS:
        redacted = pat.sub("{REDACTED_TOKEN}", redacted)
    return redacted
