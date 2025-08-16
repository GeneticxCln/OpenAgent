#!/usr/bin/env bash
set -euo pipefail

if ! command -v pre-commit >/dev/null 2>&1; then
  echo "pre-commit not found. Install with: pip install pre-commit" >&2
  exit 1
fi

pre-commit install
pre-commit autoupdate || true

echo "Pre-commit installed. You can run: pre-commit run --all-files"

