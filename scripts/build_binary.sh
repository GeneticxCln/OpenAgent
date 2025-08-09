#!/usr/bin/env bash
set -euo pipefail

# Build a standalone binary for OpenAgent using PyInstaller (one-file).
# Usage:
#   ./scripts/build_binary.sh [--name NAME] [--output DIR]
# Requires:
#   - Python venv with pyinstaller installed
#   - Project installable in that environment

NAME="openagent"
OUTDIR="dist"
PYTHON_BIN="python"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      NAME="$2"; shift; shift ;;
    --output)
      OUTDIR="$2"; shift; shift ;;
    --python)
      PYTHON_BIN="$2"; shift; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Ensure output directory exists
mkdir -p "$OUTDIR"

# Ensure pyinstaller is installed
if ! "$PYTHON_BIN" -c "import PyInstaller" >/dev/null 2>&1; then
  echo "Installing pyinstaller..."
  "$PYTHON_BIN" -m pip install --quiet pyinstaller
fi

# Build
echo "Building one-file binary..."
"$PYTHON_BIN" -m PyInstaller \
  --clean \
  --name "$NAME" \
  --onefile \
  --console \
  --hidden-import=dotenv \
  --hidden-import=rich \
  --hidden-import=typer \
  --hidden-import=psutil \
  --hidden-import=yaml \
  --hidden-import=transformers \
  --hidden-import=huggingface_hub \
  --distpath "$OUTDIR" \
  openagent/cli.py

echo "\nBinary created at $OUTDIR/$NAME (platform-specific extension may apply)"

