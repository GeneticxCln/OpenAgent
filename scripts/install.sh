#!/usr/bin/env bash
set -euo pipefail

# OpenAgent user-level installer (no root required)
# - Creates an isolated venv under ~/.local/share/openagent
# - Installs OpenAgent into that venv
# - Creates a shim at ~/.local/bin/openagent
# - Ensures ~/.local/bin is on PATH for zsh and bash

PREFIX_DIR="${HOME}/.local"
APP_DIR="${PREFIX_DIR}/share/openagent"
BIN_DIR="${PREFIX_DIR}/bin"
VENV_DIR="${APP_DIR}/venv"
SHIM_PATH="${BIN_DIR}/openagent"

log() { echo -e "[OpenAgent] $*"; }

# Ensure dirs
mkdir -p "${APP_DIR}" "${BIN_DIR}"

# Create venv
if [ ! -d "${VENV_DIR}" ]; then
  log "Creating virtual environment at ${VENV_DIR}"
  python -m venv "${VENV_DIR}"
fi

# Upgrade pip tooling
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel >/dev/null

# Install the package (from current repo)
log "Installing OpenAgent into ${VENV_DIR}"
"${VENV_DIR}/bin/pip" install -e . >/dev/null

# Create shim
cat > "${SHIM_PATH}" <<'SHIM'
#!/usr/bin/env bash
# OpenAgent launcher shim
APP_VENV="$HOME/.local/share/openagent/venv"
exec "$APP_VENV/bin/openagent" "$@"
SHIM
chmod +x "${SHIM_PATH}"
log "Installed shim: ${SHIM_PATH}"

# Ensure ~/.local/bin on PATH for zsh and bash
ZRC="$HOME/.zshrc"
BRC="$HOME/.bashrc"
LINE='export PATH="$HOME/.local/bin:$PATH"'
ensure_path_line() {
  local rc="$1"
  if [ -f "$rc" ]; then
    if ! grep -q "^export PATH=\"\$HOME/.local/bin:\$PATH\"$" "$rc" 2>/dev/null; then
      echo "$LINE" >> "$rc"
      log "Updated PATH in $rc"
    fi
  fi
}
ensure_path_line "$ZRC" || true
ensure_path_line "$BRC" || true

log "Done. If this is your first install, run: source \"$ZRC\" or open a new terminal."
log "Run: openagent (menu opens by default)"

