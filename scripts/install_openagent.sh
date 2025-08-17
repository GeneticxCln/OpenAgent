#!/usr/bin/env bash
set -euo pipefail

# OpenAgent Installer for Linux (CachyOS) with zsh
# - Creates a Python virtual environment (default)
# - Installs OpenAgent from the current repository
# - Sets up config at ~/.config/openagent
# - Adds zsh integration and completions
# - Optionally configures systemd user services (daemon and server)
#
# Usage:
#   bash scripts/install_openagent.sh [options]
#
# Options (defaults shown):
#   --prefix DIR                  Installation base (venv lives here) [~/.local/share/openagent]
#   --repo DIR                    Source repo directory [/home/sasha/OpenAgent]
#   --venv                        Use a dedicated Python venv (default)
#   --no-venv                     Install with pip --user instead of venv
#   --zsh-integrate               Append zsh snippet to ~/.zshrc (default)
#   --no-zsh-integrate            Do not modify ~/.zshrc (print snippet only)
#   --install-completion          Install zsh completion to ~/.config/openagent/_openagent (default)
#   --no-install-completion       Do not install completion (print snippet only)
#   --create-services             Create systemd user services for daemon and server (default)
#   --no-create-services          Skip creating systemd services
#   --enable-daemon               Enable and start daemon service after install (default)
#   --no-enable-daemon            Do not enable/start daemon
#   --enable-server               Enable and start server service after install (disabled by default)
#   --no-enable-server            Do not enable/start server (default)
#   # Cloud key prompts removed in local-only build
#   --server-host HOST            Server bind host [127.0.0.1]
#   --server-port PORT            Server port [8000]
#   --daemon-host HOST            Daemon host [127.0.0.1]
#   --daemon-port PORT            Daemon port [8765]
#   --yes                         Assume yes for any non-destructive prompts
#
# Notes:
# - This script does not require root. If dependencies are missing, it suggests pacman commands.
# - It will not print or log secrets.

PREFIX_DEFAULT="$HOME/.local/share/openagent"
REPO_DEFAULT="/home/sasha/OpenAgent"
ZSHRC_FILE="$HOME/.zshrc"
CONFIG_DIR="$HOME/.config/openagent"
COMPLETION_DIR="$CONFIG_DIR"
COMPLETION_FILE="$COMPLETION_DIR/_openagent"
KEYS_ENV="$CONFIG_DIR/keys.env"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
VENV_DIR_DEFAULT="$PREFIX_DEFAULT/venv"

# Defaults
PREFIX="$PREFIX_DEFAULT"
REPO_DIR="$REPO_DEFAULT"
USE_VENV=1
DO_ZSH_INTEGRATION=1
INSTALL_COMPLETION=1
CREATE_SERVICES=1
ENABLE_DAEMON=1
ENABLE_SERVER=0
SERVER_HOST="127.0.0.1"
SERVER_PORT=8000
DAEMON_HOST="127.0.0.1"
DAEMON_PORT=8765
ASSUME_YES=0

say() { printf '%b\n' "$*"; }
info() { say "[INFO] $*"; }
warn() { say "[WARN] $*"; }
err()  { say "[ERROR] $*" >&2; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || return 1
}

confirm() {
  local prompt="$1"; shift || true
  if [ "$ASSUME_YES" = "1" ]; then return 0; fi
  read -r -p "$prompt [y/N]: " ans || true
  case "$ans" in
    y|Y|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

print_zsh_snippet() {
  cat <<'EOF'
# --- OpenAgent integration (BEGIN) ---
# Add OpenAgent venv bin directory to PATH
if [ -d "$HOME/.local/share/openagent/venv/bin" ]; then
  export PATH="$HOME/.local/share/openagent/venv/bin:$PATH"
fi

# Load completions from ~/.config/openagent
if [ -d "$HOME/.config/openagent" ]; then
  fpath=($HOME/.config/openagent $fpath)
fi

# Initialize completion if not already
if [ -n "$ZSH_VERSION" ]; then
  autoload -U compinit
  if ! typeset -f compinit >/dev/null; then
    autoload -U compinit
  fi
  if [ -z "$ZDOTDIR" ] || [ ! -f "$ZDOTDIR/.zcompdump" ]; then
    compinit -i
  else
    compinit -i -C
  fi
fi
# --- OpenAgent integration (END) ---
EOF
}

install_completion_file() {
  mkdir -p "$COMPLETION_DIR"
  cat >"$COMPLETION_FILE" <<'EOF'
#compdef openagent
_arguments '*: :-\>cmds'

case $state in
  cmds)
    local -a subcmds
    subcmds=(
      chat run blocks workflow fix exec do models doctor setup integrate serve policy validate completion plugin daemon
    )
    _describe 'command' subcmds
    ;;
esac
EOF
}

create_systemd_units() {
  mkdir -p "$SYSTEMD_USER_DIR"

  # Daemon service
  cat >"$SYSTEMD_USER_DIR/openagent-daemon.service" <<EOF
[Unit]
Description=OpenAgent Background Daemon
After=network-online.target

[Service]
Type=simple
Environment="PATH=$PREFIX/venv/bin:/usr/bin"
ExecStart=$PREFIX/venv/bin/openagent daemon --host $DAEMON_HOST --port $DAEMON_PORT
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF

  # Server service
  cat >"$SYSTEMD_USER_DIR/openagent-server.service" <<EOF
[Unit]
Description=OpenAgent FastAPI Server
After=network-online.target

[Service]
Type=simple
Environment="PATH=$PREFIX/venv/bin:/usr/bin"
ExecStart=$PREFIX/venv/bin/openagent serve --host $SERVER_HOST --port $SERVER_PORT
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF
}

suggest_pacman() {
  local pkglist="$*"
  warn "Missing dependency detected. You can install with:"
  say "  sudo pacman -S --needed $pkglist"
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --prefix) PREFIX="$2"; shift 2 ;;
      --repo) REPO_DIR="$2"; shift 2 ;;
      --venv) USE_VENV=1; shift ;;
      --no-venv) USE_VENV=0; shift ;;
      --zsh-integrate) DO_ZSH_INTEGRATION=1; shift ;;
      --no-zsh-integrate) DO_ZSH_INTEGRATION=0; shift ;;
      --install-completion) INSTALL_COMPLETION=1; shift ;;
      --no-install-completion) INSTALL_COMPLETION=0; shift ;;
      --create-services) CREATE_SERVICES=1; shift ;;
      --no-create-services) CREATE_SERVICES=0; shift ;;
      --enable-daemon) ENABLE_DAEMON=1; shift ;;
      --no-enable-daemon) ENABLE_DAEMON=0; shift ;;
      --enable-server) ENABLE_SERVER=1; shift ;;
      --no-enable-server) ENABLE_SERVER=0; shift ;;
      --server-host) SERVER_HOST="$2"; shift 2 ;;
      --server-port) SERVER_PORT="$2"; shift 2 ;;
      --daemon-host) DAEMON_HOST="$2"; shift 2 ;;
      --daemon-port) DAEMON_PORT="$2"; shift 2 ;;
      --yes) ASSUME_YES=1; shift ;;
      -h|--help)
        sed -n '1,80p' "$0" | sed 's/^# \{0,1\}//' | sed '/^$/q'
        exit 0 ;;
      *)
        err "Unknown option: $1"; exit 2 ;;
    esac
  done
}

main() {
  parse_args "$@"

  info "Installer configuration:"
  say "  PREFIX:            $PREFIX"
  say "  REPO_DIR:          $REPO_DIR"
  say "  USE_VENV:          $USE_VENV"
  say "  ZSH INTEGRATION:   $([ "$DO_ZSH_INTEGRATION" = 1 ] && echo yes || echo no)"
  say "  COMPLETION:         $([ "$INSTALL_COMPLETION" = 1 ] && echo install || echo skip)"
  say "  SERVICES:          $([ "$CREATE_SERVICES" = 1 ] && echo create || echo skip)"
  say "  ENABLE DAEMON:     $([ "$ENABLE_DAEMON" = 1 ] && echo yes || echo no)"
  say "  ENABLE SERVER:     $([ "$ENABLE_SERVER" = 1 ] && echo yes || echo no)"

  # Verify repo directory
  if [ ! -d "$REPO_DIR" ]; then
    err "Repository directory not found: $REPO_DIR"
    exit 1
  fi
  if [ ! -f "$REPO_DIR/pyproject.toml" ] && [ ! -f "$REPO_DIR/setup.py" ]; then
    warn "No pyproject.toml or setup.py found in $REPO_DIR. pip install . may fail."
  fi

  # Check dependencies: python, pip, venv module
  if ! need_cmd python3; then
    suggest_pacman python
    exit 1
  fi
  PYTHON_BIN="$(command -v python3)"
  if ! need_cmd pip3; then
    warn "pip not found. Attempting to bootstrap ensurepip."
    if ! "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1; then
      suggest_pacman python-pip
      exit 1
    fi
  fi

  # Prepare directories
  mkdir -p "$PREFIX" "$CONFIG_DIR"

  # Create venv or select pip target
  VENV_DIR="$VENV_DIR_DEFAULT"
  if [ "$USE_VENV" = 1 ]; then
    VENV_DIR="$PREFIX/venv"
    if [ ! -d "$VENV_DIR" ]; then
      info "Creating virtual environment at $VENV_DIR"
      "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
      info "Using existing virtual environment at $VENV_DIR"
    fi
    VENV_BIN="$VENV_DIR/bin"
    PIP="$VENV_BIN/pip"
    PY="$VENV_BIN/python"
  else
    info "Using user-wide pip installation (no venv)"
    VENV_BIN=""
    PIP="$(command -v pip3)"
    PY="$PYTHON_BIN"
  fi

  # Upgrade packaging tools
  info "Upgrading pip, setuptools, wheel"
  "$PY" -m pip install --upgrade pip setuptools wheel

  # Install OpenAgent from repo
  info "Installing OpenAgent from $REPO_DIR"
  if [ -f "$REPO_DIR/pyproject.toml" ] || [ -f "$REPO_DIR/setup.py" ]; then
    (cd "$REPO_DIR" && "$PIP" install --upgrade .)
  else
    warn "Skipping pip install (project metadata missing)."
  fi

  # Ensure CLI is invokable
  if [ -n "$VENV_BIN" ]; then
    if [ ! -x "$VENV_BIN/openagent" ]; then
      warn "openagent entrypoint not found in venv. Checking module entry..."
      "$PY" - <<'PY'
import importlib, sys
try:
    importlib.import_module('openagent')
    sys.exit(0)
except Exception as e:
    print(f"MODULE_IMPORT_ERROR: {e}")
    sys.exit(1)
PY
    fi
  fi

  # Install completion
  if [ "$INSTALL_COMPLETION" = 1 ]; then
    info "Installing zsh completion to $COMPLETION_FILE"
    install_completion_file
  else
    info "Skipping completion installation. Completion snippet (zsh):"
    say "\n# Add to your fpath in .zshrc\nfpath=(\$HOME/.config/openagent \$fpath)\nautoload -U compinit && compinit -i\n"
  fi

  # Zsh integration
  if [ "$DO_ZSH_INTEGRATION" = 1 ]; then
    info "Appending zsh integration snippet to $ZSHRC_FILE"
    if [ -f "$ZSHRC_FILE" ] && grep -q "OpenAgent integration (BEGIN)" "$ZSHRC_FILE"; then
      info "Integration snippet already present in .zshrc; skipping append."
    else
      {
        printf '\n'
        print_zsh_snippet
      } >> "$ZSHRC_FILE"
      info "Appended integration. Reload your shell or run: source \"$ZSHRC_FILE\""
    fi
  else
    info "Printing zsh integration snippet (not modifying .zshrc):"
    print_zsh_snippet
  fi

  # Keys handling removed in local-only build

  # Create default policy/config files if missing
  if [ ! -f "$CONFIG_DIR/policy.yaml" ]; then
    info "Creating default policy.yaml"
    cat >"$CONFIG_DIR/policy.yaml" <<'EOF'
# Default OpenAgent policy
default_decision: warn
block_risky: true
allowlist: {}
EOF
  fi

  # Systemd services
  if [ "$CREATE_SERVICES" = 1 ]; then
    info "Creating systemd user services under $SYSTEMD_USER_DIR"
    create_systemd_units
    systemctl --user daemon-reload || warn "systemctl user daemon-reload failed (is user linger enabled?)"
    if [ "$ENABLE_DAEMON" = 1 ]; then
      info "Enabling and starting openagent-daemon.service"
      systemctl --user enable --now openagent-daemon.service || warn "Failed to enable/start daemon"
    else
      info "Daemon service created but not enabled. Start with: systemctl --user start openagent-daemon.service"
    fi
    if [ "$ENABLE_SERVER" = 1 ]; then
      info "Enabling and starting openagent-server.service"
      systemctl --user enable --now openagent-server.service || warn "Failed to enable/start server"
    else
      info "Server service created but not enabled. Start with: systemctl --user start openagent-server.service"
    fi
  else
    info "Skipping systemd service creation"
  fi

  say "\nInstallation complete!"
  say "\nNext steps:"
  say "  1) Start a new shell or run: source '$ZSHRC_FILE'"
  say "  2) Verify: openagent doctor"
  say "  3) Start chatting: openagent chat --model auto --provider auto"
  say "\nServices:"
  say "  • Start daemon:  systemctl --user start openagent-daemon.service"
  say "  • Start server:  systemctl --user start openagent-server.service"
  say "\nUninstall (manual):"
  say "  • Remove venv:   rm -rf '$PREFIX'"
  say "  • Remove config: rm -rf '$CONFIG_DIR'"
  say "  • Disable svc:   systemctl --user disable --now openagent-daemon.service openagent-server.service && rm -f '$SYSTEMD_USER_DIR'/openagent-*.service && systemctl --user daemon-reload"
}

main "$@"

