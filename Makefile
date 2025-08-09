SHELL := /bin/bash
VENV_DIR := venv
PY := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

.PHONY: dev venv editable shell-integration policy path help

help:
	@echo "Available targets:"
	@echo "  make dev                # Create venv, install editable, apply shell integration, enable policy"
	@echo "  make venv               # Create/upgrade virtual environment"
	@echo "  make editable           # pip install -e . (editable install)"
	@echo "  make shell-integration  # Apply zsh shell integration snippet"
	@echo "  make policy             # Enable block_risky policy"
	@echo "  make path               # Add venv/bin to PATH in ~/.zshrc if missing"

## Full development setup
dev: venv editable shell-integration policy path
	@echo
	@echo "✅ Development environment ready."
	@echo "-> Restart your terminal or run: source $$HOME/.zshrc"
	@echo "-> Try: openagent models"

## Create virtual environment if missing and upgrade base tooling
venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		python -m venv "$(VENV_DIR)"; \
	fi
	@"$(PY)" -m pip install --upgrade pip setuptools wheel

## Install project in editable mode with hatchling backend
editable:
	@"$(PIP)" install -e .

## Apply zsh integration snippet and enable helpful env vars
shell-integration:
	@"$(PY)" - <<'PY'
from openagent.terminal.integration import install_snippet
from pathlib import Path
rc_path, _ = install_snippet('zsh', apply=True)
# Ensure helpful env vars are present
zshrc = Path.home()/'.zshrc'
lines = zshrc.read_text().splitlines() if zshrc.exists() else []
def ensure(line):
    for i,s in enumerate(lines):
        if s.strip().startswith(line.split('=')[0]):
            lines[i] = line
            return
    lines.append(line)
ensure('export OPENAGENT_EXPLAIN=1')
ensure('export OPENAGENT_WARN=1')
zshrc.write_text('\n'.join(lines) + ('\n' if not lines or not lines[-1].endswith('\n') else ''))
print(f"Installed shell integration to {rc_path}")
PY
	@echo "Shell integration applied."

## Enable hard-blocking of risky commands in policy
policy:
	@"$(PY)" - <<'PY'
from openagent.terminal.validator import load_policy, save_policy
p = load_policy()
p['block_risky'] = True
save_policy(p)
print('Policy updated: block_risky=True')
PY

## Add venv/bin to PATH in ~/.zshrc if missing
path:
	@bash -lc 'grep -q "^export PATH=\"$${HOME}/OpenAgent/venv/bin:\$${PATH}\"$$" $$HOME/.zshrc || echo "export PATH=\"$${HOME}/OpenAgent/venv/bin:\$${PATH}\"" >> $$HOME/.zshrc; echo "PATH updated in ~/.zshrc"'

