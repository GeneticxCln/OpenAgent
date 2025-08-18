SHELL := /bin/bash
VENV_DIR := venv
PY := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

.PHONY: dev venv editable shell-integration policy path help test lint fmt type openapi export-openapi

help:
	@echo "Available targets:"
	@echo "  make dev                # Create venv, install editable, apply shell integration, enable policy"
	@echo "  make venv               # Create/upgrade virtual environment"
	@echo "  make editable           # pip install -e . (editable install)"
	@echo "  make shell-integration  # Apply zsh shell integration snippet"
	@echo "  make policy             # Enable block_risky policy"
	@echo "  make path               # Add venv/bin to PATH in ~/.zshrc if missing"
	@echo "  make test               # Run tests"
	@echo "  make lint               # Run flake8"
	@echo "  make fmt                # Run black and isort"
	@echo "  make type               # Run mypy"
	@echo "  make openapi            # Export OpenAPI schema to openapi.json"
	@echo "  make export-openapi     # Export OpenAPI schema (OUT=/tmp/openapi.json PORT=8050)"

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
	@"$(PY)" -c "from openagent.terminal.integration import install_snippet; p,_=install_snippet('zsh', apply=True); print(f'Installed shell integration to {p}')"
	@bash -lc "grep -q '^export OPENAGENT_EXPLAIN=' $$HOME/.zshrc && sed -i 's/^export OPENAGENT_EXPLAIN=.*/export OPENAGENT_EXPLAIN=1/' $$HOME/.zshrc || echo 'export OPENAGENT_EXPLAIN=1' >> $$HOME/.zshrc"
	@bash -lc "grep -q '^export OPENAGENT_WARN=' $$HOME/.zshrc && sed -i 's/^export OPENAGENT_WARN=.*/export OPENAGENT_WARN=1/' $$HOME/.zshrc || echo 'export OPENAGENT_WARN=1' >> $$HOME/.zshrc"
	@echo "Shell integration applied."

## Enable hard-blocking of risky commands in policy
policy:
	@"$(PY)" -c "from openagent.terminal.validator import load_policy, save_policy; p=load_policy(); p['block_risky']=True; save_policy(p); print('Policy updated: block_risky=True')"

## Add venv/bin to PATH in ~/.zshrc if missing
path:
	@bash -lc 'grep -q "^export PATH=\"$${HOME}/OpenAgent/venv/bin:\$${PATH}\"$$" $$HOME/.zshrc || echo "export PATH=\"$${HOME}/OpenAgent/venv/bin:\$${PATH}\"" >> $$HOME/.zshrc; echo "PATH updated in ~/.zshrc"'

## Developer shortcuts
TEST_ARGS ?=

test:
	pytest -q $(TEST_ARGS)

lint:
	flake8 openagent tests

fmt:
	black openagent tests
	isort openagent tests

type:
	mypy openagent

PORT ?= 8042
OUT ?= openapi.json

openapi:
	./scripts/export_openapi.sh $(PORT) $(OUT)

export-openapi: openapi

