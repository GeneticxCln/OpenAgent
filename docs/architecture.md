# Architecture

## Overview

OpenAgent is built with a modular architecture consisting of several core components:

- **Agent Core**: Central orchestration and decision-making
- **LLM Interface**: Abstraction layer for different LLM providers  
- **Context Engine**: Intelligent context gathering and management
- **Command Intelligence**: Smart command understanding and execution
- **CLI Interface**: User-facing command line interface

## Core Components

### Agent

The `Agent` class is the central orchestrator that coordinates between different components to provide intelligent assistance.

### LLM Interface

Supports multiple LLM providers:
- OpenAI GPT models
- Anthropic Claude
- Local models via Ollama

### Context Engine

Gathers relevant context from:
- Current directory and files
- Git repository state
- System information
- Command history

### Command Intelligence

Provides:
- Command suggestion and completion
- Error diagnosis and fixes
- Code generation and debugging

## Data Flow

1. User input is received via CLI
2. Context Engine gathers relevant information
3. Agent processes input with context
4. LLM generates response
5. Response is presented to user
