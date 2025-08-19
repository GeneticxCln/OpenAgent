# Configuration

## Configuration File

OpenAgent uses a configuration file located at `~/.openagent/config.yaml`.

## LLM Providers

### OpenAI

```yaml
llm:
  provider: "openai"
  model: "gpt-4"
  api_key: "your-api-key"
```

### Anthropic Claude

```yaml
llm:
  provider: "anthropic"
  model: "claude-3-sonnet"
  api_key: "your-api-key"
```

### Local Models

```yaml
llm:
  provider: "ollama"
  model: "llama2"
  base_url: "http://localhost:11434"
```

## General Settings

```yaml
general:
  debug: false
  verbose: true
  history_limit: 1000
```

## Context Settings

```yaml
context:
  max_tokens: 4096
  include_system_info: true
  include_git_info: true
```
