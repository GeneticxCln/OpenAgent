import os
from pathlib import Path
from typing import Literal, Optional

try:
    from openagent.core.command_intelligence import (
        CommandCompletionEngine,
        CompletionContext,
        create_command_completion_engine
    )
    from openagent.core.command_templates import CommandTemplates, create_command_templates
    from openagent.core.context_v2.project_analyzer import ProjectContextEngine
    from openagent.core.context_v2.history_intelligence import HistoryIntelligence
    INTELLIGENCE_AVAILABLE = True
except ImportError:
    INTELLIGENCE_AVAILABLE = False


def zsh_integration_snippet() -> str:
    """
    Return a zsh snippet that integrates OpenAgent with the interactive shell.

    Features (Warp-like behavior):
    - Policy-first: validate command (allow|warn|block). Blocks are aborted inline.
    - Inline explain: brief, non-blocking summary; uses timeout if available.
    - Confirmation: prompt to approve risky/warned commands with a single key (y).
    - Env toggles: OPENAGENT_EXPLAIN=1, OPENAGENT_WARN=1, OPENAGENT_CONFIRM=1.
    """
    return r"""
# --- OpenAgent zsh integration start ---
# Enable features in your ~/.zshrc:
#   export OPENAGENT_EXPLAIN=1       # Show short explanation
#   export OPENAGENT_WARN=1          # Print warnings
#   export OPENAGENT_CONFIRM=1       # Ask approval (y/N) for non-allow decisions
#
# Helper: run with 1.5s timeout if GNU timeout exists
_openagent_run_with_timeout() {
  if command -v timeout >/dev/null 2>&1; then
    timeout 1.5 "$@"
  else
    "$@"
  fi
}

# Pre-execution hook (runs just before executing the typed command)
function preexec() {
  local cmd="$1"
  [[ -z "$cmd" ]] && return 0
  local ts cwd logf
  ts=$(date +%s)
  cwd="$PWD"
  logf="$HOME/.config/openagent/history.jsonl"
  mkdir -p "$HOME/.config/openagent" 2>/dev/null || true

  # 1) Policy decision (quiet with reason)
  local decision_pair decision reason
  decision_pair=$(command openagent validate "$cmd" --quiet-with-reason 2>/dev/null)
  decision=${decision_pair%%|*}
  reason=${decision_pair#*|}
  [[ -z "$decision" ]] && decision="warn"

  # 2) BLOCK immediately
  if [[ "$decision" == "block" ]]; then
    echo "[OpenAgent] BLOCK: $cmd" 1>&2
    # Log block
    printf '{"ts":%s,"cwd":"%s","cmd":%s,"decision":"block","reason":%s,"approved":false}\n' \
      "$ts" "$cwd" $(printf %q "$cmd" | sed 's/^/"/;s/$/"/') $(printf %q "$reason" | sed 's/^/"/;s/$/"/') >> "$logf" 2>/dev/null || true
    # Abort current line
    false; fc -p; zle && zle push-input; return 130
  fi

  # 3) WARN inline if requested
  if [[ -n "$OPENAGENT_WARN" && "$decision" == "warn" ]]; then
    if [[ -n "$reason" ]]; then
      echo "[OpenAgent] WARNING: $reason" 1>&2
    else
      echo "[OpenAgent] WARNING: $cmd" 1>&2
    fi
  fi

  # 4) EXPLAIN briefly (non-blocking) if requested
  if [[ -n "$OPENAGENT_EXPLAIN" ]]; then
    (
      _openagent_run_with_timeout command openagent explain "$cmd" 2>/dev/null \
        | sed -n '1,2p' | sed 's/^/[OpenAgent] /'
    ) &
  fi

  # 5) CONFIRM for non-allow decisions (or if explicitly enabled)
  if [[ -n "$OPENAGENT_CONFIRM" || "$decision" == "warn" ]]; then
    printf "[OpenAgent] Proceed? [y/N] " 1>&2
    local key
    stty -g > /tmp/.openagent_stty
    stty -icanon -echo min 1 time 0
    key=$(dd bs=1 count=1 2>/dev/null)
    stty $(cat /tmp/.openagent_stty) 2>/dev/null
    echo "" 1>&2
    if [[ "$key" != "y" && "$key" != "Y" ]]; then
      echo "[OpenAgent] Cancelled" 1>&2
      # Log cancelled
      printf '{"ts":%s,"cwd":"%s","cmd":%s,"decision":"%s","reason":%s,"approved":false}\n' \
        "$ts" "$cwd" $(printf %q "$cmd" | sed 's/^/"/;s/$/"/') "$decision" $(printf %q "$reason" | sed 's/^/"/;s/$/"/') >> "$logf" 2>/dev/null || true
      false; fc -p; zle && zle push-input; return 130
    else
      # Log approved
      printf '{"ts":%s,"cwd":"%s","cmd":%s,"decision":"%s","reason":%s,"approved":true}\n' \
        "$ts" "$cwd" $(printf %q "$cmd" | sed 's/^/"/;s/$/"/') "$decision" $(printf %q "$reason" | sed 's/^/"/;s/$/"/') >> "$logf" 2>/dev/null || true
    fi
  else
    # Auto-allow: log approved=true
    printf '{"ts":%s,"cwd":"%s","cmd":%s,"decision":"allow","reason":"policy default","approved":true}\n' \
      "$ts" "$cwd" $(printf %q "$cmd" | sed 's/^/"/;s/$/"/') >> "$logf" 2>/dev/null || true
  fi
}

# Explain current ZLE buffer with Ctrl-G (interactive editor)
if [[ -n "$ZSH_VERSION" && -o interactive && $+functions[zle] -gt 0 ]]; then
  function _openagent_explain_buffer() {
    local cmd=${BUFFER}
    if [[ -z "$cmd" ]]; then
      zle -M "OpenAgent: buffer is empty"; return 0
    fi
    print -l "[OpenAgent] Explaining: $cmd"
    _openagent_run_with_timeout command openagent explain "$cmd" 2>/dev/null \
      | sed -n '1,20p' | sed 's/^/[OpenAgent] /'
    zle redisplay
  }
  zle -N _openagent_explain_buffer
  bindkey '^G' _openagent_explain_buffer
fi
# --- OpenAgent zsh integration end ---
"""


def bash_integration_snippet() -> str:
    """Return a bash snippet to integrate OpenAgent using the DEBUG trap.

    Note: Bash integration is more limited than zsh; we provide warn/explain
    and confirmation modes. Risky blocking is simulated by returning non-zero.
    """
    return r"""
# --- OpenAgent bash integration start ---
# export OPENAGENT_EXPLAIN=1
# export OPENAGENT_WARN=1
# export OPENAGENT_CONFIRM=1
# export OPENAGENT_BLOCK_RISKY=1

_openagent_is_interactive() {
  [[ $- == *i* ]]
}

_openagent_is_risky() {
  local cmd="$1"
  echo " $cmd " | grep -Eqi " (rm |rmdir |mkfs|fdisk|dd |chmod 777|chown |killall|: \(\)\{:\|:&\};:) "
}

_openagent_preexec() {
  [[ -z "$BASH_COMMAND" ]] && return 0
  local cmd="$BASH_COMMAND"

  # Consult validator
  local decision
  decision=$(command openagent validate "$cmd" --quiet 2>/dev/null)
  if [[ "$decision" == "block" ]]; then
    echo "[OpenAgent] Blocked by policy: $cmd" 1>&2
    return 130
  fi
  if [[ "$decision" == "warn" ]]; then
    echo "[OpenAgent] Warning (policy): $cmd" 1>&2
  fi
  if [[ -n "$OPENAGENT_EXPLAIN" ]]; then
    ( command openagent explain "$cmd" 2>/dev/null | sed -n '1,12p' | sed 's/^/[OpenAgent] /' ) &
  fi
  if [[ -n "$OPENAGENT_CONFIRM" ]]; then
    local summary
    summary=$(command openagent explain "$cmd" 2>/dev/null | sed -n '1,8p' | sed 's/^/[OpenAgent] /')
    if [[ -n "$summary" ]]; then
      echo "$summary"
    fi
    read -rp "[OpenAgent] Proceed? [y/N] " ans
    if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
      echo "[OpenAgent] Command cancelled" >&2
      # Replace command with ':' builtin (no-op) by exporting PROMPT_COMMAND side-effect
      BASH_COMMAND=":"
      return 130
    fi
  fi
}

# Install DEBUG trap once
if _openagent_is_interactive; then
  case "$PROMPT_COMMAND" in
    *"_openagent_preexec"*) : ;;
    *) PROMPT_COMMAND="_openagent_preexec; ${PROMPT_COMMAND}" ;;
  esac
fi
# --- OpenAgent bash integration end ---
"""


def enhanced_zsh_integration_snippet() -> str:
    """
    Return an enhanced zsh snippet with command completion and intelligence features.
    
    Features:
    - Command completion with Tab
    - Smart suggestions with Ctrl-S
    - Template recommendations with Ctrl-T
    - All existing policy and explanation features
    """
    base_snippet = zsh_integration_snippet()
    
    enhanced_features = r"""
# Enhanced OpenAgent features (command completion and intelligence)
if [[ -n "$ZSH_VERSION" && -o interactive && $+functions[zle] -gt 0 ]]; then
  
  # Command completion with Tab (enhanced)
  function _openagent_complete() {
    local cmd=${BUFFER}
    local suggestions
    suggestions=$(command openagent complete "$cmd" --max=10 2>/dev/null)
    if [[ -n "$suggestions" ]]; then
      # Parse suggestions and show them
      local IFS=$'\n'
      local -a completion_list
      completion_list=(${=suggestions})
      if [[ ${#completion_list[@]} -gt 0 ]]; then
        zle -M "Suggestions: ${completion_list[1]}"
        # Replace buffer with first suggestion
        BUFFER="${completion_list[1]}"
        CURSOR=${#BUFFER}
      fi
    fi
    zle redisplay
  }
  zle -N _openagent_complete
  bindkey '^I' _openagent_complete  # Tab
  
  # Smart suggestions with Ctrl-S
  function _openagent_smart_suggest() {
    local cmd=${BUFFER}
    if [[ -z "$cmd" ]]; then
      # Show context-aware suggestions for empty buffer
      local context_suggestions
      context_suggestions=$(command openagent suggest-context 2>/dev/null)
      if [[ -n "$context_suggestions" ]]; then
        zle -M "Context suggestions: $context_suggestions"
      else
        zle -M "OpenAgent: Type a command for smart suggestions"
      fi
    else
      # Show suggestions for partial command
      local suggestions
      suggestions=$(command openagent suggest "$cmd" --intelligent 2>/dev/null)
      if [[ -n "$suggestions" ]]; then
        zle -M "Smart suggestions: $suggestions"
      fi
    fi
    zle redisplay
  }
  zle -N _openagent_smart_suggest
  bindkey '^S' _openagent_smart_suggest
  
  # Template recommendations with Ctrl-T
  function _openagent_templates() {
    local templates
    templates=$(command openagent templates --suggest --current-dir="$PWD" 2>/dev/null)
    if [[ -n "$templates" ]]; then
      zle -M "Available templates: $templates"
      # Could implement template selection here
    else
      zle -M "OpenAgent: No templates available for current context"
    fi
    zle redisplay
  }
  zle -N _openagent_templates
  bindkey '^T' _openagent_templates
  
  # Auto-correction on Enter (if enabled)
  if [[ -n "$OPENAGENT_AUTO_CORRECT" ]]; then
    function _openagent_auto_correct() {
      local cmd=${BUFFER}
      if [[ -n "$cmd" ]]; then
        local corrected
        corrected=$(command openagent correct "$cmd" 2>/dev/null)
        if [[ -n "$corrected" && "$corrected" != "$cmd" ]]; then
          zle -M "Auto-corrected: $cmd -> $corrected"
          BUFFER="$corrected"
          CURSOR=${#BUFFER}
        fi
      fi
      zle accept-line
    }
    zle -N _openagent_auto_correct
    bindkey '^M' _openagent_auto_correct  # Enter
  fi
  
fi
"""
    
    # Insert enhanced features before the end marker
    return base_snippet.replace(
        "# --- OpenAgent zsh integration end ---",
        enhanced_features + "# --- OpenAgent zsh integration end ---"
    )


def enhanced_bash_integration_snippet() -> str:
    """
    Return an enhanced bash snippet with basic command intelligence features.
    
    Note: Bash has more limited capabilities than zsh for interactive features.
    """
    base_snippet = bash_integration_snippet()
    
    enhanced_features = r"""
# Enhanced OpenAgent features for bash (limited)
if _openagent_is_interactive; then
  
  # Command completion function (can be bound to keys)
  _openagent_bash_complete() {
    local cmd="$1"
    local suggestions
    suggestions=$(command openagent complete "$cmd" --max=5 2>/dev/null)
    if [[ -n "$suggestions" ]]; then
      echo "[OpenAgent] Suggestions: $suggestions" >&2
    fi
  }
  
  # Template suggestions function
  _openagent_bash_templates() {
    local templates
    templates=$(command openagent templates --suggest --current-dir="$PWD" 2>/dev/null)
    if [[ -n "$templates" ]]; then
      echo "[OpenAgent] Templates: $templates" >&2
    fi
  }
  
  # Add to PROMPT_COMMAND for basic intelligence
  if [[ -n "$OPENAGENT_SMART_PROMPT" ]]; then
    _openagent_smart_prompt() {
      # Show context info in prompt (very basic)
      local context_info
      context_info=$(command openagent context-info --brief 2>/dev/null)
      if [[ -n "$context_info" ]]; then
        PS1="[OA: $context_info] $PS1"
      fi
    }
    PROMPT_COMMAND="_openagent_smart_prompt; $PROMPT_COMMAND"
  fi
  
fi
"""
    
    return base_snippet.replace(
        "# --- OpenAgent bash integration end ---",
        enhanced_features + "# --- OpenAgent bash integration end ---"
    )


def install_snippet(
    shell: Literal["zsh", "bash"], apply: bool = False, enhanced: bool = False
) -> tuple[Path, str]:
    """Return (path, snippet). If apply=True, append to shell rc file."""
    if shell == "zsh":
        snippet = enhanced_zsh_integration_snippet() if enhanced else zsh_integration_snippet()
        rc_path = Path.home() / ".zshrc"
        marker_start = "# --- OpenAgent zsh integration start ---"
        marker_end = "# --- OpenAgent zsh integration end ---"
    elif shell == "bash":
        snippet = enhanced_bash_integration_snippet() if enhanced else bash_integration_snippet()
        rc_path = Path.home() / ".bashrc"
        marker_start = "# --- OpenAgent bash integration start ---"
        marker_end = "# --- OpenAgent bash integration end ---"
    else:
        raise ValueError("Supported shells: zsh, bash")

    if apply:
        text = rc_path.read_text() if rc_path.exists() else ""
        if marker_start in text and marker_end in text:
            # already installed; replace existing block
            import re

            new_text = re.sub(
                rf"{marker_start}[\s\S]*?{marker_end}",
                snippet.strip(),
                text,
                count=1,
            )
        else:
            new_text = (
                text + ("\n\n" if text and not text.endswith("\n") else "") + snippet
            )
        rc_path.write_text(new_text)
    return rc_path, snippet


def create_completion_context() -> Optional[CompletionContext]:
    """
    Create a completion context for the current terminal session.
    
    Returns:
        CompletionContext if intelligence systems are available, None otherwise
    """
    if not INTELLIGENCE_AVAILABLE:
        return None
    
    from openagent.core.context_v2.project_analyzer import ProjectType
    
    try:
        current_dir = Path.cwd()
        
        # Detect project type
        project_engine = ProjectContextEngine()
        workspace = project_engine.analyze_workspace(current_dir)
        
        # Get environment variables
        env_vars = dict(os.environ)
        
        # Create completion context
        context = CompletionContext(
            current_directory=current_dir,
            project_type=workspace.project_type if workspace else None,
            git_repo=workspace.git_context.is_repo if workspace else False,
            git_branch=workspace.git_context.current_branch if workspace else None,
            recent_commands=[],  # Would need to load from history
            environment_vars=env_vars
        )
        
        return context
    except Exception:
        return None


def get_command_suggestions(partial_command: str, max_suggestions: int = 10) -> list[str]:
    """
    Get command suggestions for a partial command.
    
    Args:
        partial_command: Partial command input
        max_suggestions: Maximum number of suggestions to return
        
    Returns:
        List of command suggestions
    """
    if not INTELLIGENCE_AVAILABLE:
        return []
    
    try:
        context = create_completion_context()
        if not context:
            return []
        
        completion_engine = create_command_completion_engine()
        suggestions = completion_engine.suggest_commands(partial_command, context, max_suggestions)
        
        return [s.text for s in suggestions]
    except Exception:
        return []


def get_template_suggestions(current_dir: Optional[Path] = None) -> list[str]:
    """
    Get template suggestions for the current context.
    
    Args:
        current_dir: Current directory (defaults to cwd)
        
    Returns:
        List of template names
    """
    if not INTELLIGENCE_AVAILABLE:
        return []
    
    try:
        if current_dir is None:
            current_dir = Path.cwd()
        
        # Get workspace context
        project_engine = ProjectContextEngine()
        workspace = project_engine.analyze_workspace(current_dir)
        
        if not workspace:
            return []
        
        # Get template suggestions
        templates = create_command_templates()
        suggestions = templates.suggest_templates(workspace)
        
        return [t.name for t in suggestions]
    except Exception:
        return []


def auto_correct_command(command: str) -> Optional[str]:
    """
    Auto-correct a command if possible.
    
    Args:
        command: Command to correct
        
    Returns:
        Corrected command if correction found, None otherwise
    """
    if not INTELLIGENCE_AVAILABLE:
        return None
    
    try:
        completion_engine = create_command_completion_engine()
        return completion_engine.auto_correct_command(command)
    except Exception:
        return None
