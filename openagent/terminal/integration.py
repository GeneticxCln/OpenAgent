from pathlib import Path
from typing import Literal


def zsh_integration_snippet() -> str:
    """
    Return a zsh snippet that integrates OpenAgent with the interactive shell.

    Features:
    - Pre-exec explanation: if OPENAGENT_EXPLAIN=1, prints a short explanation
      of the command before it runs, using `openagent explain`.
    - Risky command warning: if OPENAGENT_WARN=1, prints a warning for obviously
      dangerous patterns (rm -rf /, mkfs, fdisk, etc.).
    - Confirmation mode: if OPENAGENT_CONFIRM=1, prompt to proceed (shows a
      brief explanation) before running the command. Default is No.
    - Block risky: if OPENAGENT_BLOCK_RISKY=1, block execution of risky commands.
    - Optional keybinding: Ctrl-g to explain the current buffer (if ZLE is enabled).
    """
    return r'''
# --- OpenAgent zsh integration start ---
# Enable or disable features via environment variables in your shell rc file:
#   export OPENAGENT_EXPLAIN=1       # Explain commands before running (non-blocking)
#   export OPENAGENT_WARN=1          # Warn on risky commands
#   export OPENAGENT_CONFIRM=1       # Ask for confirmation before running (shows summary)
#   export OPENAGENT_BLOCK_RISKY=1   # Block risky commands outright
#
# Pre-execution hook
function preexec() {
  local cmd="$1"

  # Consult OpenAgent validator for decision
  local decision
  decision=$(command openagent validate "$cmd" --quiet 2>/dev/null)
  if [[ -z "$decision" ]]; then
    decision="warn"
  fi

  if [[ "$decision" == "block" ]]; then
    echo "[OpenAgent] Blocked by policy: $cmd" 1>&2
    false
    fc -p
    zle && zle push-input
    return 130
  fi
  if [[ "$decision" == "warn" ]]; then
    echo "[OpenAgent] Warning (policy): $cmd" 1>&2
  fi
    risky=1
  fi

  # Block risky outright if requested
  if [[ -n "$OPENAGENT_BLOCK_RISKY" && $risky -eq 1 ]]; then
    echo "[OpenAgent] Blocked risky command: $cmd" >&2
    # abort the command by replacing it with a no-op and killing the current line
    false
    fc -p
    zle && zle push-input
    return 130
  fi

  # Warn on risky commands
  if [[ -n "$OPENAGENT_WARN" && $risky -eq 1 ]]; then
    echo "[OpenAgent] Warning: command may be dangerous: $cmd" >&2
  fi

  # Explain command in background (non-blocking)
  if [[ -n "$OPENAGENT_EXPLAIN" ]]; then
    (
      command openagent explain "$cmd" 2>/dev/null | sed -n '1,12p' | sed 's/^/[OpenAgent] /'
    ) &
  fi

  # Confirmation mode (blocks until user decides)
  if [[ -n "$OPENAGENT_CONFIRM" ]]; then
    local summary
    summary=$(command openagent explain "$cmd" 2>/dev/null | sed -n '1,8p' | sed 's/^/[OpenAgent] /')
    if [[ -n "$summary" ]]; then
      echo "$summary"
    fi
    printf "[OpenAgent] Proceed? [y/N] " >&2
    # read a single keypress (y/Y to proceed)
    local key
    stty -g > /tmp/.openagent_stty
    stty -icanon -echo min 1 time 0
    key=$(dd bs=1 count=1 2>/dev/null)
    stty $(cat /tmp/.openagent_stty) 2>/dev/null
    echo "" >&2
    if [[ "$key" != "y" && "$key" != "Y" ]]; then
      echo "[OpenAgent] Command cancelled" >&2
      false
      fc -p
      zle && zle push-input
      return 130
    fi
  fi
}

# Explain current ZLE buffer with Ctrl-G (if interactive line editor is active)
if [[ -n "$ZSH_VERSION" && -o interactive && $+functions[zle] -gt 0 ]]; then
  function _openagent_explain_buffer() {
    local cmd=${BUFFER}
    if [[ -z "$cmd" ]]; then
      zle -M "OpenAgent: buffer is empty"
      return 0
    fi
    print -l "[OpenAgent] Explaining: $cmd"
    command openagent explain "$cmd" 2>/dev/null | sed -n '1,20p' | sed 's/^/[OpenAgent] /'
    zle redisplay
  }
  zle -N _openagent_explain_buffer
  bindkey '^G' _openagent_explain_buffer
fi
# --- OpenAgent zsh integration end ---
'''


def bash_integration_snippet() -> str:
    """Return a bash snippet to integrate OpenAgent using the DEBUG trap.

    Note: Bash integration is more limited than zsh; we provide warn/explain
    and confirmation modes. Risky blocking is simulated by returning non-zero.
    """
    return r'''
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
'''


def install_snippet(shell: Literal["zsh", "bash"], apply: bool = False) -> tuple[Path, str]:
    """Return (path, snippet). If apply=True, append to shell rc file."""
    if shell == "zsh":
        snippet = zsh_integration_snippet()
        rc_path = Path.home() / ".zshrc"
        marker_start = "# --- OpenAgent zsh integration start ---"
        marker_end = "# --- OpenAgent zsh integration end ---"
    elif shell == "bash":
        snippet = bash_integration_snippet()
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
            new_text = text + ("\n\n" if text and not text.endswith("\n") else "") + snippet
        rc_path.write_text(new_text)
    return rc_path, snippet
