from pathlib import Path
from typing import Literal


def zsh_integration_snippet() -> str:
    """
    Return a zsh snippet that integrates OpenAgent with the interactive shell.

    Features (Warp-like behavior):
    - Policy-first: validate command (allow|warn|block). Blocks are aborted inline.
    - Inline explain: brief, non-blocking summary; uses timeout if available.
    - Confirmation: prompt to approve risky/warned commands with a single key (y).
    - Env toggles: OPENAGENT_EXPLAIN=1, OPENAGENT_WARN=1, OPENAGENT_CONFIRM=1.
    """
    return r'''
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

  # 1) Policy decision (quiet)
  local decision
  decision=$(command openagent validate "$cmd" --quiet 2>/dev/null)
  [[ -z "$decision" ]] && decision="warn"

  # 2) BLOCK immediately
  if [[ "$decision" == "block" ]]; then
    echo "[OpenAgent] BLOCK: $cmd" 1>&2
    # Abort current line
    false; fc -p; zle && zle push-input; return 130
  fi

  # 3) WARN inline if requested
  if [[ -n "$OPENAGENT_WARN" && "$decision" == "warn" ]]; then
    echo "[OpenAgent] WARNING: $cmd" 1>&2
  fi

  # 4) EXPLAIN briefly (non-blocking) if requested
  if [[ -n "$OPENAGENT_EXPLAIN" ]]; then
    (
      _openagent_run_with_timeout command openagent explain "$cmd" 2>/dev/null \
        | sed -n '1,8p' | sed 's/^/[OpenAgent] /'
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
      false; fc -p; zle && zle push-input; return 130
    fi
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
