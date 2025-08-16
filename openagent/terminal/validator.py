import shlex
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_POLICY = {
    "risky_patterns": [
        "rm -rf", "rm -r", "mkfs", "fdisk", "dd ", "chmod 777", "chmod -R 777", "chown ", "killall", "sudo rm", "rm /", "rm -rf /"
    ],
    "allowlist": {
        # command: allowed flags (prefix match), empty list means no restriction on flags
        "ls": ["-l", "-a", "-h", "-la", "--all", "--human-readable"],
        "grep": ["-i", "-r", "-n", "-E", "--color", "--include", "--exclude"],
        "find": ["-name", "-type", "-maxdepth", "-mindepth", "-mtime", "-size", "-exec"],
        "cat": [],
        "echo": [],
        "pwd": [],
        "whoami": [],
        "date": [],
        "env": [],
        "python": [],
        "pip": ["install", "list", "show"],
        "git": ["status", "log", "diff", "show", "add", "commit", "push", "pull", "fetch", "checkout", "switch", "rebase"],
        "docker": ["ps", "images", "pull", "run", "logs", "exec", "stop", "start", "compose"],
        "docker-compose": ["up", "down", "logs", "ps", "pull", "build"],
        "pacman": ["-S", "-Ss", "-Qi", "-Qs", "-Sy", "-Syu"],
        "paru": ["-S", "-Ss", "-Qi", "-Qs", "-Sy", "-Syu"],
        "yay": ["-S", "-Ss", "-Qi", "-Qs", "-Sy", "-Syu"],
        "systemctl": ["status", "start", "stop", "restart", "enable", "disable", "--user"],
        "journalctl": ["-u", "-xe", "-f"],
        "npm": ["install", "ci", "run", "start", "test", "build"],
        "yarn": ["install", "run", "start", "test", "build"],
        "pnpm": ["install", "run", "start", "test", "build"],
        "pipx": ["install", "list", "upgrade", "upgrade-all"],
        "pip": ["install", "list", "show", "freeze"],
        "node": ["--version"],
        "python": ["-m", "--version"],
        "curl": ["-I", "-s", "-L", "--head", "--silent"],
        "make": ["build", "test", "install"],
    },
    "default_decision": "block"  # allow | warn | block
}

CONFIG_PATH = Path.home() / ".config" / "openagent" / "policy.yaml"


def load_policy() -> Dict:
    if CONFIG_PATH.exists():
        try:
            return yaml.safe_load(CONFIG_PATH.read_text()) or {}
        except Exception:
            return DEFAULT_POLICY.copy()
    return DEFAULT_POLICY.copy()


def save_policy(policy: Dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.safe_dump(policy, sort_keys=False))


def parse_command(cmd: str) -> Tuple[str, List[str]]:
    try:
        parts = shlex.split(cmd)
    except Exception:
        parts = cmd.strip().split()
    if not parts:
        return "", []
    return parts[0], parts[1:]


def is_risky(cmd: str, policy: Dict) -> bool:
    c = cmd.lower()
    for pat in policy.get("risky_patterns", DEFAULT_POLICY["risky_patterns"]):
        if pat in c:
            return True
    return False


def flags_allowed(cmd: str, args: List[str], policy: Dict) -> bool:
    allowlist = policy.get("allowlist", {})
    if cmd not in allowlist:
        # no specific rule; follow default decision later
        return True
    allowed = allowlist[cmd]
    if not allowed:
        return True
    # simplistic check: each token should start with an allowed flag or be a non-flag (path, subcommand)
    for a in args:
        if a.startswith("-"):
            if not any(a.startswith(prefix) for prefix in allowed):
                return False
    return True


def validate(cmdline: str) -> Tuple[str, str]:
    """
    Return (decision, reason): decision in {allow, warn, block}.
    """
    policy = load_policy()
    cmd, args = parse_command(cmdline)
    if not cmd:
        return "allow", "empty command"

    if is_risky(cmdline, policy):
        # risky patterns may be blocked
        if policy.get("block_risky", False):
            return "block", "matches risky pattern"
        return "warn", "matches risky pattern"

    if not flags_allowed(cmd, args, policy):
        return "warn", "flags not in allowlist"

    return policy.get("default_decision", DEFAULT_POLICY["default_decision"]), "policy default"
