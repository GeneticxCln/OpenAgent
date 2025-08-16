"""
Safety and Policy Engine for OpenAgent.

This module provides comprehensive safety controls including:
- Command approval and risk assessment
- Allowlist/denylist pattern matching
- Sandboxing capabilities
- Tamper-evident audit logging
"""

import asyncio
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Pattern
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for command execution."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


class PolicyDecision(Enum):
    """Policy decisions for command execution."""
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"
    EXPLAIN_ONLY = "explain_only"


@dataclass
class CommandPolicy:
    """Policy configuration for command execution."""
    default_mode: str = "explain_only"  # "explain_only", "approve", "execute"
    require_approval_for_medium: bool = True
    block_high_risk: bool = True
    allow_admin_override: bool = False
    sandbox_mode: bool = False
    audit_enabled: bool = True
    
    # Pattern lists
    allowlist_patterns: List[str] = None
    denylist_patterns: List[str] = None
    safe_paths: List[str] = None
    restricted_paths: List[str] = None
    
    def __post_init__(self):
        """Initialize default patterns if not provided."""
        if self.allowlist_patterns is None:
            self.allowlist_patterns = [
                r"^ls(\s|$)", r"^pwd$", r"^cd\s", r"^echo\s",
                r"^git (status|log|diff|branch|remote)",
                r"^python\s.*\.py$", r"^pip list", r"^npm list",
                r"^cat\s", r"^grep\s", r"^find\s.*-name",
                r"^docker ps", r"^kubectl get",
            ]
        
        if self.denylist_patterns is None:
            self.denylist_patterns = [
                r"rm\s+-rf\s+/", r"chmod\s+777", r":\(\)\s*\{.*:\|:.*\}",  # Fork bomb (flexible)
                r"dd\s+if=/dev/(zero|random)", r"mkfs\.", 
                r">\s*/dev/sd[a-z]", r"fdisk\s+/dev/",
                r"curl.*\|\s*(bash|sh)", r"wget.*\|\s*(bash|sh)",
                r"base64\s+-d.*\|\s*(bash|sh)",
                r"/etc/passwd", r"/etc/shadow", r"~/.ssh/",
                r"sudo\s+passwd", r"userdel\s", r"groupdel\s",
            ]
        
        if self.safe_paths is None:
            self.safe_paths = [
                os.path.expanduser("~/Documents"),
                os.path.expanduser("~/Downloads"),
                os.path.expanduser("~/Desktop"),
                "/tmp", "/var/tmp",
            ]
        
        if self.restricted_paths is None:
            self.restricted_paths = [
                "/etc", "/boot", "/sys", "/proc",
                "/usr/bin", "/usr/sbin", "/bin", "/sbin",
                os.path.expanduser("~/.ssh"),
                os.path.expanduser("~/.gnupg"),
            ]


@dataclass
class AuditEntry:
    """Audit log entry for command execution."""
    timestamp: float
    block_id: str
    user_id: Optional[str]
    command: str
    argv: List[str]
    risk_level: str
    policy_decision: str
    executed: bool
    exit_code: Optional[int]
    error: Optional[str]
    parent_hash: Optional[str] = None
    hash: Optional[str] = None
    # New metadata fields (non-breaking):
    iso_timestamp: Optional[str] = None
    version: str = "v1"
    
    def calculate_hash(self) -> str:
        """Calculate hash for this entry including parent hash for chain integrity."""
        content = json.dumps({
            "timestamp": self.timestamp,
            "block_id": self.block_id,
            "command": self.command,
            "argv": self.argv,
            "parent_hash": self.parent_hash,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class PolicyEngine:
    """
    Main policy engine for command safety and execution control.
    """
    
    def __init__(self, policy: Optional[CommandPolicy] = None, audit_path: Optional[Path] = None):
        """
        Initialize the policy engine.
        
        Args:
            policy: Policy configuration (uses defaults if not provided)
            audit_path: Path for audit logs (defaults to ~/.openagent/audit/)
        """
        self.policy = policy or CommandPolicy()
        self.audit_path = audit_path or Path.home() / ".openagent" / "audit"
        self.audit_path.mkdir(parents=True, exist_ok=True)
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Initialize audit chain
        self._last_audit_hash = self._get_last_audit_hash()
        
        logger.info(f"Policy engine initialized with mode: {self.policy.default_mode}")
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self._allowlist_re = [re.compile(p, re.IGNORECASE) for p in self.policy.allowlist_patterns]
        self._denylist_re = [re.compile(p, re.IGNORECASE) for p in self.policy.denylist_patterns]
    
    def _get_last_audit_hash(self) -> Optional[str]:
        """Get the hash of the last audit entry for chain integrity."""
        audit_file = self.audit_path / f"audit_{time.strftime('%Y%m')}.jsonl"
        if not audit_file.exists():
            return None
        
        try:
            with open(audit_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_entry = json.loads(lines[-1])
                    return last_entry.get("hash")
        except Exception as e:
            logger.error(f"Failed to read last audit hash: {e}")
        
        return None
    
    async def evaluate_command(
        self, 
        command: str, 
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[PolicyDecision, RiskLevel, List[str]]:
        """
        Evaluate a command against the policy.
        
        Args:
            command: Command to evaluate
            user_id: Optional user identifier
            context: Optional execution context
            
        Returns:
            Tuple of (policy decision, risk level, list of reasons)
        """
        # Parse command into argv for detailed analysis
        try:
            import shlex
            argv = shlex.split(command)
        except Exception:
            argv = command.split()
        
        # Check denylist first
        is_denylisted = False
        denylist_pattern = None
        for pattern in self._denylist_re:
            if pattern.search(command):
                logger.warning(f"Command matched denylist: {pattern.pattern}")
                is_denylisted = True
                denylist_pattern = pattern.pattern
                break
        
        # Assess risk level
        risk_level, risk_reasons = self._assess_risk(command, argv, context)
        
        # If denylisted, treat as BLOCKED risk
        if is_denylisted:
            risk_level = RiskLevel.BLOCKED
            risk_reasons = [f"Matched denylist pattern: {denylist_pattern}"]
        
        # Check allowlist
        is_allowlisted = any(pattern.search(command) for pattern in self._allowlist_re)
        
        # Explain-only mode overrides everything
        if self.policy.default_mode == "explain_only":
            return PolicyDecision.EXPLAIN_ONLY, risk_level, risk_reasons
        
        # Determine policy decision based on risk and configuration
        if risk_level == RiskLevel.BLOCKED:
            return PolicyDecision.DENY, risk_level, risk_reasons
        
        if risk_level == RiskLevel.HIGH:
            if self.policy.block_high_risk and not self.policy.allow_admin_override:
                return PolicyDecision.DENY, risk_level, risk_reasons
            return PolicyDecision.REQUIRE_APPROVAL, risk_level, risk_reasons
        
        if risk_level == RiskLevel.MEDIUM:
            if self.policy.require_approval_for_medium and not is_allowlisted:
                return PolicyDecision.REQUIRE_APPROVAL, risk_level, risk_reasons
        
        if is_allowlisted:
            return PolicyDecision.ALLOW, risk_level, ["Command is allowlisted"]
        
        # Default based on policy mode
        if self.policy.default_mode == "approve":
            return PolicyDecision.REQUIRE_APPROVAL, risk_level, risk_reasons
        elif self.policy.default_mode == "execute":
            return PolicyDecision.ALLOW, risk_level, risk_reasons
        else:
            return PolicyDecision.EXPLAIN_ONLY, risk_level, risk_reasons
    
    def _assess_risk(
        self, 
        command: str, 
        argv: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[RiskLevel, List[str]]:
        """
        Assess the risk level of a command.
        
        Args:
            command: Full command string
            argv: Parsed command arguments
            context: Optional execution context
            
        Returns:
            Tuple of (risk level, list of reasons)
        """
        reasons = []
        risk_score = 0
        
        # Check for sudo/privilege escalation
        if argv and argv[0] in ["sudo", "su", "doas"]:
            risk_score += 10
            reasons.append("Requires elevated privileges")
        
        # Check for dangerous commands
        dangerous_commands = {
            "rm": (8, "destructive operation"), 
            "rmdir": (5, "directory removal"), 
            "dd": (10, "low-level disk operation"), 
            "mkfs": (10, "filesystem creation"),
            "fdisk": (10, "disk partitioning"), 
            "parted": (10, "disk partitioning"), 
            "format": (10, "disk formatting"),
            "chmod": (5, "permission modification"), 
            "chown": (5, "ownership change"), 
            "killall": (6, "process termination"),
            "systemctl": (4, "service management"), 
            "service": (4, "service management"),
        }
        
        if argv:
            base_cmd = os.path.basename(argv[0])
            if base_cmd in dangerous_commands:
                score, description = dangerous_commands[base_cmd]
                risk_score += score
                reasons.append(f"Potentially dangerous command: {description}")
        
        # Check for dangerous flags
        dangerous_flags = {
            "-rf": 5, "--force": 3, "--recursive": 2,
            "--no-preserve-root": 10, "-exec": 5,
        }
        
        for arg in argv[1:]:
            for flag, score in dangerous_flags.items():
                if flag in arg:
                    risk_score += score
                    reasons.append(f"Dangerous flag: {flag}")
        
        # Check for path-based risks
        for path in self.policy.restricted_paths:
            if path in command:
                risk_score += 8
                reasons.append(f"Accesses restricted path: {path}")
        
        # Check for output redirection to sensitive locations
        if re.search(r'>\s*/(?:etc|boot|sys|proc)', command):
            risk_score += 10
            reasons.append("Writes to system directory")
        
        # Check for network operations
        network_cmds = ["curl", "wget", "nc", "netcat", "telnet", "ssh"]
        if argv and any(cmd in argv[0] for cmd in network_cmds):
            if "|" in command and any(sh in command for sh in ["sh", "bash", "zsh"]):
                risk_score += 10
                reasons.append("Pipes network content to shell")
            else:
                risk_score += 2
                reasons.append("Network operation")
        
        # Check for script execution
        if re.search(r'\.(sh|bash|zsh|fish|ksh|csh|tcsh)(\s|$)', command):
            risk_score += 5
            reasons.append("Executes shell script")
        
        # Determine risk level based on score
        if risk_score >= 15:
            return RiskLevel.BLOCKED, reasons
        elif risk_score >= 10:
            return RiskLevel.HIGH, reasons
        elif risk_score >= 5:
            return RiskLevel.MEDIUM, reasons
        else:
            return RiskLevel.LOW, reasons if reasons else ["Command appears safe"]
    
    async def audit_command(
        self,
        command: str,
        argv: List[str],
        risk_level: RiskLevel,
        policy_decision: PolicyDecision,
        executed: bool,
        exit_code: Optional[int] = None,
        error: Optional[str] = None,
        user_id: Optional[str] = None,
        block_id: Optional[str] = None,
    ):
        """
        Log command execution to audit trail.
        
        Args:
            command: Command that was evaluated
            argv: Parsed command arguments
            risk_level: Assessed risk level
            policy_decision: Policy decision made
            executed: Whether command was actually executed
            exit_code: Exit code if executed
            error: Error message if any
            user_id: User identifier
            block_id: Block identifier for command grouping
        """
        if not self.policy.audit_enabled:
            return
        
# Create audit entry
        entry = AuditEntry(
            timestamp=time.time(),
            block_id=block_id or hashlib.md5(f"{command}{time.time()}".encode()).hexdigest()[:12],
            user_id=user_id,
            command=command,
            argv=argv,
            risk_level=risk_level.value,
            policy_decision=policy_decision.value,
            executed=executed,
            exit_code=exit_code,
            error=error,
            parent_hash=self._last_audit_hash,
        )
        # Enrich with ISO timestamp and version
        try:
            from datetime import datetime, timezone
            entry.iso_timestamp = datetime.now(timezone.utc).isoformat()
        except Exception:
            entry.iso_timestamp = None
        entry.version = "v1"
        
        # Calculate hash for chain integrity
        entry.hash = entry.calculate_hash()
        self._last_audit_hash = entry.hash
        
        # Write to audit log (monthly rotation)
        audit_file = self.audit_path / f"audit_{time.strftime('%Y%m')}.jsonl"
        try:
            with open(audit_file, 'a') as f:
                f.write(json.dumps(asdict(entry)) + '\n')
            logger.debug(f"Audit entry written: {entry.block_id}")
        except Exception as e:
            logger.error(f"Failed to write audit entry: {e}")
    
    async def execute_sandboxed(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Execute command in a sandboxed environment with resource limits.
        
        Args:
            command: Command to execute
            cwd: Working directory (restricted to safe paths)
            timeout: Execution timeout in seconds
            
        Returns:
            Execution result dictionary
        """
        if not self.policy.sandbox_mode:
            raise ValueError("Sandbox mode is not enabled in policy")
        
        # Validate working directory
        if cwd:
            cwd_path = Path(cwd).resolve()
            if not any(str(cwd_path).startswith(str(Path(safe).resolve())) 
                      for safe in self.policy.safe_paths):
                return {
                    "success": False,
                    "error": f"Working directory not in safe paths: {cwd}",
                    "sandboxed": True,
                }
        
        # Build sandbox command with restrictions
        sandbox_cmd = self._build_sandbox_command(command, cwd)
        
        try:
            # Execute with timeout
            process = await asyncio.create_subprocess_shell(
                sandbox_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.terminate()
                await process.wait()
                return {
                    "success": False,
                    "error": f"Command timeout after {timeout} seconds",
                    "sandboxed": True,
                }
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "exit_code": process.returncode,
                "sandboxed": True,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "sandboxed": True,
            }
    
    def _build_sandbox_command(self, command: str, cwd: Optional[str] = None) -> str:
        """
        Build sandboxed command with resource limits.
        
        Args:
            command: Original command
            cwd: Working directory
            
        Returns:
            Sandboxed command string
        """
        # Use Linux namespaces and resource limits if available
        sandbox_prefix = []
        
        # Check for unshare availability (Linux namespaces)
        if subprocess.run(["which", "unshare"], capture_output=True).returncode == 0:
            # Create new namespaces (user, pid, mount, network)
            sandbox_prefix.append("unshare --user --pid --mount --net --fork")
        
        # Apply resource limits
        limits = [
            "ulimit -t 30",      # CPU time limit (30 seconds)
            "ulimit -v 524288",  # Virtual memory limit (512MB)
            "ulimit -f 10240",   # File size limit (10MB)
            "ulimit -n 256",     # Open files limit
        ]
        
        # Combine sandbox prefix, limits, and command
        if sandbox_prefix:
            sandbox_cmd = f"{' '.join(sandbox_prefix)} sh -c '{'; '.join(limits)}; {command}'"
        else:
            # Fallback to just resource limits
            sandbox_cmd = f"sh -c '{'; '.join(limits)}; {command}'"
        
        return sandbox_cmd
    
    def verify_audit_integrity(self, start_date: Optional[str] = None) -> bool:
        """
        Verify the integrity of the audit log chain.
        
        Args:
            start_date: Optional start date (YYYYMM format)
            
        Returns:
            True if audit chain is valid, False otherwise
        """
        result = self.verify_audit_integrity_report(start_date)
        return bool(result.get("valid"))

    def verify_audit_integrity_report(self, start_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a structured integrity verification report.
        
        Args:
            start_date: Optional start date (YYYYMM format)
            
        Returns:
            Dict with keys: valid (bool), checked (int), first_block_id, last_block_id, first_file, last_file, error (optional)
        """
        report: Dict[str, Any] = {"valid": True, "checked": 0}
        audit_files = sorted(self.audit_path.glob("audit_*.jsonl"))
        if start_date:
            audit_files = [f for f in audit_files if f.stem >= f"audit_{start_date}"]
        previous_hash = None
        first_block = None
        last_block = None
        try:
            for audit_file in audit_files:
                with open(audit_file, 'r') as f:
                    for line in f:
                        entry_dict = json.loads(line)
                        entry = AuditEntry(**entry_dict)
                        if report.get("first_file") is None:
                            report["first_file"] = str(audit_file)
                        report["last_file"] = str(audit_file)
                        # Verify parent hash matches
                        if previous_hash and entry.parent_hash != previous_hash:
                            logger.error(f"Audit chain broken at {entry.block_id}")
                            report.update({
                                "valid": False,
                                "error": f"chain_broken_at:{entry.block_id}",
                            })
                            return report
                        # Verify entry hash
                        calculated_hash = entry.calculate_hash()
                        if entry.hash != calculated_hash:
                            logger.error(f"Audit entry tampered: {entry.block_id}")
                            report.update({
                                "valid": False,
                                "error": f"tampered_at:{entry.block_id}",
                            })
                            return report
                        previous_hash = entry.hash
                        report["checked"] += 1
                        if first_block is None:
                            first_block = entry.block_id
                        last_block = entry.block_id
            report["first_block_id"] = first_block
            report["last_block_id"] = last_block
            return report
        except Exception as e:
            logger.error(f"Failed to verify audit files: {e}")
            report.update({"valid": False, "error": str(e)})
            return report
    
    def export_audit_report(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_format: str = "json"
    ) -> str:
        """
        Export audit report for analysis.
        
        Args:
            start_date: Start date (YYYYMM format)
            end_date: End date (YYYYMM format)
            output_format: Output format (json, csv, summary)
            
        Returns:
            Formatted audit report
        """
        # Collect audit entries
        entries = []
        audit_files = sorted(self.audit_path.glob("audit_*.jsonl"))
        
        for audit_file in audit_files:
            file_date = audit_file.stem.replace("audit_", "")
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue
            
            try:
                with open(audit_file, 'r') as f:
                    for line in f:
                        entries.append(json.loads(line))
            except Exception as e:
                logger.error(f"Failed to read audit file {audit_file}: {e}")
        
        if output_format == "json":
            return json.dumps(entries, indent=2)
        
        elif output_format == "summary":
            summary = {
                "total_commands": len(entries),
                "executed": sum(1 for e in entries if e.get("executed")),
                "blocked": sum(1 for e in entries if e.get("policy_decision") == "deny"),
                "risk_levels": {},
                "top_commands": {},
            }
            
            for entry in entries:
                # Count risk levels
                risk = entry.get("risk_level", "unknown")
                summary["risk_levels"][risk] = summary["risk_levels"].get(risk, 0) + 1
                
                # Count command types
                if entry.get("argv"):
                    cmd = entry["argv"][0] if entry["argv"] else "unknown"
                    summary["top_commands"][cmd] = summary["top_commands"].get(cmd, 0) + 1
            
            # Sort top commands
            summary["top_commands"] = dict(
                sorted(summary["top_commands"].items(), key=lambda x: x[1], reverse=True)[:10]
            )
            
            return json.dumps(summary, indent=2)
        
        else:
            return json.dumps({"error": f"Unsupported format: {output_format}"})


# Global policy instance (can be configured at startup)
_global_policy_engine: Optional[PolicyEngine] = None


def get_policy_engine() -> PolicyEngine:
    """Get the global policy engine instance."""
    global _global_policy_engine
    if _global_policy_engine is None:
        _global_policy_engine = PolicyEngine()
    return _global_policy_engine


def configure_policy(policy: CommandPolicy, audit_path: Optional[Path] = None):
    """Configure the global policy engine."""
    global _global_policy_engine
    _global_policy_engine = PolicyEngine(policy, audit_path)
    logger.info("Global policy engine configured")
