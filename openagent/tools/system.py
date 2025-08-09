"""
System and terminal tools for OpenAgent.

This module provides tools for system operations, command execution,
file management, and terminal assistance.
"""

import asyncio
import os
import shutil
import subprocess
import platform
import psutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from openagent.core.base import BaseTool, ToolResult
from openagent.core.exceptions import ToolError


class CommandExecutor(BaseTool):
    """
    Tool for executing shell commands safely with security checks.
    """
    
    def __init__(self, default_explain_only: bool = True, **kwargs):
        super().__init__(
            name="command_executor",
            description="Execute shell commands with safety checks and explanations",
            **kwargs
        )
        
        # Default behavior: explain commands rather than execute
        self.default_explain_only = default_explain_only
        
        # Dangerous commands that should be restricted
        self.dangerous_commands = {
            "rm", "rmdir", "del", "format", "fdisk", "mkfs",
            "dd", "chmod 777", "chown", "su", "sudo",
            "passwd", "userdel", "groupdel", "killall",
        }
        
        # Commands that are generally safe to run
        self.safe_commands = {
            "ls", "dir", "pwd", "cd", "cat", "more", "less",
            "head", "tail", "grep", "find", "locate", "which",
            "ps", "top", "htop", "df", "du", "free", "uptime",
            "date", "whoami", "id", "env", "echo", "printf",
            "git", "pip", "python", "node", "npm", "curl", "wget",
        }
    
    async def execute(self, input_data: Union[str, Dict[str, Any]]) -> ToolResult:
        """Execute a shell command with safety checks."""
        try:
            if isinstance(input_data, dict):
                command = input_data.get("command", "")
                explain_only = input_data.get("explain_only", self.default_explain_only)
            else:
                command = str(input_data).strip()
                explain_only = self.default_explain_only
            
            if not command:
                return ToolResult(
                    success=False,
                    content="",
                    error="No command provided"
                )
            
            # Security check
            is_safe, reason = self._is_command_safe(command)
            
            if not is_safe and not explain_only:
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Command rejected for security reasons: {reason}"
                )
            
            # If explain_only is True, just provide explanation
            if explain_only:
                explanation = await self._explain_command(command)
                return ToolResult(
                    success=True,
                    content=explanation,
                    metadata={"command": command, "explained": True, "explain_only_default": self.default_explain_only}
                )
            
            # Execute the command
            result = await self._execute_command(command)
            
            return ToolResult(
                success=result["success"],
                content=result["output"],
                error=result.get("error"),
                metadata={
                    "command": command,
                    "exit_code": result.get("exit_code"),
                    "execution_time": result.get("execution_time")
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Command execution failed: {str(e)}"
            )
    
    def _is_command_safe(self, command: str) -> tuple[bool, str]:
        """Check if a command is safe to execute."""
        command_lower = command.lower().strip()
        
        # Check for dangerous commands
        for dangerous in self.dangerous_commands:
            if dangerous in command_lower:
                return False, f"Contains dangerous command: {dangerous}"
        
        # Check for potentially harmful patterns
        harmful_patterns = [
            "> /", ">> /",  # Writing to system directories
            "& rm ", "&& rm ",  # Chaining with rm
            "curl | sh", "wget | sh",  # Pipe to shell
            "$(", "`",  # Command substitution (can be dangerous)
        ]
        
        for pattern in harmful_patterns:
            if pattern in command_lower:
                return False, f"Contains potentially harmful pattern: {pattern}"
        
        # Check command length (very long commands might be suspicious)
        if len(command) > 1000:
            return False, "Command too long (potential security risk)"
        
        return True, "Command appears safe"
    
    async def _execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a shell command and return results."""
        import time
        start_time = time.time()
        
        try:
            # Use shell=True for command execution but with timeout
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            # Wait for command with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=30.0  # 30 second timeout
                )
            except asyncio.TimeoutError:
                process.terminate()
                await process.wait()
                return {
                    "success": False,
                    "output": "",
                    "error": "Command timed out after 30 seconds",
                    "exit_code": -1,
                    "execution_time": time.time() - start_time
                }
            
            # Decode output
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')
            
            # Combine output
            output = ""
            if stdout_str:
                output += stdout_str
            if stderr_str:
                if output:
                    output += "\\n--- STDERR ---\\n"
                output += stderr_str
            
            return {
                "success": process.returncode == 0,
                "output": output.strip(),
                "error": stderr_str.strip() if process.returncode != 0 else None,
                "exit_code": process.returncode,
                "execution_time": time.time() - start_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "exit_code": -1,
                "execution_time": time.time() - start_time
            }
    
    async def _explain_command(self, command: str) -> str:
        """Provide an explanation of what a command does."""
        # This is a simplified explanation system
        # In a full implementation, this could use the LLM
        
        parts = command.split()
        if not parts:
            return "Empty command"
        
        main_cmd = parts[0]
        
        explanations = {
            "ls": "List directory contents",
            "pwd": "Print working directory (current directory path)",
            "cd": "Change directory",
            "mkdir": "Create directory",
            "rmdir": "Remove empty directory",
            "cp": "Copy files or directories", 
            "mv": "Move/rename files or directories",
            "rm": "Remove/delete files or directories",
            "cat": "Display file contents",
            "grep": "Search for patterns in text",
            "find": "Search for files and directories",
            "ps": "Show running processes",
            "top": "Display running processes (interactive)",
            "df": "Show disk space usage",
            "free": "Show memory usage",
            "git": "Git version control command",
            "pip": "Python package installer",
            "python": "Python interpreter",
            "curl": "Transfer data from/to servers",
            "wget": "Download files from web",
        }
        
        base_explanation = explanations.get(main_cmd, f"Execute '{main_cmd}' command")
        
        # Add flags explanation if present
        flags_info = ""
        if len(parts) > 1:
            flags = [p for p in parts[1:] if p.startswith('-')]
            if flags:
                flags_info = f" with flags: {', '.join(flags)}"
        
        # Include the command name in the explanation for clarity
        return f"{main_cmd}: {base_explanation}{flags_info}"


class FileManager(BaseTool):
    """
    Tool for file and directory operations.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="file_manager", 
            description="Manage files and directories (list, read, create, copy, move)",
            **kwargs
        )
    
    async def execute(self, input_data: Union[str, Dict[str, Any]]) -> ToolResult:
        """Execute file management operations."""
        try:
            if isinstance(input_data, dict):
                operation = input_data.get("operation", "list")
                path = input_data.get("path", ".")
                content = input_data.get("content", "")
                destination = input_data.get("destination", "")
            else:
                # Parse simple string input
                parts = str(input_data).strip().split()
                operation = parts[0] if parts else "list"
                path = parts[1] if len(parts) > 1 else "."
                content = ""
                destination = parts[2] if len(parts) > 2 else ""
            
            result = await self._execute_file_operation(operation, path, content, destination)
            
            return ToolResult(
                success=result["success"],
                content=result["content"],
                error=result.get("error"),
                metadata=result.get("metadata", {})
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"File operation failed: {str(e)}"
            )
    
    async def _execute_file_operation(self, operation: str, path: str, content: str, destination: str) -> Dict[str, Any]:
        """Execute specific file operations."""
        try:
            path_obj = Path(path).expanduser().resolve()
            
            if operation == "list":
                return await self._list_directory(path_obj)
            elif operation == "read":
                return await self._read_file(path_obj)
            elif operation == "write":
                return await self._write_file(path_obj, content)
            elif operation == "copy":
                return await self._copy_item(path_obj, Path(destination))
            elif operation == "move":
                return await self._move_item(path_obj, Path(destination))
            elif operation == "delete":
                return await self._delete_item(path_obj)
            elif operation == "info":
                return await self._get_file_info(path_obj)
            else:
                return {
                    "success": False,
                    "content": "",
                    "error": f"Unknown operation: {operation}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "content": "",
                "error": str(e)
            }
    
    async def _list_directory(self, path: Path) -> Dict[str, Any]:
        """List directory contents."""
        if not path.exists():
            return {
                "success": False,
                "content": "",
                "error": f"Path does not exist: {path}"
            }
        
        if not path.is_dir():
            return {
                "success": False,
                "content": "",
                "error": f"Path is not a directory: {path}"
            }
        
        items = []
        for item in sorted(path.iterdir()):
            item_type = "directory" if item.is_dir() else "file"
            size = item.stat().st_size if item.is_file() else 0
            modified = item.stat().st_mtime
            
            items.append({
                "name": item.name,
                "type": item_type,
                "size": size,
                "modified": modified
            })
        
        content = f"Contents of {path}:\\n"
        for item in items:
            size_str = f"{item['size']} bytes" if item['type'] == 'file' else ""
            content += f"{item['type'].upper()}: {item['name']} {size_str}\\n"
        
        return {
            "success": True,
            "content": content.strip(),
            "metadata": {"path": str(path), "items": items}
        }
    
    async def _read_file(self, path: Path) -> Dict[str, Any]:
        """Read file contents."""
        if not path.exists():
            return {
                "success": False,
                "content": "",
                "error": f"File does not exist: {path}"
            }
        
        if not path.is_file():
            return {
                "success": False,
                "content": "",
                "error": f"Path is not a file: {path}"
            }
        
        try:
            content = path.read_text(encoding='utf-8')
            return {
                "success": True,
                "content": content,
                "metadata": {"path": str(path), "size": len(content)}
            }
        except UnicodeDecodeError:
            # Try reading as binary for non-text files
            return {
                "success": False,
                "content": "",
                "error": f"File appears to be binary: {path}"
            }
    
    async def _write_file(self, path: Path, content: str) -> Dict[str, Any]:
        """Write content to file."""
        try:
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            path.write_text(content, encoding='utf-8')
            
            return {
                "success": True,
                "content": f"Successfully wrote {len(content)} characters to {path}",
                "metadata": {"path": str(path), "size": len(content)}
            }
        except Exception as e:
            return {
                "success": False,
                "content": "",
                "error": f"Failed to write file: {e}"
            }
    
    async def _get_file_info(self, path: Path) -> Dict[str, Any]:
        """Get detailed file/directory information."""
        if not path.exists():
            return {
                "success": False,
                "content": "",
                "error": f"Path does not exist: {path}"
            }
        
        stat = path.stat()
        info = {
            "name": path.name,
            "path": str(path),
            "type": "directory" if path.is_dir() else "file",
            "size": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "accessed": stat.st_atime,
            "permissions": oct(stat.st_mode)[-3:],
        }
        
        if path.is_file():
            info["extension"] = path.suffix
        
        content = f"Information for {path}:\\n"
        for key, value in info.items():
            content += f"{key.capitalize()}: {value}\\n"
        
        return {
            "success": True,
            "content": content.strip(),
            "metadata": info
        }


class SystemInfo(BaseTool):
    """
    Tool for getting system information and monitoring.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="system_info",
            description="Get system information, resource usage, and monitoring data",
            **kwargs
        )
    
    async def execute(self, input_data: Union[str, Dict[str, Any]]) -> ToolResult:
        """Get system information."""
        try:
            if isinstance(input_data, dict):
                info_type = input_data.get("type", "overview")
            else:
                info_type = str(input_data).strip().lower() or "overview"
            
            result = await self._get_system_info(info_type)
            
            return ToolResult(
                success=True,
                content=result["content"],
                metadata=result["metadata"]
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Failed to get system info: {str(e)}"
            )
    
    async def _get_system_info(self, info_type: str) -> Dict[str, Any]:
        """Get specific type of system information."""
        if info_type == "overview":
            return self._get_overview()
        elif info_type == "cpu":
            return self._get_cpu_info()
        elif info_type == "memory":
            return self._get_memory_info()
        elif info_type == "disk":
            return self._get_disk_info()
        elif info_type == "network":
            return self._get_network_info()
        elif info_type == "processes":
            return self._get_process_info()
        else:
            return {
                "content": "Available info types: overview, cpu, memory, disk, network, processes",
                "metadata": {}
            }
    
    def _get_overview(self) -> Dict[str, Any]:
        """Get system overview."""
        uname = platform.uname()
        boot_time = psutil.boot_time()
        
        content = f"""System Overview:
Operating System: {uname.system} {uname.release}
Architecture: {uname.machine}
Processor: {uname.processor}
Hostname: {uname.node}
Boot Time: {boot_time}

CPU Usage: {psutil.cpu_percent(interval=1)}%
Memory Usage: {psutil.virtual_memory().percent}%
Disk Usage: {psutil.disk_usage('/').percent}%
"""
        
        return {
            "content": content.strip(),
            "metadata": {
                "system": uname.system,
                "release": uname.release,
                "machine": uname.machine,
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
            }
        }
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        cpu_count = psutil.cpu_count()
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        
        content = f"""CPU Information:
Physical Cores: {cpu_count}
Logical Cores: {cpu_count_logical}
Current Frequency: {cpu_freq.current:.2f} MHz
Max Frequency: {cpu_freq.max:.2f} MHz

Per-Core Usage:
"""
        for i, percent in enumerate(cpu_percent):
            content += f"Core {i}: {percent}%\\n"
        
        return {
            "content": content.strip(),
            "metadata": {
                "physical_cores": cpu_count,
                "logical_cores": cpu_count_logical,
                "frequency": cpu_freq._asdict() if cpu_freq else None,
                "per_core_usage": cpu_percent,
            }
        }
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        content = f"""Memory Information:
Total RAM: {memory.total / (1024**3):.2f} GB
Available RAM: {memory.available / (1024**3):.2f} GB
Used RAM: {memory.used / (1024**3):.2f} GB
Memory Usage: {memory.percent}%

Swap Memory:
Total: {swap.total / (1024**3):.2f} GB
Used: {swap.used / (1024**3):.2f} GB
Swap Usage: {swap.percent}%
"""
        
        return {
            "content": content.strip(),
            "metadata": {
                "memory": memory._asdict(),
                "swap": swap._asdict(),
            }
        }
    
    def _get_disk_info(self) -> Dict[str, Any]:
        """Get disk information."""
        partitions = psutil.disk_partitions()
        
        content = "Disk Information:\\n"
        partition_info = []
        
        for partition in partitions:
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                content += f"""
Partition: {partition.device}
Mountpoint: {partition.mountpoint}
File System: {partition.fstype}
Total: {usage.total / (1024**3):.2f} GB
Used: {usage.used / (1024**3):.2f} GB
Free: {usage.free / (1024**3):.2f} GB
Usage: {(usage.used / usage.total * 100):.1f}%
"""
                partition_info.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "usage": usage._asdict(),
                })
            except PermissionError:
                content += f"\\nPartition: {partition.device} (Permission denied)\\n"
        
        return {
            "content": content.strip(),
            "metadata": {
                "partitions": partition_info,
            }
        }
    
    def _get_process_info(self) -> Dict[str, Any]:
        """Get running process information."""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by CPU usage
        processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
        top_processes = processes[:10]
        
        content = "Top 10 Processes by CPU Usage:\\n"
        for proc in top_processes:
            content += f"PID: {proc['pid']}, Name: {proc['name']}, CPU: {proc['cpu_percent']}%, Memory: {proc['memory_percent']:.1f}%\\n"
        
        return {
            "content": content.strip(),
            "metadata": {
                "total_processes": len(processes),
                "top_processes": top_processes,
            }
        }
