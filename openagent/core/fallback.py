"""
Fast fallback system for OpenAgent.

This module provides rule-based responses for common queries when the AI model
is unavailable, slow, or gives poor responses.
"""

import re
from typing import Optional, Dict, Any


class FastFallback:
    """Rule-based fallback system for common terminal tasks."""
    
    def __init__(self):
        self.command_explanations = {
            "ls": "List directory contents. Shows files and folders in the current directory.",
            "ls -la": "List all files (including hidden) in long format with permissions, ownership, and timestamps.",
            "pwd": "Print working directory. Shows the full path of current directory.",
            "cd": "Change directory. Use 'cd path' to navigate to a directory.",
            "mkdir": "Create directory. Use 'mkdir dirname' to create a new folder.",
            "rm": "Remove files. Use 'rm filename' to delete a file. BE CAREFUL - this is permanent!",
            "cp": "Copy files. Use 'cp source destination' to copy files or directories.",
            "mv": "Move/rename files. Use 'mv source destination' to move or rename.",
            "grep": "Search text. Use 'grep pattern file' to find text in files.",
            "find": "Find files. Use 'find path -name pattern' to search for files.",
            "ps": "Show running processes. Use 'ps aux' to see all processes.",
            "top": "Show system processes in real-time. Press 'q' to quit.",
            "df": "Show disk usage. Use 'df -h' for human-readable format.",
            "du": "Show directory sizes. Use 'du -sh *' to see sizes of all items.",
            "chmod": "Change file permissions. Use 'chmod 755 file' to set permissions.",
            "chown": "Change file ownership. Use 'chown user:group file'.",
            "tar": "Archive files. Use 'tar -czf archive.tar.gz files' to compress.",
            "zip": "Create zip archives. Use 'zip archive.zip files' to compress.",
            "wget": "Download files from web. Use 'wget URL' to download.",
            "curl": "Transfer data from servers. Use 'curl URL' to fetch content.",
            "ssh": "Secure shell connection. Use 'ssh user@host' to connect remotely.",
            "scp": "Secure copy over network. Use 'scp file user@host:path' to copy.",
            "git": "Git version control. Use 'git status' to check repository status.",
            "docker": "Container platform. Use 'docker ps' to see running containers.",
            "echo": "Print text to terminal. Use 'echo \"hello\"' to display text.",
            "cat": "Display file contents. Use 'cat filename' to show file content.",
            "less": "View file contents page by page. Use 'less filename' then 'q' to quit.",
            "tail": "Show end of files. Use 'tail -f logfile' to follow log files.",
            "head": "Show beginning of files. Use 'head -10 file' to see first 10 lines.",
        }
        
        self.math_patterns = [
            (r"(?:what.{0,10}is|calculate|compute|solve)\s*(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)", self._handle_math),
            (r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)", self._handle_math),
        ]
        
        self.common_patterns = [
            (r"(?:how to|how do i|help me|show me).{0,30}list.{0,10}files", "Use 'ls' to list files, or 'ls -la' for detailed view with hidden files."),
            (r"(?:how to|how do i).{0,30}change.{0,10}directory", "Use 'cd directory_name' to change to a directory, or 'cd ..' to go up one level."),
            (r"(?:how to|how do i).{0,30}create.{0,10}(?:folder|directory)", "Use 'mkdir folder_name' to create a new directory."),
            (r"(?:how to|how do i).{0,30}delete.{0,10}file", "Use 'rm filename' to delete a file. Use 'rm -r dirname' for directories. BE CAREFUL!"),
            (r"(?:how to|how do i).{0,30}copy.{0,10}file", "Use 'cp source destination' to copy files, or 'cp -r source dest' for directories."),
            (r"(?:check|show|see).{0,20}(?:disk|space|storage)", "Use 'df -h' to check disk usage, or 'du -sh *' to see directory sizes."),
            (r"(?:check|show|see).{0,20}(?:processes|running)", "Use 'ps aux' to see all processes, or 'top' for real-time process monitor."),
            (r"(?:check|show|see).{0,20}(?:memory|ram)", "Use 'free -h' to check memory usage, or 'htop' for detailed system info."),
            (r"(?:download|get|fetch).{0,20}(?:file|url)", "Use 'wget URL' or 'curl -O URL' to download files from the internet."),
            (r"(?:what.{0,5}is|explain).{0,10}git", "Git is a version control system. Use 'git status' to check repo state, 'git add .' to stage files, 'git commit -m \"message\"' to save changes."),
            (r"(?:what.{0,5}is|explain).{0,10}docker", "Docker is a containerization platform. Use 'docker ps' to see running containers, 'docker images' to list images."),
            (r"(?:install|setup).{0,10}python", "Install Python: On Ubuntu/Debian: 'sudo apt install python3', On macOS: 'brew install python3', On Windows: download from python.org"),
            (r"(?:install|setup).{0,10}node", "Install Node.js: Use official installer from nodejs.org, or 'nvm install node' if you have NVM."),
        ]

    def can_handle(self, prompt: str) -> bool:
        """Check if this prompt can be handled by fallback system."""
        prompt_lower = prompt.lower().strip()
        
        # Check for math
        for pattern, _ in self.math_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                return True
        
        # Check for common patterns
        for pattern, _ in self.common_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                return True
                
        # Check for direct command explanations
        for cmd in self.command_explanations:
            if f"explain {cmd}" in prompt_lower or f"what is {cmd}" in prompt_lower:
                return True
        
        return False
    
    def handle_prompt(self, prompt: str) -> Optional[str]:
        """Handle a prompt using fallback rules."""
        prompt_lower = prompt.lower().strip()
        
        # Try math first
        for pattern, handler in self.math_patterns:
            match = re.search(pattern, prompt_lower, re.IGNORECASE)
            if match:
                return handler(match)
        
        # Try common patterns
        for pattern, response in self.common_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                return f"💡 Quick Answer: {response}"
        
        # Try command explanations
        for cmd, explanation in self.command_explanations.items():
            if f"explain {cmd}" in prompt_lower or f"what is {cmd}" in prompt_lower:
                return f"📖 Command: `{cmd}`\n\n{explanation}"
        
        return None
    
    def _handle_math(self, match) -> str:
        """Handle simple math operations."""
        try:
            if len(match.groups()) >= 3:
                num1 = float(match.group(1))
                op = match.group(2)
                num2 = float(match.group(3))
                
                if op == '+':
                    result = num1 + num2
                elif op == '-':
                    result = num1 - num2
                elif op == '*':
                    result = num1 * num2
                elif op == '/':
                    if num2 == 0:
                        return "❌ Error: Cannot divide by zero"
                    result = num1 / num2
                else:
                    return "❌ Error: Unsupported operation"
                
                # Format result nicely
                if result.is_integer():
                    return f"🧮 {num1} {op} {num2} = {int(result)}"
                else:
                    return f"🧮 {num1} {op} {num2} = {result:.2f}"
            
        except (ValueError, AttributeError):
            pass
        
        return "❌ Error: Could not parse math expression"


# Global fallback instance
_fallback = FastFallback()


def can_handle_fast(prompt: str) -> bool:
    """Check if prompt can be handled by fast fallback."""
    return _fallback.can_handle(prompt)


def handle_fast(prompt: str) -> Optional[str]:
    """Handle prompt with fast fallback system."""
    return _fallback.handle_prompt(prompt)
