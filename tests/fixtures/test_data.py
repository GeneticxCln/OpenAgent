"""
Test fixtures and sample data for OpenAgent tests.

This module provides reusable test data and fixtures
for consistent testing across the test suite.
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Any

# Sample code snippets for testing
SAMPLE_PYTHON_CODE = {
    "hello_world": """
def hello_world(name="World"):
    \"\"\"A simple hello world function.\"\"\"
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(hello_world())
""",
    
    "fibonacci": """
def fibonacci(n):
    \"\"\"Calculate nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
    
    "buggy_code": """
def divide_numbers(a, b):
    # Bug: no zero division check
    return a / b

def process_list(items):
    # Bug: modifying list while iterating
    for item in items:
        if item < 0:
            items.remove(item)
    return items
""",
    
    "complex_class": """
class DataProcessor:
    \"\"\"A complex class for testing analysis.\"\"\"
    
    def __init__(self, config: dict):
        self.config = config
        self.data = []
        
    def load_data(self, source: str) -> bool:
        \"\"\"Load data from source.\"\"\"
        try:
            # Simulate data loading
            self.data = [1, 2, 3, 4, 5]
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
            
    def process(self) -> List[int]:
        \"\"\"Process the loaded data.\"\"\"
        if not self.data:
            raise ValueError("No data to process")
            
        result = []
        for item in self.data:
            if item > self.config.get('threshold', 0):
                result.append(item * 2)
        return result
"""
}

# Sample shell commands for testing
SAMPLE_COMMANDS = {
    "safe": [
        "ls -la",
        "pwd",
        "whoami",
        "date",
        "cat file.txt",
        "grep pattern file.txt",
        "find . -name '*.py'",
        "ps aux | grep python",
        "df -h",
        "git status",
        "python --version",
        "pip list"
    ],
    
    "risky": [
        "rm -rf /",
        "sudo rm -rf /home/user",
        "mkfs.ext4 /dev/sda1",
        "fdisk /dev/sda",
        "chmod 777 /etc/passwd",
        "killall -9 python",
        "dd if=/dev/zero of=/dev/sda",
        "chown root:root /etc/shadow"
    ],
    
    "with_flags": [
        "ls -la --color=auto",
        "grep -r -i pattern .",
        "find . -type f -name '*.py' -exec grep -l 'import' {} +",
        "git log --oneline --graph",
        "docker run -it --rm ubuntu:latest",
        "pip install --upgrade package",
        "python -m pytest -v --cov=."
    ]
}

# Sample LLM responses for testing
SAMPLE_RESPONSES = {
    "code_explanation": """
This Python function implements the Fibonacci sequence calculation:

1. **Base case**: If n <= 1, return n directly
2. **Recursive case**: Return sum of fibonacci(n-1) + fibonacci(n-2)
3. **Time complexity**: O(2^n) - exponential (inefficient for large n)
4. **Space complexity**: O(n) due to call stack

**Improvements suggested**:
- Use memoization or dynamic programming for better performance
- Add input validation for negative numbers
- Consider iterative approach for large numbers
""",
    
    "command_explanation": """
The command `ls -la` lists directory contents with detailed information:

- `ls`: List directory contents
- `-l`: Long format (shows permissions, size, date, etc.)
- `-a`: Show all files including hidden files (starting with .)

**Output includes**:
- File permissions (rwxrwxrwx)
- Number of links
- Owner and group
- File size
- Last modification date
- File/directory name

This is a safe read-only command with no security risks.
""",
    
    "code_generation": """
```python
def calculate_fibonacci(n: int) -> int:
    \"\"\"
    Calculate the nth Fibonacci number using dynamic programming.
    
    Args:
        n (int): The position in the Fibonacci sequence
        
    Returns:
        int: The nth Fibonacci number
        
    Raises:
        ValueError: If n is negative
    \"\"\"
    if n < 0:
        raise ValueError("Fibonacci sequence is not defined for negative numbers")
    
    if n <= 1:
        return n
    
    # Use dynamic programming for efficiency
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b

# Example usage
if __name__ == "__main__":
    for i in range(10):
        print(f"F({i}) = {calculate_fibonacci(i)}")
```
""",
    
    "error_message": """
I encountered an error while processing your request. This could be due to:

1. **Model loading issues**: The AI model might not be properly loaded
2. **Input parsing error**: The input format might be invalid
3. **Resource constraints**: Insufficient memory or compute resources
4. **Network connectivity**: Issues connecting to required services

Please try again with a simpler request, or check the system logs for more details.
"""
}

# Test configuration data
TEST_CONFIG = {
    "default_model": "tiny-llama",
    "default_device": "cpu",
    "test_timeout": 30,
    "max_test_iterations": 5,
    "mock_responses": True
}

# Policy test data
SAMPLE_POLICIES = {
    "strict": {
        "risky_patterns": [
            "rm", "rmdir", "mkfs", "fdisk", "dd", "chmod 777", 
            "chown", "killall", "sudo", "su"
        ],
        "allowlist": {
            "ls": ["-l", "-a", "-h", "--color"],
            "cat": [],
            "echo": [],
            "pwd": []
        },
        "default_decision": "block",
        "block_risky": True
    },
    
    "permissive": {
        "risky_patterns": ["rm -rf /", "mkfs", "fdisk"],
        "allowlist": {},
        "default_decision": "allow",
        "block_risky": False
    },
    
    "balanced": {
        "risky_patterns": [
            "rm", "rmdir", "mkfs", "fdisk", "dd", "chmod 777"
        ],
        "allowlist": {
            "git": ["status", "log", "diff", "add", "commit", "push", "pull"],
            "pip": ["install", "list", "show", "freeze"],
            "python": ["-m", "-c", "--version"]
        },
        "default_decision": "warn",
        "block_risky": False
    }
}


def create_test_files(base_dir: Path) -> Dict[str, Path]:
    """
    Create temporary test files with sample content.
    
    Args:
        base_dir: Base directory to create files in
        
    Returns:
        Dictionary mapping file types to their paths
    """
    files = {}
    
    # Create Python files
    for name, code in SAMPLE_PYTHON_CODE.items():
        file_path = base_dir / f"{name}.py"
        file_path.write_text(code)
        files[f"python_{name}"] = file_path
    
    # Create text files
    text_files = {
        "readme": "# Test Project\n\nThis is a test project for OpenAgent.",
        "config": "debug=true\nlog_level=info\nmax_workers=4",
        "data": "1,2,3,4,5\n6,7,8,9,10\n11,12,13,14,15"
    }
    
    for name, content in text_files.items():
        file_path = base_dir / f"{name}.txt"
        file_path.write_text(content)
        files[f"text_{name}"] = file_path
    
    # Create shell scripts
    shell_scripts = {
        "deploy": "#!/bin/bash\necho 'Deploying application...'\npython manage.py migrate",
        "backup": "#!/bin/bash\ntar -czf backup.tar.gz /home/user/data"
    }
    
    for name, content in shell_scripts.items():
        file_path = base_dir / f"{name}.sh"
        file_path.write_text(content)
        file_path.chmod(0o755)
        files[f"shell_{name}"] = file_path
    
    return files


def create_temp_test_env():
    """
    Create a temporary test environment with files and directories.
    
    Returns:
        Tuple of (temp_dir, file_paths)
    """
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create directory structure
    (temp_dir / "src").mkdir()
    (temp_dir / "tests").mkdir()
    (temp_dir / "docs").mkdir()
    (temp_dir / "config").mkdir()
    
    # Create files
    file_paths = create_test_files(temp_dir / "src")
    
    return temp_dir, file_paths


def get_mock_agent_config() -> Dict[str, Any]:
    """Get a mock agent configuration for testing."""
    return {
        "name": "TestAgent",
        "description": "Test agent for unit testing",
        "model_name": "tiny-llama",
        "llm_config": {
            "device": "cpu",
            "temperature": 0.5,
            "max_length": 1024
        },
        "max_iterations": 3,
        "safe_mode": True
    }


def get_mock_tool_results() -> List[Dict[str, Any]]:
    """Get mock tool execution results."""
    return [
        {
            "tool_name": "command_executor",
            "success": True,
            "content": "total 8\n-rw-r--r-- 1 user user 156 Jan 1 12:00 test.py",
            "metadata": {"command": "ls -la", "execution_time": 0.1}
        },
        {
            "tool_name": "file_manager",
            "success": True,
            "content": "File read successfully",
            "metadata": {"operation": "read", "file_size": 1024}
        },
        {
            "tool_name": "system_info",
            "success": True,
            "content": "CPU Usage: 45%\nMemory Usage: 60%",
            "metadata": {"timestamp": "2025-01-01T12:00:00Z"}
        }
    ]


class MockLLMResponses:
    """Mock LLM responses for consistent testing."""
    
    @staticmethod
    def get_code_analysis_response(code: str, language: str) -> str:
        """Get mock code analysis response."""
        if "bug" in code.lower() or "error" in code.lower():
            return """
**Code Analysis Results:**

🚨 **Issues Found:**
1. **Potential bug**: Missing error handling
2. **Performance concern**: Inefficient algorithm
3. **Best practice**: Missing type hints

🔧 **Recommendations:**
- Add proper error handling
- Consider using more efficient algorithms
- Add comprehensive docstrings
- Include input validation
"""
        else:
            return """
**Code Analysis Results:**

✅ **Code Quality: Good**

**Strengths:**
- Clear function structure
- Appropriate naming conventions
- Basic error handling present

**Minor Suggestions:**
- Consider adding type hints
- Add more comprehensive tests
- Include docstring examples
"""
    
    @staticmethod
    def get_command_explanation(command: str) -> str:
        """Get mock command explanation."""
        if any(risky in command.lower() for risky in ["rm", "mkfs", "fdisk"]):
            return f"""
⚠️ **WARNING: Potentially Dangerous Command**

Command: `{command}`

**What it does:**
This command performs system-level operations that could cause data loss or system damage.

**Risks:**
- Permanent data deletion
- System corruption
- Security vulnerabilities

**Recommendation:**
- Use with extreme caution
- Ensure backups exist
- Verify command syntax
- Consider safer alternatives
"""
        else:
            return f"""
**Command Explanation:**

`{command}`

**Purpose:** Safe system operation
**Risk Level:** Low
**Description:** This command performs read-only or safe operations without modifying system files.

**Common Usage:**
- Information gathering
- File viewing
- System monitoring

**Safety:** This command is generally safe to execute.
"""
    
    @staticmethod
    def get_code_generation_response(description: str, language: str) -> str:
        """Get mock code generation response."""
        return f"""
```{language}
# Generated code for: {description}
# Language: {language}

def generated_function():
    \"\"\"
    This is a generated function based on the request:
    {description}
    \"\"\"
    # Implementation would go here
    return "Generated code placeholder"

# Example usage
if __name__ == "__main__":
    result = generated_function()
    print(result)
```

**Notes:**
- This is a basic implementation
- Add error handling as needed
- Consider performance optimization
- Add unit tests for production use
"""
