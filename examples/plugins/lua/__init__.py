"""
Lua Plugin for OpenAgent

This plugin provides Lua script execution capabilities, allowing users to run
Lua code snippets and scripts directly through the OpenAgent interface.
"""

import asyncio
import subprocess
import tempfile
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, field_validator

from openagent.core.base import BaseTool, ToolResult


class LuaConfig(BaseModel):
    """Configuration for the Lua plugin."""
    
    lua_executable: str = "lua"  # Lua interpreter command
    timeout: int = 30  # Execution timeout in seconds
    max_output_size: int = 10000  # Maximum output size in characters
    allow_file_io: bool = False  # Allow file I/O operations
    enable_debug: bool = True  # Enable debug output
    
    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v <= 0 or v > 300:  # Max 5 minutes
            raise ValueError("Timeout must be between 1 and 300 seconds")
        return v
    
    @field_validator("max_output_size")
    @classmethod
    def validate_output_size(cls, v: int) -> int:
        if v <= 0 or v > 100000:  # Max 100KB
            raise ValueError("Max output size must be between 1 and 100000 characters")
        return v


class LuaTool(BaseTool):
    """Tool for executing Lua scripts and code snippets."""
    
    def __init__(self, config: Optional[LuaConfig] = None):
        super().__init__(
            name="lua",
            description="Execute Lua scripts and code snippets with safety restrictions"
        )
        self.config = config or LuaConfig()
        
    async def execute(self, code: str) -> ToolResult:
        """
        Execute Lua code safely.
        
        Args:
            code: The Lua code to execute
            
        Returns:
            ToolResult containing execution results
        """
        try:
            # Validate input
            if not code.strip():
                return ToolResult(
                    success=False,
                    content="❌ No Lua code provided"
                )
            
            # Check for potentially dangerous operations
            if not self.config.allow_file_io:
                dangerous_patterns = [
                    'io.', 'os.execute', 'os.remove', 'os.rename', 'loadfile',
                    'dofile', 'require("io")', 'require("os")', 'debug.debug'
                ]
                
                for pattern in dangerous_patterns:
                    if pattern in code.lower():
                        return ToolResult(
                            success=False,
                            content=f"❌ Potentially dangerous operation detected: {pattern}\\nFile I/O operations are disabled."
                        )
            
            # Execute Lua code
            result = await self._execute_lua_code(code)
            
            if result["success"]:
                output = result["output"]
                if len(output) > self.config.max_output_size:
                    output = output[:self.config.max_output_size] + "\\n... (output truncated)"
                
                content = f"🔧 Lua Execution Result:\\n```\\n{output}```"
                if result["stderr"]:
                    content += f"\\n\\n⚠️ Warnings/Errors:\\n```\\n{result['stderr']}```"
                
                return ToolResult(
                    success=True,
                    content=content,
                    metadata={
                        "execution_time": result.get("execution_time", 0),
                        "output_length": len(result["output"]),
                        "has_errors": bool(result["stderr"])
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    content=f"❌ Lua execution failed:\\n```\\n{result['error']}```"
                )
                
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                content=f"❌ Lua execution timed out after {self.config.timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Error executing Lua code: {str(e)}"
            )
    
    async def _execute_lua_code(self, code: str) -> Dict[str, Any]:
        """
        Execute Lua code in a secure environment.
        
        Args:
            code: Lua code to execute
            
        Returns:
            Dictionary with execution results
        """
        import time
        start_time = time.time()
        
        # Create temporary file for the Lua script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            # Add safety wrapper if file I/O is disabled
            if not self.config.allow_file_io:
                safe_code = f"""
-- OpenAgent Lua Sandbox
-- File I/O operations are disabled for security
local original_io = io
local original_os_execute = os.execute
local original_os_remove = os.remove
local original_loadfile = loadfile
local original_dofile = dofile

io = nil
os.execute = function() error("os.execute is disabled in sandbox mode") end
os.remove = function() error("os.remove is disabled in sandbox mode") end
loadfile = function() error("loadfile is disabled in sandbox mode") end
dofile = function() error("dofile is disabled in sandbox mode") end

-- User code starts here
{code}
"""
            else:
                safe_code = code
            
            f.write(safe_code)
            temp_file = f.name
        
        try:
            # Execute the Lua script
            process = await asyncio.create_subprocess_exec(
                self.config.lua_executable,
                temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tempfile.gettempdir()  # Run in temp directory
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout
            )
            
            execution_time = time.time() - start_time
            
            return {
                "success": process.returncode == 0,
                "output": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "return_code": process.returncode,
                "execution_time": execution_time,
                "error": stderr.decode('utf-8', errors='replace') if process.returncode != 0 else None
            }
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
    
    async def validate_lua_installation(self) -> bool:
        """Check if Lua is properly installed."""
        try:
            process = await asyncio.create_subprocess_exec(
                self.config.lua_executable,
                '-v',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5)
            return process.returncode == 0
        except:
            return False


class LuaPlugin:
    """Lua plugin for OpenAgent."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = LuaConfig(**(config or {}))
    
    def get_tools(self) -> list[BaseTool]:
        """Return the tools provided by this plugin."""
        return [LuaTool(self.config)]
    
    def get_info(self) -> Dict[str, Any]:
        """Return plugin information."""
        return {
            "name": "lua",
            "version": "1.0.0",
            "description": "Execute Lua scripts and code snippets with safety restrictions",
            "author": "OpenAgent Contributors",
            "tags": ["lua", "scripting", "code-execution", "programming"],
            "requirements": ["lua"],
            "permissions": [
                "execute_code",
                "temporary_files" + (" file_io" if self.config.allow_file_io else "")
            ]
        }


# Factory function for plugin loading
def create_plugin(config: Optional[Dict[str, Any]] = None) -> LuaPlugin:
    """Create and return a LuaPlugin instance."""
    return LuaPlugin(config)


# Example Lua scripts for testing
EXAMPLE_SCRIPTS = {
    "hello_world": """
print("Hello from Lua!")
print("OpenAgent Lua Plugin is working!")
""",
    
    "math_operations": """
-- Basic math operations
local a, b = 10, 5
print("a = " .. a .. ", b = " .. b)
print("Addition: " .. a .. " + " .. b .. " = " .. (a + b))
print("Subtraction: " .. a .. " - " .. b .. " = " .. (a - b))
print("Multiplication: " .. a .. " * " .. b .. " = " .. (a * b))
print("Division: " .. a .. " / " .. b .. " = " .. (a / b))
print("Modulo: " .. a .. " % " .. b .. " = " .. (a % b))
""",
    
    "table_operations": """
-- Table operations
local fruits = {"apple", "banana", "orange", "grape"}
print("Fruits table:")
for i, fruit in ipairs(fruits) do
    print(i .. ". " .. fruit)
end

local person = {
    name = "John",
    age = 30,
    city = "New York"
}
print("\\nPerson info:")
for key, value in pairs(person) do
    print(key .. ": " .. tostring(value))
end
""",
    
    "functions": """
-- Function example
function factorial(n)
    if n <= 1 then
        return 1
    else
        return n * factorial(n - 1)
    end
end

print("Factorial examples:")
for i = 1, 5 do
    print("factorial(" .. i .. ") = " .. factorial(i))
end
""",
    
    "string_operations": """
-- String operations
local text = "OpenAgent Lua Plugin"
print("Original: " .. text)
print("Length: " .. #text)
print("Uppercase: " .. string.upper(text))
print("Lowercase: " .. string.lower(text))
print("Reversed: " .. string.reverse(text))

-- Pattern matching
local words = {}
for word in string.gmatch(text, "%w+") do
    table.insert(words, word)
end
print("Words: " .. table.concat(words, ", "))
"""
}
