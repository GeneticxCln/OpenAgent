"""
WASM Plugin for OpenAgent

This plugin provides WebAssembly (WASM) execution capabilities, allowing users to
compile and run WASM modules from various source languages like C, C++, Rust, etc.
"""

import asyncio
import subprocess
import tempfile
import os
import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, field_validator

from openagent.core.base import BaseTool, ToolResult


class WasmConfig(BaseModel):
    """Configuration for the WASM plugin."""
    
    wasmtime_executable: str = "wasmtime"  # Wasmtime runtime executable
    wat2wasm_executable: str = "wat2wasm"  # WebAssembly Text to Binary converter
    wasm2wat_executable: str = "wasm2wat"  # WebAssembly Binary to Text converter
    timeout: int = 30  # Execution timeout in seconds
    max_output_size: int = 10000  # Maximum output size in characters
    max_memory: str = "10MB"  # Maximum memory allocation
    enable_wasi: bool = True  # Enable WASI (WebAssembly System Interface)
    allowed_imports: List[str] = ["wasi_snapshot_preview1"]  # Allowed import modules
    
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


class WasmTool(BaseTool):
    """Tool for executing WebAssembly modules and WebAssembly Text format."""
    
    def __init__(self, config: Optional[WasmConfig] = None):
        super().__init__(
            name="wasm",
            description="Execute WebAssembly modules and convert between WAT/WASM formats"
        )
        self.config = config or WasmConfig()
        
    async def execute(self, input_data: Union[str, Dict[str, Any]]) -> ToolResult:
        """
        Execute WASM operations.
        
        Args:
            input_data: Can be:
                - String: WebAssembly Text (WAT) format code
                - Dict: {"action": "run|compile|decompile", "data": "...", "args": [...]}
                
        Returns:
            ToolResult containing execution results
        """
        try:
            # Parse input
            if isinstance(input_data, str):
                # Assume it's WAT code to compile and run
                return await self._compile_and_run_wat(input_data.strip())
            elif isinstance(input_data, dict):
                action = input_data.get("action", "run")
                data = input_data.get("data", "")
                args = input_data.get("args", [])
                
                if action == "compile":
                    return await self._compile_wat_to_wasm(data)
                elif action == "decompile":
                    return await self._decompile_wasm_to_wat(data)
                elif action == "run":
                    if data.startswith("(module"):  # WAT format
                        return await self._compile_and_run_wat(data, args)
                    else:  # Assume binary WASM (base64 encoded)
                        return await self._run_wasm_binary(data, args)
                else:
                    return ToolResult(
                        success=False,
                        content=f"❌ Unknown action: {action}. Use 'run', 'compile', or 'decompile'"
                    )
            else:
                return ToolResult(
                    success=False,
                    content="❌ Input must be string (WAT code) or dict with action/data"
                )
                
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                content=f"❌ WASM operation timed out after {self.config.timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Error executing WASM operation: {str(e)}"
            )
    
    async def _compile_and_run_wat(self, wat_code: str, args: List[str] = None) -> ToolResult:
        """Compile WAT to WASM and execute it."""
        if not wat_code.strip():
            return ToolResult(
                success=False,
                content="❌ No WebAssembly Text (WAT) code provided"
            )
        
        # First compile WAT to WASM
        compile_result = await self._compile_wat_to_wasm(wat_code)
        if not compile_result.success:
            return compile_result
        
        # Extract WASM binary from compile result
        wasm_binary = compile_result.metadata.get("wasm_binary", "")
        if not wasm_binary:
            return ToolResult(
                success=False,
                content="❌ Failed to extract WASM binary from compilation"
            )
        
        # Run the compiled WASM
        return await self._run_wasm_binary(wasm_binary, args or [])
    
    async def _compile_wat_to_wasm(self, wat_code: str) -> ToolResult:
        """Compile WebAssembly Text to binary format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.wat', delete=False) as wat_file:
            wat_file.write(wat_code)
            wat_path = wat_file.name
        
        with tempfile.NamedTemporaryFile(suffix='.wasm', delete=False) as wasm_file:
            wasm_path = wasm_file.name
        
        try:
            # Compile WAT to WASM
            process = await asyncio.create_subprocess_exec(
                self.config.wat2wasm_executable,
                wat_path,
                '-o', wasm_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout
            )
            
            if process.returncode == 0:
                # Read compiled WASM binary
                with open(wasm_path, 'rb') as f:
                    wasm_binary = base64.b64encode(f.read()).decode('utf-8')
                
                return ToolResult(
                    success=True,
                    content=f"✅ WAT compiled to WASM successfully\\n📦 Binary size: {len(base64.b64decode(wasm_binary))} bytes",
                    metadata={
                        "wasm_binary": wasm_binary,
                        "binary_size": len(base64.b64decode(wasm_binary))
                    }
                )
            else:
                error_msg = stderr.decode('utf-8', errors='replace')
                return ToolResult(
                    success=False,
                    content=f"❌ WAT compilation failed:\\n```\\n{error_msg}```"
                )
                
        finally:
            # Clean up temporary files
            for path in [wat_path, wasm_path]:
                try:
                    os.unlink(path)
                except:
                    pass
    
    async def _decompile_wasm_to_wat(self, wasm_binary_b64: str) -> ToolResult:
        """Decompile WASM binary to WebAssembly Text format."""
        try:
            wasm_binary = base64.b64decode(wasm_binary_b64)
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Invalid base64 WASM binary: {str(e)}"
            )
        
        with tempfile.NamedTemporaryFile(suffix='.wasm', delete=False) as wasm_file:
            wasm_file.write(wasm_binary)
            wasm_path = wasm_file.name
        
        try:
            # Decompile WASM to WAT
            process = await asyncio.create_subprocess_exec(
                self.config.wasm2wat_executable,
                wasm_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout
            )
            
            if process.returncode == 0:
                wat_code = stdout.decode('utf-8', errors='replace')
                return ToolResult(
                    success=True,
                    content=f"✅ WASM decompiled to WAT successfully\\n\\n```wat\\n{wat_code}```",
                    metadata={
                        "wat_code": wat_code,
                        "original_size": len(wasm_binary)
                    }
                )
            else:
                error_msg = stderr.decode('utf-8', errors='replace')
                return ToolResult(
                    success=False,
                    content=f"❌ WASM decompilation failed:\\n```\\n{error_msg}```"
                )
                
        finally:
            try:
                os.unlink(wasm_path)
            except:
                pass
    
    async def _run_wasm_binary(self, wasm_binary_b64: str, args: List[str]) -> ToolResult:
        """Execute a WASM binary."""
        try:
            wasm_binary = base64.b64decode(wasm_binary_b64)
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Invalid base64 WASM binary: {str(e)}"
            )
        
        with tempfile.NamedTemporaryFile(suffix='.wasm', delete=False) as wasm_file:
            wasm_file.write(wasm_binary)
            wasm_path = wasm_file.name
        
        try:
            # Prepare wasmtime command
            cmd = [self.config.wasmtime_executable]
            
            # Add security and resource limits
            cmd.extend([
                '--max-memory', self.config.max_memory,
                '--disable-all-features'  # Disable potentially dangerous features
            ])
            
            # Enable WASI if configured
            if self.config.enable_wasi:
                cmd.extend(['--wasi-inherit-environ', '--wasi-inherit-args'])
            
            # Add the WASM file
            cmd.append(wasm_path)
            
            # Add any arguments
            cmd.extend(args)
            
            # Execute WASM module
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tempfile.gettempdir()
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout
            )
            
            output = stdout.decode('utf-8', errors='replace')
            error_output = stderr.decode('utf-8', errors='replace')
            
            # Truncate output if too long
            if len(output) > self.config.max_output_size:
                output = output[:self.config.max_output_size] + "\\n... (output truncated)"
            
            if process.returncode == 0:
                content = f"🚀 WASM Execution Result:\\n```\\n{output}```"
                if error_output:
                    content += f"\\n\\n⚠️ Stderr:\\n```\\n{error_output}```"
                
                return ToolResult(
                    success=True,
                    content=content,
                    metadata={
                        "return_code": process.returncode,
                        "output_length": len(stdout),
                        "has_stderr": bool(error_output),
                        "binary_size": len(wasm_binary)
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    content=f"❌ WASM execution failed (exit code {process.returncode}):\\n```\\n{error_output}```"
                )
                
        finally:
            try:
                os.unlink(wasm_path)
            except:
                pass
    
    async def validate_wasm_tools(self) -> Dict[str, bool]:
        """Validate that all required WASM tools are installed."""
        tools = {
            'wasmtime': self.config.wasmtime_executable,
            'wat2wasm': self.config.wat2wasm_executable,
            'wasm2wat': self.config.wasm2wat_executable
        }
        
        results = {}
        for name, executable in tools.items():
            try:
                process = await asyncio.create_subprocess_exec(
                    executable,
                    '--version',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5)
                results[name] = process.returncode == 0
            except:
                results[name] = False
        
        return results


class WasmPlugin:
    """WASM plugin for OpenAgent."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = WasmConfig(**(config or {}))
    
    def get_tools(self) -> list[BaseTool]:
        """Return the tools provided by this plugin."""
        return [WasmTool(self.config)]
    
    def get_info(self) -> Dict[str, Any]:
        """Return plugin information."""
        return {
            "name": "wasm",
            "version": "1.0.0",
            "description": "Execute WebAssembly modules and convert between WAT/WASM formats",
            "author": "OpenAgent Contributors",
            "tags": ["webassembly", "wasm", "wat", "runtime", "compilation"],
            "requirements": ["wasmtime", "wat2wasm", "wasm2wat"],
            "permissions": [
                "execute_wasm",
                "temporary_files",
                "process_execution"
            ]
        }


# Factory function for plugin loading
def create_plugin(config: Optional[Dict[str, Any]] = None) -> WasmPlugin:
    """Create and return a WasmPlugin instance."""
    return WasmPlugin(config)


# Example WASM programs in WAT format
EXAMPLE_PROGRAMS = {
    "hello_world": """
(module
  (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "Hello from WASM!\\n")
  
  (func $main (export "_start")
    ;; Write "Hello from WASM!" to stdout
    (i32.store (i32.const 24) (i32.const 0))   ;; iov.iov_base
    (i32.store (i32.const 28) (i32.const 17))  ;; iov.iov_len (length of string)
    
    (call $fd_write
      (i32.const 1)   ;; stdout file descriptor
      (i32.const 24)  ;; iovec array
      (i32.const 1)   ;; iovec count
      (i32.const 32)  ;; bytes written
    )
    drop
  )
)
""",
    
    "fibonacci": """
(module
  (func $fibonacci (export "fibonacci") (param $n i32) (result i32)
    (if (result i32)
      (i32.lt_s (local.get $n) (i32.const 2))
      (then (local.get $n))
      (else
        (i32.add
          (call $fibonacci (i32.sub (local.get $n) (i32.const 1)))
          (call $fibonacci (i32.sub (local.get $n) (i32.const 2)))
        )
      )
    )
  )
  
  (func $main (export "_start")
    ;; Calculate fibonacci(10) and store result
    (call $fibonacci (i32.const 10))
    drop
  )
)
""",
    
    "math_operations": """
(module
  (func $add (export "add") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add
  )
  
  (func $multiply (export "multiply") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.mul
  )
  
  (func $factorial (export "factorial") (param $n i32) (result i32)
    (if (result i32)
      (i32.le_s (local.get $n) (i32.const 1))
      (then (i32.const 1))
      (else
        (i32.mul
          (local.get $n)
          (call $factorial (i32.sub (local.get $n) (i32.const 1)))
        )
      )
    )
  )
)
""",
    
    "memory_operations": """
(module
  (memory (export "memory") 1)
  
  (func $store_and_load (export "store_and_load") (result i32)
    ;; Store value 42 at memory address 0
    (i32.store (i32.const 0) (i32.const 42))
    
    ;; Load and return the value
    (i32.load (i32.const 0))
  )
  
  (func $sum_array (export "sum_array") (param $start i32) (param $length i32) (result i32)
    (local $sum i32)
    (local $i i32)
    
    (loop $loop
      (local.set $sum
        (i32.add
          (local.get $sum)
          (i32.load (local.get $start))
        )
      )
      
      (local.set $start (i32.add (local.get $start) (i32.const 4)))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      
      (br_if $loop (i32.lt_s (local.get $i) (local.get $length)))
    )
    
    local.get $sum
  )
)
"""
}
