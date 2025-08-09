# WASM Plugin for OpenAgent

Execute WebAssembly modules and convert between WAT/WASM formats with comprehensive tooling support.

## Features

- 🚀 **WebAssembly Execution** - Run WASM modules using Wasmtime runtime
- 🔄 **Format Conversion** - Convert between WAT (text) and WASM (binary) formats
- 🛡️ **Security Controls** - Sandboxed execution with memory and resource limits
- 📊 **WASI Support** - WebAssembly System Interface for system interactions
- ⏱️ **Timeout Protection** - Configurable execution timeouts
- 🔧 **Tool Validation** - Automatic validation of required WASM toolchain
- 📈 **Performance Metrics** - Execution time and resource usage tracking

## Installation

### Prerequisites

Install the required WebAssembly tools:

#### 1. Wasmtime (WASM Runtime)

```bash
# Quick install (Linux/macOS)
curl https://wasmtime.dev/install.sh -sSf | bash

# Or download from releases
# https://github.com/bytecodealliance/wasmtime/releases
```

#### 2. WABT (WebAssembly Binary Toolkit)

```bash
# Ubuntu/Debian
sudo apt-get install wabt

# macOS  
brew install wabt

# Windows
# Download from: https://github.com/WebAssembly/wabt/releases

# Or build from source
git clone --recursive https://github.com/WebAssembly/wabt
cd wabt
mkdir build
cd build
cmake ..
cmake --build .
```

### Plugin Installation

```bash
# List available plugins
openagent plugin list

# Install the WASM plugin
openagent plugin install wasm

# Verify installation
openagent plugin info wasm
```

### Verify Tools Installation

```bash
# Check if all tools are available
wasmtime --version
wat2wasm --version  
wasm2wat --version
```

## Configuration

Configure the plugin with these options:

```yaml
wasmtime_executable: "wasmtime"    # Path to Wasmtime runtime
wat2wasm_executable: "wat2wasm"    # Path to WAT compiler
wasm2wat_executable: "wasm2wat"    # Path to WASM decompiler
timeout: 30                        # Execution timeout (seconds)
max_output_size: 10000            # Maximum output size (characters)
max_memory: "10MB"                # Memory limit for WASM modules
enable_wasi: true                 # Enable WASI support
allowed_imports: ["wasi_snapshot_preview1"]  # Allowed import modules
```

## Usage Examples

### Basic WAT Execution

```bash
# Simple WebAssembly Text execution
openagent run "execute this WASM code: (module (func (export \"test\") (result i32) i32.const 42))"
```

### Hello World with WASI

```wat
(module
  (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "Hello from WASM!\n")
  
  (func $main (export "_start")
    ;; Write "Hello from WASM!" to stdout
    (i32.store (i32.const 24) (i32.const 0))   ;; iov.iov_base
    (i32.store (i32.const 28) (i32.const 17))  ;; iov.iov_len
    
    (call $fd_write
      (i32.const 1)   ;; stdout file descriptor
      (i32.const 24)  ;; iovec array
      (i32.const 1)   ;; iovec count
      (i32.const 32)  ;; bytes written
    )
    drop
  )
)
```

### Math Operations Module

```wat
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
```

### Memory Operations

```wat
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
```

## Advanced Usage

### Compilation Only

Compile WAT to WASM without execution:

```json
{
  "action": "compile",
  "data": "(module (func (export \"test\") (result i32) i32.const 42))"
}
```

### Decompilation

Decompile WASM binary back to WAT:

```json
{
  "action": "decompile", 
  "data": "AGFzbQEAAAABBQFgAAF/AwIBAAcIAQR0ZXN0AAAKBgEEAEEqCw=="
}
```

### Execute with Arguments

Run WASM module with command-line arguments:

```json
{
  "action": "run",
  "data": "(module ...)",
  "args": ["arg1", "arg2", "--flag"]
}
```

## Security Features

### Sandboxing

WebAssembly provides inherent sandboxing, and the plugin adds additional security layers:

- **Memory limits** - Configurable maximum memory allocation
- **Execution timeout** - Prevents infinite loops and runaway processes
- **Import restrictions** - Only allowed import modules are permitted
- **Process isolation** - Each execution runs in a separate process
- **Resource monitoring** - Track and limit system resource usage

### WASI Controls

When WASI is enabled, the plugin provides controlled access to system interfaces:

- **File system access** - Limited to temporary directories
- **Environment variables** - Controlled inheritance
- **Standard I/O** - Managed stdin/stdout/stderr access
- **Network access** - Disabled by default

### Resource Limits

```yaml
# Example security configuration
max_memory: "5MB"          # Limit memory usage
timeout: 15                # Quick timeout for untrusted code
enable_wasi: false         # Disable system access
allowed_imports: []        # No imports allowed
```

## Compilation from Source Languages

### Compiling C to WASM

```c
// hello.c
#include <stdio.h>

int main() {
    printf("Hello from C compiled to WASM!\n");
    return 0;
}
```

```bash
# Compile with clang (requires wasi-sdk)
clang --target=wasm32-wasi -o hello.wasm hello.c

# Convert to base64 for plugin usage
base64 hello.wasm
```

### Compiling Rust to WASM

```rust
// hello.rs
fn main() {
    println!("Hello from Rust compiled to WASM!");
}
```

```bash
# Add WASM target
rustup target add wasm32-wasi

# Compile
rustc --target wasm32-wasi hello.rs -o hello.wasm

# Use with plugin
base64 hello.wasm
```

## Error Handling

The plugin provides comprehensive error reporting:

```bash
❌ WAT compilation failed:
```
hello.wat:3:5: error: unexpected token "invalid_instruction"
    invalid_instruction
    ^^^^^^^^^^^^^^^^^^^
```

```bash
❌ WASM execution failed (exit code 1):
```
Error: failed to run main module `module.wasm`

Caused by:
    0: WebAssembly runtime error: unreachable
    1: wasm trap: unreachable
```

## API Integration

### Direct Tool Usage

```python
from openagent.plugins.wasm import WasmTool, WasmConfig

# Configure the tool
config = WasmConfig(
    timeout=60,
    max_memory="50MB",
    enable_wasi=True
)

# Create and use the tool
wasm_tool = WasmTool(config)

# Execute WAT code
result = await wasm_tool.execute("""
(module
  (func (export "test") (result i32)
    i32.const 42
  )
)
""")

print(result.content)
```

### Agent Integration

```python
from openagent.core.agent import Agent
from openagent.plugins.wasm import create_plugin

# Create agent with WASM plugin
wasm_plugin = create_plugin({
    "max_memory": "20MB",
    "enable_wasi": True
})

agent = Agent(
    name="WasmAgent", 
    tools=wasm_plugin.get_tools()
)

# Use the agent
response = await agent.process_message(
    "Create a WASM module that calculates fibonacci numbers"
)
```

### Tool Validation

```python
from openagent.plugins.wasm import WasmTool

tool = WasmTool()
validation = await tool.validate_wasm_tools()

print("Tool availability:")
for tool_name, available in validation.items():
    status = "✅" if available else "❌"
    print(f"{status} {tool_name}")
```

## Performance Considerations

### Optimization Tips

1. **Memory Management**: Use appropriate memory limits
2. **Compilation**: Pre-compile frequently used modules
3. **Caching**: Cache compiled WASM binaries when possible
4. **Resource Monitoring**: Monitor execution metrics

### Benchmarking

The plugin provides execution metrics:

```python
result = await wasm_tool.execute(wat_code)
metadata = result.metadata

print(f"Execution time: {metadata.get('execution_time', 0):.2f}s")
print(f"Binary size: {metadata.get('binary_size', 0)} bytes")
print(f"Output length: {metadata.get('output_length', 0)} characters")
```

## Troubleshooting

### Common Issues

**"wasmtime: command not found"**
```bash
# Install Wasmtime
curl https://wasmtime.dev/install.sh -sSf | bash
source ~/.bashrc
```

**"wat2wasm: command not found"**
```bash
# Install WABT tools
sudo apt-get install wabt  # Ubuntu/Debian
brew install wabt          # macOS
```

**Memory allocation errors**
```yaml
# Increase memory limit
max_memory: "50MB"  # Increase from default 10MB
```

**WASI import errors**
```yaml
# Enable WASI support
enable_wasi: true
allowed_imports: ["wasi_snapshot_preview1"]
```

### Debug Mode

Enable verbose output for debugging:

```yaml
# In plugin configuration
debug: true

# Or via environment
export WASMTIME_LOG=debug
```

## Examples Repository

The plugin includes example WASM programs:

- `hello_world` - Basic WASI output example
- `fibonacci` - Recursive function implementation  
- `math_operations` - Exported function examples
- `memory_operations` - Linear memory manipulation

Access examples programmatically:

```python
from openagent.plugins.wasm import EXAMPLE_PROGRAMS
print(EXAMPLE_PROGRAMS['fibonacci'])
```

## WebAssembly Resources

### Learning WebAssembly

- [WebAssembly.org](https://webassembly.org/)
- [MDN WebAssembly Guide](https://developer.mozilla.org/en-US/docs/WebAssembly)
- [WebAssembly Text Format Spec](https://webassembly.github.io/spec/core/text/index.html)

### Tools and Runtimes

- [Wasmtime](https://wasmtime.dev/) - Fast and secure WebAssembly runtime
- [WABT](https://github.com/WebAssembly/wabt) - WebAssembly Binary Toolkit
- [wasi-sdk](https://github.com/WebAssembly/wasi-sdk) - WASI-enabled WebAssembly C/C++ toolchain

### Compilers

- [Emscripten](https://emscripten.org/) - C/C++ to WebAssembly compiler
- [wasm-pack](https://rustwasm.github.io/wasm-pack/) - Rust to WebAssembly workflow
- [AssemblyScript](https://www.assemblyscript.org/) - TypeScript-like language for WebAssembly

## Contributing

To contribute to the WASM plugin:

1. Fork the OpenAgent repository
2. Set up WebAssembly development environment
3. Add comprehensive tests for new features
4. Ensure security and sandboxing are maintained
5. Update documentation and examples
6. Submit a pull request

## License

This plugin is licensed under the MIT License as part of the OpenAgent project.
