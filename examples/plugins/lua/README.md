# Lua Plugin for OpenAgent

Execute Lua scripts and code snippets directly through OpenAgent with built-in security restrictions and sandboxing.

## Features

- ✅ **Safe Lua Execution** - Sandboxed environment with configurable security
- 🔒 **Security Controls** - Disable dangerous operations like file I/O by default  
- ⏱️ **Timeout Protection** - Configurable execution timeouts
- 📏 **Output Limiting** - Prevent excessive output with size limits
- 🛡️ **Error Handling** - Comprehensive error reporting and graceful failures
- 📊 **Execution Metrics** - Track execution time and output statistics

## Installation

### Prerequisites

First, ensure Lua is installed on your system:

```bash
# Ubuntu/Debian
sudo apt-get install lua5.4

# macOS
brew install lua

# Windows
# Download from https://www.lua.org/download.html
```

### Plugin Installation

The plugin is included with OpenAgent. To enable it:

```bash
# List available plugins
openagent plugin list

# Install the Lua plugin
openagent plugin install lua

# Verify installation
openagent plugin info lua
```

## Configuration

The plugin can be configured with the following options:

```yaml
lua_executable: "lua"        # Path to Lua interpreter
timeout: 30                  # Execution timeout (seconds)
max_output_size: 10000      # Maximum output size (characters)  
allow_file_io: false        # Enable/disable file I/O operations
enable_debug: true          # Enable debug output
```

## Usage Examples

### Basic Lua Execution

```bash
# Simple script execution
openagent run "execute this lua code: print('Hello from Lua!')"
```

### Math Operations

```lua
-- Basic calculations
local a, b = 15, 7
print("Addition: " .. a .. " + " .. b .. " = " .. (a + b))
print("Multiplication: " .. a .. " * " .. b .. " = " .. (a * b))
print("Division: " .. a .. " / " .. b .. " = " .. (a / b))
```

### Table Operations  

```lua
-- Working with Lua tables
local languages = {"Python", "Lua", "JavaScript", "Rust"}

print("Programming languages:")
for i, lang in ipairs(languages) do
    print(i .. ". " .. lang)
end

-- Key-value tables
local developer = {
    name = "Alice",
    language = "Lua",
    experience = 5
}

print("\nDeveloper info:")
for key, value in pairs(developer) do
    print(key .. ": " .. tostring(value))
end
```

### Functions and Logic

```lua
-- Recursive factorial function
function factorial(n)
    if n <= 1 then
        return 1
    else
        return n * factorial(n - 1)
    end
end

print("Factorial calculations:")
for i = 1, 8 do
    print("factorial(" .. i .. ") = " .. factorial(i))
end
```

### String Processing

```lua
-- String manipulation
local text = "OpenAgent Lua Plugin"
print("Original: " .. text)
print("Length: " .. #text)
print("Uppercase: " .. string.upper(text))
print("Words: " .. string.gsub(text, " ", ", "))

-- Pattern matching
local words = {}
for word in string.gmatch(text, "%w+") do
    table.insert(words, word)
end
print("Word list: " .. table.concat(words, " | "))
```

## Security Features

### Sandboxing

By default, the plugin runs Lua code in a sandboxed environment with the following restrictions:

- **File I/O disabled** - `io.*`, `loadfile`, `dofile` functions are blocked
- **System access restricted** - `os.execute`, `os.remove` functions are blocked  
- **Debug access limited** - `debug.debug` function is blocked
- **Module loading controlled** - Restricted `require()` usage

### Safe Mode

When `allow_file_io` is set to `false` (default), the plugin automatically wraps user code with safety restrictions:

```lua
-- Auto-generated safety wrapper
io = nil
os.execute = function() error("os.execute is disabled in sandbox mode") end
os.remove = function() error("os.remove is disabled in sandbox mode") end
loadfile = function() error("loadfile is disabled in sandbox mode") end
dofile = function() error("dofile is disabled in sandbox mode") end

-- Your code runs here
```

### Resource Limits

- **Execution timeout** - Scripts are terminated after the configured timeout
- **Output size limits** - Large outputs are truncated to prevent memory issues
- **Memory isolation** - Each execution runs in a separate process

## Error Handling

The plugin provides detailed error information:

```bash
❌ Lua execution failed:
```
stdin:3: attempt to call a nil value (global 'nonexistent_function')
stack traceback:
    stdin:3: in main chunk
    [C]: in ?
```
```

## API Integration

### Direct Tool Usage

```python
from openagent.plugins.lua import LuaTool, LuaConfig

# Configure the tool
config = LuaConfig(
    timeout=60,
    allow_file_io=True,  # Enable for trusted scripts
    max_output_size=50000
)

# Create and use the tool
lua_tool = LuaTool(config)
result = await lua_tool.execute("""
    print("Direct API usage")
    return 42
""")

print(result.content)
```

### Agent Integration

```python
from openagent.core.agent import Agent
from openagent.plugins.lua import create_plugin

# Create agent with Lua plugin
lua_plugin = create_plugin({"timeout": 45})
agent = Agent(
    name="LuaAgent",
    tools=lua_plugin.get_tools()
)

# Use the agent
response = await agent.process_message(
    "Calculate the sum of numbers from 1 to 100 using Lua"
)
```

## Troubleshooting

### Common Issues

**"lua: command not found"**
```bash
# Install Lua
sudo apt-get install lua5.4  # Ubuntu/Debian
brew install lua             # macOS
```

**Execution timeout errors**
```bash
# Increase timeout in configuration
timeout: 60  # Increase from default 30 seconds
```

**Output truncated**
```bash  
# Increase output limit
max_output_size: 50000  # Increase from default 10000 characters
```

### Debug Mode

Enable debug output for troubleshooting:

```yaml
enable_debug: true
```

This provides additional information about:
- Execution time measurements
- Security wrapper application
- Process creation and cleanup
- Error context and stack traces

## Examples Repository

The plugin includes several example scripts in `EXAMPLE_SCRIPTS`:

- `hello_world` - Basic output and version info
- `math_operations` - Arithmetic calculations  
- `table_operations` - Working with arrays and dictionaries
- `functions` - Function definition and recursion
- `string_operations` - Text processing and pattern matching

Access these examples programmatically:

```python
from openagent.plugins.lua import EXAMPLE_SCRIPTS
print(EXAMPLE_SCRIPTS['fibonacci'])
```

## Contributing

To contribute to the Lua plugin:

1. Fork the OpenAgent repository
2. Create your feature branch
3. Add tests for new functionality  
4. Ensure security considerations are addressed
5. Submit a pull request

## License

This plugin is licensed under the MIT License as part of the OpenAgent project.
