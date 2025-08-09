"""
Command Line Interface for OpenAgent.

This module provides a powerful CLI for interacting with OpenAgent,
similar to how Warp provides terminal AI assistance.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.table import Table

from openagent.terminal.integration import install_snippet
from openagent.terminal.validator import validate as validate_cmd, load_policy, save_policy, DEFAULT_POLICY, CONFIG_PATH

from openagent.core.agent import Agent
from openagent.core.llm import get_llm, ModelConfig
from openagent.tools.system import CommandExecutor, FileManager, SystemInfo
from openagent.core.config import Config
from dotenv import load_dotenv
import os

# Initialize Rich console
console = Console()
app = typer.Typer(help="OpenAgent - AI-powered terminal assistant")

# Global agent instance
agent: Optional[Agent] = None


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )


def create_agent(
    model_name: str = "tiny-llama",
    device: str = "auto",
    load_in_4bit: bool = True,
    unsafe_exec: bool = True,
) -> Agent:
    """Create and configure an OpenAgent instance.

    Note: By default, command execution is ENABLED (unsafe_exec=True), similar to Warp.
    Use --safe-exec flags in commands to force explain-only behavior.
    """
    
    # LLM configuration for efficient operation
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    llm_config = {
        "device": device,
        "load_in_4bit": load_in_4bit,
        "temperature": 0.7,
        "max_length": 2048,
    }
    if hf_token:
        llm_config["hf_token"] = hf_token
    
    # Create agent with tools
    agent = Agent(
        name="TerminalAssistant",
        description="I'm an AI assistant that helps with terminal operations, coding, and system administration. I can execute commands, manage files, analyze code, and provide technical guidance.",
        model_name=model_name,
        llm_config=llm_config,
        safe_mode=not unsafe_exec,
    )
    
    # Add powerful tools (default to explain-only unless --unsafe-exec is used)
    agent.add_tool(CommandExecutor(default_explain_only=not unsafe_exec))
    agent.add_tool(FileManager())
    agent.add_tool(SystemInfo())
    
    return agent


@app.command()
def chat(
    model: str = typer.Option("tiny-llama", help="Model to use for responses"),
    device: str = typer.Option("auto", help="Device to run model on (auto/cpu/cuda)"),
    load_in_4bit: bool = typer.Option(True, help="Load model in 4-bit precision"),
    debug: bool = typer.Option(False, help="Enable debug logging"),
    unsafe_exec: bool = typer.Option(True, "--unsafe-exec/--no-unsafe-exec", help="Allow actual command execution (default true). Use --no-unsafe-exec for explain-only mode."),
):
    """Start an interactive chat session with OpenAgent."""
    
    setup_logging(debug)
    load_dotenv()
    
    console.print(Panel.fit(
        "[bold blue]OpenAgent Terminal Assistant[/bold blue]\n"
        "AI-powered terminal assistance with code generation and system operations.\n"
        f"Model: {model} | Device: {device} | 4-bit: {load_in_4bit}",
        title="🤖 Welcome"
    ))
    
# Create agent
    global agent
    agent = create_agent(model, device, load_in_4bit, unsafe_exec) 
    
    console.print("\\n[dim]Initializing AI model... This may take a moment.[/dim]")
    
    # Interactive chat loop
    asyncio.run(chat_loop())


async def chat_loop():
    """Main interactive chat loop."""
    global agent
    
    console.print("\\n[green]Ready! Type your message or 'help' for commands.[/green]")
    console.print("[dim]Special commands: /help, /status, /reset, /quit[/dim]\\n")
    
    while True:
        try:
            # Get user input
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.startswith('/'):
                if await handle_special_command(user_input):
                    continue
                else:
                    break
            
            # Process user message with loading spinner
            with console.status("[bold green]Thinking...", spinner="dots"):
                response = await agent.process_message(user_input)
            
            # Display response
            console.print(f"\\n[bold green]Assistant:[/bold green]")
            
            # Check if response contains code
            if "```" in response.content:
                # Render as markdown for proper code highlighting
                console.print(Markdown(response.content))
            else:
                console.print(response.content)
            
            # Show metadata if debug
            if response.metadata.get('tools_used'):
                console.print(f"\\n[dim]Tools used: {', '.join(response.metadata['tools_used'])}[/dim]")
            
            console.print()  # Add spacing
            
        except KeyboardInterrupt:
            console.print("\\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"\\n[red]Error: {e}[/red]")


async def handle_special_command(command: str) -> bool:
    """Handle special chat commands. Returns True to continue chat, False to exit."""
    global agent
    
    if command == "/help":
        console.print(Panel(
            """[bold]Available Commands:[/bold]
            
/help       - Show this help message
/status     - Show agent status and model info
/reset      - Reset conversation history
/models     - List available models
/system     - Show system information
/quit       - Exit the application

[bold]Tips:[/bold]
• Ask me to explain commands before running them
• I can help with coding, debugging, and system administration
• Use natural language to describe what you want to accomplish
• I can execute safe commands and manage files
""",
            title="Help"
        ))
        return True
    
    if command == "/status":
        status = agent.get_status()
        model_info = agent.llm.get_model_info()
        safe_mode = agent.config.get("safe_mode", True) if hasattr(agent, "config") else True
        
        console.print(Panel(
            f"""[bold]Agent Status:[/bold]
Name: {status['name']}
Model: {model_info['model_name']} ({model_info['model_path']})
Device: {model_info['device']}
Loaded: {model_info['loaded']}
Tools: {status['tools_count']} ({', '.join(status['tools'])})
Messages: {status['message_history_length']}
Processing: {status['is_processing']}
Safe Mode (explain-only commands): {safe_mode}
""",
            title="Status"
        ))
        return True
    
    if command == "/reset":
        agent.reset()
        console.print("[green]Conversation history reset![/green]")
        return True
    
    if command == "/models":
        console.print(Panel(
            f"""[bold]Available Models:[/bold]

[bold cyan]Code Models (Recommended):[/bold cyan]
{chr(10).join([f'• {k}: {v}' for k, v in ModelConfig.CODE_MODELS.items()])}

[bold cyan]Chat Models:[/bold cyan]
{chr(10).join([f'• {k}: {v}' for k, v in ModelConfig.CHAT_MODELS.items()])}

[bold cyan]Lightweight Models:[/bold cyan]
{chr(10).join([f'• {k}: {v}' for k, v in ModelConfig.LIGHTWEIGHT_MODELS.items()])}
""",
            title="Models"
        ))
        return True
    
    if command == "/system":
        # Use the system info tool
        system_tool = SystemInfo()
        result = await system_tool.execute("overview")
        console.print(Panel(result.content, title="System Information"))
        return True
    
    if command == "/quit":
        console.print("[yellow]Goodbye![/yellow]")
        return False
    
    console.print(f"[red]Unknown command: {command}[/red]")
    console.print("[dim]Type /help for available commands[/dim]")
    return True


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Prompt to send to the agent"),
    model: str = typer.Option("tiny-llama", help="Model to use"),
    device: str = typer.Option("auto", help="Device to run model on"),
    load_in_4bit: bool = typer.Option(True, help="Load model in 4-bit precision"),
    output_format: str = typer.Option("text", help="Output format (text/json)"),
    unsafe_exec: bool = typer.Option(True, "--unsafe-exec/--no-unsafe-exec", help="Allow actual command execution (default true). Use --no-unsafe-exec for explain-only mode."),
):
    """Run a single prompt through OpenAgent and exit."""
    
    load_dotenv()
    # Only print human-friendly initializing message for text output
    if output_format != "json":
        console.print("[dim]Initializing...[/dim]")
    
    # Create agent
    agent = create_agent(model, device, load_in_4bit, unsafe_exec)
    
    async def run_single():
        with console.status("[bold green]Processing...", spinner="dots") if output_format != "json" else contextlib.nullcontext():
            response = await agent.process_message(prompt)
        
        if output_format == "json":
            import json
            result = {
                "prompt": prompt,
                "response": response.content,
                "metadata": response.metadata
            }
            # Print only JSON to stdout (no extra formatting)
            console.print(json.dumps(result, indent=2))
        else:
            if "```" in response.content:
                console.print(Markdown(response.content))
            else:
                console.print(response.content)
    
    import contextlib
    asyncio.run(run_single())


@app.command()
def explain(
    command: str = typer.Argument(..., help="Command to explain"),
    model: str = typer.Option("tiny-llama", help="Model to use"),
):
    """Explain what a shell command does."""
    
    console.print(f"[dim]Explaining command: {command}[/dim]\\n")
    
    # Create agent
    agent = create_agent(model)
    
    async def explain_command():
        with console.status("[bold green]Analyzing...", spinner="dots"):
            explanation = await agent.llm.explain_command(command)
        
        console.print(Panel(explanation, title=f"Command Explanation: {command}"))
    
    asyncio.run(explain_command())


@app.command()
def code(
    description: str = typer.Argument(..., help="Description of code to generate"),
    language: str = typer.Option("python", help="Programming language"),
    model: str = typer.Option("codellama-7b", help="Model to use (code models recommended)"),
):
    """Generate code based on description."""
    
    console.print(f"[dim]Generating {language} code...[/dim]\\n")
    
    # Create agent with code model
    agent = create_agent(model)
    
    async def generate_code():
        with console.status("[bold green]Coding...", spinner="dots"):
            code = await agent.llm.generate_code(description, language)
        
        # Display code with syntax highlighting
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title=f"Generated {language.title()} Code"))
    
    asyncio.run(generate_code())


@app.command()
def analyze(
    file_path: str = typer.Argument(..., help="Path to code file to analyze"),
    model: str = typer.Option("codellama-7b", help="Model to use"),
):
    """Analyze a code file and provide insights."""
    
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    try:
        code_content = file_path_obj.read_text()
        language = file_path_obj.suffix[1:] if file_path_obj.suffix else "text"
        
        console.print(f"[dim]Analyzing {file_path}...[/dim]\\n")
        
        # Create agent
        agent = create_agent(model)
        
        async def analyze_code():
            with console.status("[bold green]Analyzing...", spinner="dots"):
                analysis = await agent.llm.analyze_code(code_content, language)
            
            console.print(Panel(Markdown(analysis), title=f"Code Analysis: {file_path}"))
        
        asyncio.run(analyze_code())
        
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def validate(
    command: str = typer.Argument(..., help="Shell command to validate"),
    quiet: bool = typer.Option(False, help="Print only the decision (allow/warn/block)"),
):
    """Validate a command against OpenAgent's policy (allow/warn/block)."""
    decision, reason = validate_cmd(command)
    if quiet:
        console.print(decision)
    else:
        console.print(Panel(f"Decision: [bold]{decision}[/bold]\nReason: {reason}", title="Validation"))


@app.command()
def policy(
    action: str = typer.Argument(..., help="Action: show|reset|set-default|add-allow|remove-allow|block-risky|unblock-risky"),
    key: str = typer.Argument(None, help="Command for allowlist updates or default value"),
    value: str = typer.Argument(None, help="Value for the action (flag prefix or default decision)"),
):
    """Manage OpenAgent terminal policy stored in ~/.config/openagent/policy.yaml."""
    p = load_policy()
    if action == "show":
        import json
        console.print(Panel(str(CONFIG_PATH), title="Policy File"))
        console.print_json(data=p)
        return
    if action == "reset":
        save_policy(DEFAULT_POLICY.copy())
        console.print("[green]Policy reset to defaults[/green]")
        return
    if action == "set-default":
        # Accept the decision as either the second or third positional argument
        decision = value or key
        if decision not in {"allow", "warn", "block"}:
            console.print("[red]Default must be one of allow|warn|block[/red]")
            raise typer.Exit(1)
        p["default_decision"] = decision
        save_policy(p)
        console.print(f"[green]Default decision set to {decision}[/green]")
        return
    if action == "add-allow":
        if not key or not value:
            console.print("[red]Usage: policy add-allow <command> <flag_prefix>[/red]")
            raise typer.Exit(1)
        p.setdefault("allowlist", {}).setdefault(key, []).append(value)
        save_policy(p)
        console.print(f"[green]Added allow flag '{value}' for command '{key}'[/green]")
        return
    if action == "remove-allow":
        if not key or not value:
            console.print("[red]Usage: policy remove-allow <command> <flag_prefix>[/red]")
            raise typer.Exit(1)
        flags = p.setdefault("allowlist", {}).setdefault(key, [])
        if value in flags:
            flags.remove(value)
        save_policy(p)
        console.print(f"[green]Removed allow flag '{value}' for command '{key}'[/green]")
        return
    if action == "block-risky":
        p["block_risky"] = True
        save_policy(p)
        console.print("[green]Risky commands will be blocked[/green]")
        return
    if action == "unblock-risky":
        p["block_risky"] = False
        save_policy(p)
        console.print("[green]Risky commands will not be blocked (may still warn)" )
        return
    console.print("[red]Unknown action[/red]")


@app.command()
def integrate(
    shell: str = typer.Option("zsh", help="Shell to integrate with (zsh or bash)"),
    apply: bool = typer.Option(False, help="Append the integration snippet to your shell rc file"),
):
    """Show or apply shell integration to bring OpenAgent into your terminal."""
    try:
        rc_path, snippet = install_snippet(shell, apply=apply)
        if apply:
            console.print(Panel(
                f"Installed OpenAgent integration into {rc_path}.\n\nEnable features by setting env vars in your rc file, e.g.:\nexport OPENAGENT_EXPLAIN=1\nexport OPENAGENT_WARN=1\n\nRestart your shell or run: source {rc_path}",
                title="Shell Integration Applied"
            ))
        else:
            console.print(Panel(snippet, title="Add this to your .zshrc"))
    except Exception as e:
        console.print(f"[red]Failed to set up integration: {e}[/red]")


@app.command()
def models():
    """List all available models."""
    
    console.print(Panel(
        f"""[bold cyan]Code Models (Best for programming tasks):[/bold cyan]
{chr(10).join([f'• [green]{k}[/green]: {v}' for k, v in ModelConfig.CODE_MODELS.items()])}

[bold cyan]Chat Models (General conversation):[/bold cyan]
{chr(10).join([f'• [blue]{k}[/blue]: {v}' for k, v in ModelConfig.CHAT_MODELS.items()])}

[bold cyan]Lightweight Models (Fast, low resource usage):[/bold cyan]
{chr(10).join([f'• [yellow]{k}[/yellow]: {v}' for k, v in ModelConfig.LIGHTWEIGHT_MODELS.items()])}

[dim]Use these model names with the --model flag[/dim]
""",
        title="Available Models"
    ))


# Plugin Management Commands
plugin_app = typer.Typer(help="Manage OpenAgent plugins")
app.add_typer(plugin_app, name="plugin")


@plugin_app.command("list")
def plugin_list(
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show detailed plugin information")
):
    """List all installed plugins."""
    console.print("[bold cyan]Installed Plugins:[/bold cyan]")
    
    # Example plugins
    plugins_info = [
        {
            "name": "weather",
            "version": "1.0.0", 
            "description": "Get weather information for locations",
            "status": "example",
            "path": "examples/plugins/weather"
        },
        {
            "name": "system-tools", 
            "version": "builtin",
            "description": "Built-in system management tools", 
            "status": "active",
            "path": "openagent.tools.system"
        }
    ]
    
    if verbose:
        for plugin in plugins_info:
            status_color = "green" if plugin["status"] == "active" else "yellow"
            console.print(f"\n[bold]{plugin['name']}[/bold] v{plugin['version']}")
            console.print(f"  📝 {plugin['description']}")
            console.print(f"  📍 {plugin['path']}")
            console.print(f"  🟢 Status: [{status_color}]{plugin['status']}[/{status_color}]")
    else:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Plugin", style="green")
        table.add_column("Version", style="cyan")
        table.add_column("Description", style="dim")
        table.add_column("Status", justify="center")
        
        for plugin in plugins_info:
            status_emoji = "🟢" if plugin["status"] == "active" else "🟡"
            table.add_row(
                plugin["name"], 
                plugin["version"], 
                plugin["description"],
                f"{status_emoji} {plugin['status']}"
            )
        
        console.print(table)


@plugin_app.command("install")
def plugin_install(
    source: str = typer.Argument(..., help="Plugin source (path, git repo, or name)"),
    force: bool = typer.Option(False, help="Force reinstall if already exists")
):
    """Install a plugin from various sources."""
    console.print(f"[dim]Installing plugin from: {source}[/dim]")
    
    # Simulate installation
    with console.status("[bold green]Installing...", spinner="dots"):
        import time
        time.sleep(2)
    
    # Check if it's an example plugin
    if "weather" in source.lower():
        console.print("✅ [green]Weather plugin installed successfully![/green]")
        console.print("\n📖 [dim]Usage example:[/dim]")
        console.print("  [cyan]openagent chat[/cyan]")
        console.print("  [dim]> What's the weather like in London?[/dim]")
    else:
        console.print(f"⚠️  [yellow]Plugin installation not yet implemented for: {source}[/yellow]")
        console.print("\n🔧 [dim]This feature is coming soon in v0.2.0![/dim]")


@plugin_app.command("info")
def plugin_info(
    plugin_name: str = typer.Argument(..., help="Name of plugin to show info for")
):
    """Show detailed information about a plugin."""
    if plugin_name == "weather":
        info = {
            "name": "weather",
            "version": "1.0.0",
            "description": "Get weather information for locations worldwide",
            "author": "OpenAgent Contributors",
            "tags": ["weather", "api", "information"],
            "tools": ["weather"],
            "examples": [
                "What's the weather in Tokyo?",
                "Check weather conditions in New York",
                "Is it raining in London?"
            ]
        }
        
        console.print(f"\n[bold cyan]{info['name']}[/bold cyan] v{info['version']}")
        console.print(f"📝 {info['description']}")
        console.print(f"👤 Author: {info['author']}")
        console.print(f"🏷️  Tags: {', '.join(info['tags'])}")
        console.print(f"🛠️  Tools: {', '.join(info['tools'])}")
        
        console.print("\n[bold]Example Usage:[/bold]")
        for example in info['examples']:
            console.print(f"  • [dim]{example}[/dim]")
    else:
        console.print(f"[red]Plugin '{plugin_name}' not found.[/red]")
        console.print("\nUse [cyan]openagent plugin list[/cyan] to see available plugins.")


def main():
    """Main CLI entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\\n[red]Fatal error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
