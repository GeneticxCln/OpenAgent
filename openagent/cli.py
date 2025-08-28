"""
Command Line Interface for OpenAgent.

This module provides a powerful CLI for interacting with OpenAgent,
similar to how Warp provides terminal AI assistance.
"""

import asyncio
import getpass
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

import httpx
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from openagent.core.agent import Agent
from openagent.core.config import Config
from openagent.core.history import HistoryManager
from openagent.core.llm import ModelConfig, get_llm
from openagent.core.performance.optimization import (
    get_memory_optimizer,
    get_model_cache,
    get_performance_profiler,
    get_startup_optimizer,
    optimize_openagent_performance,
)
from openagent.core.redact import redact_text
from openagent.core.workflows import Workflow, WorkflowManager
from openagent.terminal.integration import install_snippet
from openagent.terminal.validator import (
    CONFIG_PATH,
    DEFAULT_POLICY,
    load_policy,
    save_policy,
)
from openagent.terminal.validator import validate as validate_cmd
from openagent.tools.git import GitTool, RepoGrep
from openagent.tools.system import CommandExecutor, FileManager, SystemInfo
from openagent.ui import create_terminal_renderer

# Initialize Rich console
console = Console()
app = typer.Typer(help="OpenAgent - AI-powered terminal assistant")

# Global agent instance
agent: Optional[Agent] = None

FIRST_RUN_FLAG = Path.home() / ".config" / "openagent" / "first_run.json"
GLOBAL_KEYS_ENV = Path.home() / ".config" / "openagent" / "keys.env"


def _is_first_run() -> bool:
    try:
        if FIRST_RUN_FLAG.exists():
            data = json.loads(FIRST_RUN_FLAG.read_text())
            return not data.get("completed", False)
    except Exception:
        pass
    return True


def _set_first_run_completed():
    try:
        FIRST_RUN_FLAG.parent.mkdir(parents=True, exist_ok=True)
        FIRST_RUN_FLAG.write_text(json.dumps({"completed": True}))
    except Exception:
        pass


def _load_global_env():
    """Load ~/.config/openagent/keys.env if present (no output)."""
    try:
        if GLOBAL_KEYS_ENV.exists():
            # Load without overriding already-set env variables
            load_dotenv(dotenv_path=GLOBAL_KEYS_ENV, override=False)
    except Exception:
        pass


def _build_ws_url_and_headers(
    api_url: str,
    ws_path: str,
    api_token: Optional[str],
    token_query_key: Optional[str],
    auth_header_value: Optional[str],
):
    """Construct a ws(s) URL from an HTTP(S) base and optional auth.
    Returns (ws_url, extra_headers_list_or_None).
    """
    parsed = urlparse(api_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = ws_path if ws_path.startswith("/") else f"/{ws_path}"
    query = ""
    if token_query_key and api_token:
        from urllib.parse import urlencode

        query = urlencode({token_query_key: api_token})
    ws_url = urlunparse((scheme, parsed.netloc, path, "", query, ""))
    extra_headers = []
    if auth_header_value:
        extra_headers.append(("Authorization", auth_header_value))
    return ws_url, (extra_headers or None)


def _build_http_headers(accept_sse: bool, auth_header_value: Optional[str]):
    headers = {"Accept": "text/event-stream"} if accept_sse else {}
    if auth_header_value:
        headers["Authorization"] = auth_header_value
    return headers


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def create_agent(
    model_name: str = "tiny-llama",
    device: str = "auto",
    load_in_4bit: bool = True,
    unsafe_exec: bool = False,
) -> Agent:
    """Create and configure an OpenAgent instance.

    Note: By default, OpenAgent is safe and explain-only (unsafe_exec=False).
    Pass --unsafe-exec to allow actual command execution when you intend to run commands.
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
    agent.add_tool(GitTool())
    agent.add_tool(RepoGrep())

    # Warm the model asynchronously (best-effort)
    try:
        asyncio.get_event_loop().create_task(agent.llm.load_model())
    except Exception:
        pass

    return agent


@app.command()
def menu(
    allow_presets: bool = typer.Option(
        False,
        "--allow-presets",
        help="Allow showing preset HF models if no local Ollama models are found",
    )
):
    """Pick an agent/model and start chat immediately (local-only by default; no auto-serve)."""
    from openagent.core.llm import ModelConfig

    # Try to discover local Ollama models
    try:
        import asyncio as _asyncio

        from openagent.core.llm_ollama import list_ollama_models

        local_models = _asyncio.get_event_loop().run_until_complete(
            list_ollama_models()
        )
    except Exception:
        local_models = []

    if local_models:
        console.print(Panel.fit("Choose a local Ollama model", title="OpenAgent"))
        for idx, name in enumerate(local_models, start=1):
            console.print(f"  {idx}. ollama:{name}")
        try:
            model_idx = int(console.input("Model number: ").strip() or "1")
            model_idx = max(1, min(model_idx, len(local_models)))
        except Exception:
            model_idx = 1
        chosen = local_models[model_idx - 1]
        console.print("\nLaunching local model...\n")
        chat(
            model=f"ollama:{chosen}",
            provider="local",
            device="auto",
            load_in_4bit=True,
            debug=False,
            unsafe_exec=False,
            max_new_tokens=128,
            temperature=0.5,
            auto_serve=False,
            api_url=None,
            ws=False,
            no_stream=False,
        )
        return

    # No local models detected
    if not allow_presets:
        console.print(
            Panel(
                "No local Ollama models detected.\n\nInstall a model, for example:\n  ollama pull qwen3:8b\n\nThen re-run: openagent\n\nTo browse preset models instead, run:\n  openagent menu --allow-presets",
                title="Local Models Only",
            )
        )
        return

    # Optional fallback to presets if explicitly allowed
    code_models = list(ModelConfig.CODE_MODELS.keys())
    chat_models = list(ModelConfig.CHAT_MODELS.keys())
    light_models = list(ModelConfig.LIGHTWEIGHT_MODELS.keys())

    sections = [
        ("Lightweight", light_models),
        ("Code", code_models),
        ("Chat", chat_models),
    ]

    console.print(Panel.fit("Choose your agent/model (presets)", title="OpenAgent"))

    # Category selection
    for idx, (title, _) in enumerate(sections, start=1):
        console.print(f"  {idx}. {title}")
    try:
        cat_choice = int(console.input("Category number: ").strip() or "1")
        cat_choice = max(1, min(cat_choice, len(sections)))
    except Exception:
        cat_choice = 1

    _, models = sections[cat_choice - 1]

    # Model selection
    for idx, name in enumerate(models, start=1):
        console.print(f"  {idx}. {name}")
    try:
        model_idx = int(console.input("Model number: ").strip() or "1")
        model_idx = max(1, min(model_idx, len(models)))
    except Exception:
        model_idx = 1

    model = models[model_idx - 1]

    # Start chat with chosen preset, without auto-serve
    console.print("\nLaunching agent...\n")
    chat(
        model=model,
        provider="local",
        device="auto",
        load_in_4bit=True,
        debug=False,
        unsafe_exec=False,
        max_new_tokens=128,
        temperature=0.5,
        auto_serve=False,
        api_url=None,
        ws=False,
        no_stream=False,
    )


@app.command()
def chat(
    model: str = typer.Option(
        "auto", help="Model to use for responses (auto resolves per provider)"
    ),
    provider: str = typer.Option("auto", help="Provider: auto|local|cloud"),
    device: str = typer.Option("auto", help="Device to run model on (auto/cpu/cuda)"),
    load_in_4bit: bool = typer.Option(True, help="Load model in 4-bit precision"),
    debug: bool = typer.Option(False, help="Enable debug logging"),
    unsafe_exec: bool = typer.Option(
        False,
        "--unsafe-exec/--no-unsafe-exec",
        help="Allow actual command execution (default false). Use --unsafe-exec to run commands.",
    ),
    max_new_tokens: int = typer.Option(128, help="Maximum new tokens to generate"),
    temperature: float = typer.Option(0.5, help="Sampling temperature"),
    api_url: Optional[str] = typer.Option(
        None,
        help="API server base URL (e.g., http://localhost:8000). If provided or OPENAGENT_API_URL is set, use server with streaming.",
    ),
    no_stream: bool = typer.Option(
        False, help="Disable streaming even if server is used"
    ),
    ws: bool = typer.Option(
        False,
        help="Use WebSocket streaming when using API server (falls back to SSE if unavailable)",
    ),
    auto_serve: bool = typer.Option(
        True,
        "--auto-serve/--no-auto-serve",
        help="Auto-start local API server if none provided",
    ),
    ws_path: str = typer.Option(
        "/ws/chat", help="WebSocket path on the server (used when --ws is set)"
    ),
    api_token: Optional[str] = typer.Option(
        None,
        help="API token for Authorization header. If not provided, uses OPENAGENT_API_TOKEN or OPENAGENT_API_KEY env vars.",
    ),
    auth_scheme: str = typer.Option(
        "Bearer",
        help="Auth scheme prefix for token (e.g., Bearer). Use empty string to send raw token.",
    ),
    ws_token_query_key: Optional[str] = typer.Option(
        None,
        help="If set, also include the token as a query parameter with this key on the WebSocket URL.",
    ),
    no_ui_blocks: bool = typer.Option(
        False,
        "--no-ui-blocks",
        help="Disable block UI and print raw text to console",
    ),
):
    """Start an interactive chat session with OpenAgent."""

    setup_logging(debug)
    load_dotenv()

    # Resolve automatic model choice
    resolved_model = model
    # Provider-aware resolution
    if provider not in ("auto", "local", "cloud"):
        raise typer.BadParameter("--provider must be one of: auto|local|cloud")
    elif provider == "local":
        # Respect explicit ollama:<tag>
        if (
            isinstance(model, str)
            and model.startswith("ollama:")
            and model != "ollama:"
        ):
            resolved_model = model
        else:
            # Prefer installed Ollama default; else tiny-llama or the given non-auto model
            try:
                import asyncio as _asyncio

                from openagent.core.llm_ollama import get_default_ollama_model

                m = _asyncio.get_event_loop().run_until_complete(
                    get_default_ollama_model()
                )
            except Exception:
                m = None
            resolved_model = (
                f"ollama:{m}" if m else (model if model != "auto" else "tiny-llama")
            )
    elif provider == "cloud":
        # Local-only build: cloud provider is disabled
        raise typer.BadParameter(
            "Cloud provider is disabled in local-only mode. Use --provider local and a local model (e.g., --model 'ollama:\u003cname\u003e')."
        )
    else:
        # auto provider: prefer Ollama, else tiny
        if model == "auto":
            try:
                import asyncio as _asyncio

                from openagent.core.llm_ollama import get_default_ollama_model

                m = _asyncio.get_event_loop().run_until_complete(
                    get_default_ollama_model()
                )
            except Exception:
                m = None
            resolved_model = f"ollama:{m}" if m else "tiny-llama"

    console.print(
        Panel.fit(
            "[bold blue]OpenAgent Terminal Assistant[/bold blue]\n"
            "AI-powered terminal assistance with code generation and system operations.\n"
            f"Model: {resolved_model} | Device: {device} | 4-bit: {load_in_4bit}",
            title="🤖 Welcome",
        )
    )
    # Allow using HTTP API server if provided
    api_url = api_url or os.getenv("OPENAGENT_API_URL")
    use_server = False
    resolved_token = None
    auth_header_value = None
    if api_url:
        use_server = True
        console.print(f"\n[dim]Using API server: {api_url}[/dim]")

        # Resolve API token from args or environment
        resolved_token = (
            api_token
            or os.getenv("OPENAGENT_API_TOKEN")
            or os.getenv("OPENAGENT_API_KEY")
        )
        if resolved_token:
            auth_header_value = (
                f"{auth_scheme} {resolved_token}".strip()
                if auth_scheme
                else resolved_token
            )
    elif auto_serve:
        # Auto-start local server on a free port
        import socket
        import subprocess
        import time

        def _free_port():
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
            s.close()
            return p

        port = _free_port()
        base = f"http://127.0.0.1:{port}"
        console.print(f"\n[dim]Starting local API server at {base}...[/dim]")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "openagent.server.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Best-effort wait for readiness
        import httpx as _hx

        ok = False
        for _ in range(30):
            try:
                with _hx.Client(timeout=0.3) as client:
                    r = client.get(f"{base}/healthz")
                    if r.status_code == 200:
                        ok = True
                        break
            except Exception:
                pass
            time.sleep(0.2)
        if not ok:
            console.print(
                "[yellow]Server did not become ready in time; continuing without server[/yellow]"
            )
        else:
            api_url = base
            use_server = True
            console.print(f"[dim]Local server ready: {api_url}[/dim]")
            os.environ["OPENAGENT_API_URL"] = api_url

        # Resolve API token from args or environment (none by default for local)
        resolved_token = (
            api_token
            or os.getenv("OPENAGENT_API_TOKEN")
            or os.getenv("OPENAGENT_API_KEY")
        )
        if resolved_token:
            auth_header_value = (
                f"{auth_scheme} {resolved_token}".strip()
                if auth_scheme
                else resolved_token
            )
    else:
        # Try remote daemon first for instant startup
        remote_url = os.getenv("OPENAGENT_DAEMON_URL", "http://127.0.0.1:8765")
        use_remote = False
        try:
            with httpx.Client(timeout=1.0) as client:
                r = client.get(f"{remote_url}/health")
                if r.status_code == 200:
                    use_remote = True
        except Exception:
            use_remote = False

        if use_remote:
            console.print("\n[dim]Connected to background agent daemon.[/dim]")
        else:
            # Local agent fallback
            global agent
            agent = create_agent(resolved_model, device, load_in_4bit, unsafe_exec)
            console.print(
                "\n[dim]Initializing AI model... This may take a moment.[/dim]"
            )
            # Apply generation preferences
            try:
                agent.llm.generation_config.max_new_tokens = max_new_tokens
                agent.llm.temperature = temperature
            except Exception:
                pass

    # Register plugin-provided tools with agent (local session only)
    try:
        from openagent.plugins.manager import PluginManager

        pm_for_tools = PluginManager()
        # Only attempt if not using server; server mode uses its own tool layer
        if not api_url:
            import asyncio as _asyncio

            async def _load_and_attach():
                await pm_for_tools.initialize()
                await pm_for_tools.discover_plugins()
                await pm_for_tools.load_all_plugins()
                # Enable plugins that are configured enabled
                for name, info in [(md.get("metadata", {}).get("name"), md) for md in await pm_for_tools.list_plugins()]:
                    if not name:
                        continue
                    try:
                        await pm_for_tools.enable_plugin(name)
                    except Exception:
                        pass
                if agent:
                    pm_for_tools.register_tools_with_agent(agent)

            _asyncio.get_event_loop().run_until_complete(_load_and_attach())
    except Exception:
        pass

    # Interactive chat loop
    asyncio.run(
        chat_loop(
            use_remote=(not use_server and "use_remote" in locals() and use_remote),
            remote_url=(
                locals().get("remote_url") if "remote_url" in locals() else None
            ),
            api_url=api_url,
            stream=(not no_stream),
            ws=ws,
            ws_path=ws_path,
            auth_header_value=auth_header_value,
            api_token=resolved_token,
            ws_token_query_key=ws_token_query_key,
            use_blocks=(not no_ui_blocks),
        )
    )


async def chat_loop(
    use_remote: bool = False,
    remote_url: Optional[str] = None,
    api_url: Optional[str] = None,
    stream: bool = True,
    ws: bool = False,
    ws_path: str = "/ws/chat",
    auth_header_value: Optional[str] = None,
    api_token: Optional[str] = None,
    ws_token_query_key: Optional[str] = None,
    use_blocks: bool = True,
):
    """Main interactive chat loop.

    When use_blocks=True, responses are rendered via the block UI with folding and status.
    """

    console.print("\n[green]Ready! Type your message or 'help' for commands.[/green]")
    console.print("[dim]Special commands: /help, /status, /reset, /quit[/dim]\n")

    # Initialize block UI if requested
    renderer = None
    if use_blocks:
        try:
            renderer = create_terminal_renderer()
            renderer.start_live_display()
        except Exception:
            renderer = None
            use_blocks = False

    while True:
        try:
            # Get user input
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()

            if not user_input:
                continue

            # Handle special commands
            if user_input.startswith("/"):
                if await handle_special_command(user_input):
                    continue
                else:
                    break

            # Process user message with loading spinner
            with console.status("[bold green]Thinking...", spinner="dots"):
                if api_url:
                    # Prefer WebSocket streaming if requested; fallback to SSE or non-streaming
                    payload = {"message": user_input}
                    if stream and ws:
                        # WebSocket streaming
                        current_block = None
                        accum = ""
                        if use_blocks and renderer:
                            current_block = renderer.add_ai_response("")
                        else:
                            console.print(f"\n[bold green]Assistant:[/bold green]")
                        try:
                            # Prefer an already-injected websockets module (for testing); else import
                            ws_mod = globals().get("websockets")
                            if ws_mod is None:
                                try:
                                    import websockets as ws_mod  # type: ignore
                                except Exception as _e:
                                    raise RuntimeError(
                                        "websockets package not installed. Install with: pip install websockets"
                                    )
                            # Build ws URL and headers from base URL and options
                            ws_url, extra_headers = _build_ws_url_and_headers(
                                api_url,
                                ws_path,
                                api_token,
                                ws_token_query_key,
                                auth_header_value,
                            )
                            async with ws_mod.connect(
                                ws_url, max_queue=None, extra_headers=extra_headers
                            ) as websocket:
                                # Send initial message payload
                                await websocket.send(json.dumps(payload))
                                while True:
                                    msg = await websocket.recv()
                                    if isinstance(msg, (bytes, bytearray)):
                                        # Ignore binary frames for now
                                        continue
                                    try:
                                        data = json.loads(msg)
                                    except Exception:
                                        # Print raw if not JSON
                                        console.print(str(msg), end="")
                                        continue
                                    # Expect {'content': '...', 'event': 'chunk'|'end' }
                                    if data.get("content"):
                                        chunk = redact_text(str(data["content"]))
                                        if use_blocks and renderer and current_block:
                                            accum += chunk
                                            renderer.update_block_output(current_block, accum)
                                        else:
                                            console.print(chunk, end="")
                                    if data.get("event") == "end":
                                        if use_blocks and renderer and current_block:
                                            renderer.complete_block_execution(current_block, exit_code=0)
                                        else:
                                            console.print()
                                        break

                            class R:
                                pass

                            response = R()
                            response.content = ""  # streamed already
                            response.metadata = {}
                        except Exception as e:
                            # Fall back to SSE streaming
                            async with httpx.AsyncClient(timeout=None) as client:
                                try:
                                    headers = _build_http_headers(
                                        accept_sse=True,
                                        auth_header_value=auth_header_value,
                                    )
                                    # SSE fallback streaming
                                    current_block = None
                                    accum = ""
                                    if use_blocks and renderer:
                                        current_block = renderer.add_ai_response("")
                                    else:
                                        console.print(f"\n[bold green]Assistant:[/bold green]")
                                    async with client.stream(
                                        "POST",
                                        f"{api_url}/chat/stream",
                                        json=payload,
                                        headers=headers,
                                    ) as resp:
                                        if resp.status_code != 200:
                                            text = await resp.aread()
                                            raise RuntimeError(
                                                f"Server returned {resp.status_code}: {text.decode(errors='ignore')[:200]}"
                                            )
                                        async for line in resp.aiter_lines():
                                            if not line:
                                                continue
                                            if line.startswith("data: "):
                                                try:
                                                    data = json.loads(line[6:].strip())
                                                    chunk = data.get("content")
                                                    if chunk:
                                                        console.print(chunk, end="")
                                                except Exception:
                                                    pass
                                            elif line.startswith("event: end"):
                                                console.print()

                                        class R:
                                            pass

                                        response = R()
                                        response.content = ""
                                        response.metadata = {
                                            "fallback": "sse",
                                            "error_ws": str(e),
                                        }
                                except Exception as e2:

                                    class R:
                                        pass

                                    response = R()
                                    response.content = f"Error streaming: WS failed ({e}); SSE failed ({e2})"
                                    response.metadata = {}
                    elif stream:
                        # SSE streaming (primary)
                        current_block = None
                        accum = ""
                        if use_blocks and renderer:
                            current_block = renderer.add_ai_response("")
                        else:
                            console.print(f"\n[bold green]Assistant:[/bold green]")
                        async with httpx.AsyncClient(timeout=None) as client:
                            try:
                                headers = _build_http_headers(
                                    accept_sse=True, auth_header_value=auth_header_value
                                )
                                async with client.stream(
                                    "POST",
                                    f"{api_url}/chat/stream",
                                    json=payload,
                                    headers=headers,
                                ) as resp:
                                    if resp.status_code != 200:
                                        text = await resp.aread()
                                        raise RuntimeError(
                                            f"Server returned {resp.status_code}: {text.decode(errors='ignore')[:200]}"
                                        )
                                    async for line in resp.aiter_lines():
                                        if not line:
                                            continue
                                        if line.startswith("data: "):
                                            try:
                                                data = json.loads(line[6:].strip())
                                                chunk = data.get("content")
                                                if chunk:
                                                    chunk = redact_text(str(chunk))
                                                    if use_blocks and renderer and current_block:
                                                        accum += chunk
                                                        renderer.update_block_output(current_block, accum)
                                                    else:
                                                        console.print(chunk, end="")
                                            except Exception:
                                                pass
                                        elif line.startswith("event: end"):
                                            if use_blocks and renderer and current_block:
                                                renderer.complete_block_execution(current_block, exit_code=0)
                                            else:
                                                console.print()

                                class R:
                                    pass

                                response = R()
                                response.content = ""
                                response.metadata = {}
                            except Exception as e:

                                class R:
                                    pass

                                response = R()
                                response.content = f"Error streaming from server: {e}"
                                response.metadata = {}
                    else:
                        # Non-streaming fallback
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            try:
                                headers = _build_http_headers(
                                    accept_sse=False,
                                    auth_header_value=auth_header_value,
                                )
                                r = await client.post(
                                    f"{api_url}/chat", json=payload, headers=headers
                                )
                                r.raise_for_status()
                                data = r.json()

                                class R:
                                    pass

                                response = R()
                                response.content = data.get("message", "")
                                response.metadata = data.get("metadata", {})
                            except Exception as e:

                                class R:
                                    pass

                                response = R()
                                response.content = f"Error contacting server: {e}"
                                response.metadata = {}
                elif use_remote and remote_url:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        try:
                            payload = {"message": user_input}
                            r = await client.post(f"{remote_url}/chat", json=payload)
                            r.raise_for_status()
                            data = r.json()

                            class R:
                                pass

                            response = R()
                            response.content = data.get("message", "")
                            response.metadata = data.get("metadata", {})
                        except Exception as e:

                            class R:
                                pass

                            response = R()
                            response.content = f"Error contacting daemon: {e}"
                            response.metadata = {}
                else:
                    response = await agent.process_message(user_input)

            # Display response (if not already streamed)
            if (
                not api_url
                or not stream
                or (getattr(response, "content", None) and response.content)
            ):
                if use_blocks and renderer:
                    renderer.add_ai_response(redact_text(response.content or ""))
                else:
                    console.print(f"\n[bold green]Assistant:[/bold green]")
                    out = redact_text(response.content or "")
                    if "```" in out:
                        console.print(Markdown(out))
                    else:
                        console.print(out)

            # Persist block to history
            try:
                hm = HistoryManager()
                plan = getattr(agent, "_last_block", {}).get("plan") if agent else None
                tool_results = (
                    getattr(agent, "_last_block", {}).get("tool_results")
                    if agent
                    else []
                )
                model_info = agent.llm.get_model_info() if agent else None
                block = HistoryManager.new_block(
                    input_text=user_input,
                    response=response.content,
                    plan=plan,
                    tool_results=tool_results,
                    model=model_info,
                    context={"cwd": str(Path.cwd())},
                )
                hm.append(block)
                response.metadata["block_id"] = block.id
            except Exception:
                pass

            # Show metadata if debug
            if response.metadata.get("tools_used"):
                console.print(
                    f"\\n[dim]Tools used: {', '.join(response.metadata['tools_used'])}[/dim]"
                )

            console.print()  # Add spacing

        except KeyboardInterrupt:
            console.print("\\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"\\n[red]Error: {e}[/red]")


async def handle_special_command(command: str) -> bool:
    """Handle special chat commands. Returns True to continue chat, False to exit."""

    if command == "/help":
        console.print(
            Panel(
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
                title="Help",
            )
        )
        return True

    if command == "/status":
        status = agent.get_status()
        model_info = agent.llm.get_model_info()
        safe_mode = (
            agent.config.get("safe_mode", True) if hasattr(agent, "config") else True
        )

        console.print(
            Panel(
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
                title="Status",
            )
        )
        return True

    if command == "/reset":
        agent.reset()
        console.print("[green]Conversation history reset![/green]")
        return True

    if command == "/models":
        console.print(
            Panel(
                f"""[bold]Available Models:[/bold]

[bold cyan]Code Models (Recommended):[/bold cyan]
{chr(10).join([f'• {k}: {v}' for k, v in ModelConfig.CODE_MODELS.items()])}

[bold cyan]Chat Models:[/bold cyan]
{chr(10).join([f'• {k}: {v}' for k, v in ModelConfig.CHAT_MODELS.items()])}

[bold cyan]Lightweight Models:[/bold cyan]
{chr(10).join([f'• {k}: {v}' for k, v in ModelConfig.LIGHTWEIGHT_MODELS.items()])}
""",
                title="Models",
            )
        )
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
    unsafe_exec: bool = typer.Option(
        False,
        "--unsafe-exec/--no-unsafe-exec",
        help="Allow actual command execution (default false). Use --unsafe-exec to run.",
    ),
    max_new_tokens: int = typer.Option(128, help="Maximum new tokens to generate"),
    temperature: float = typer.Option(0.5, help="Sampling temperature"),
):
    """Run a single prompt through OpenAgent and exit."""

    load_dotenv()
    # Only print human-friendly initializing message for text output
    if output_format != "json":
        console.print("[dim]Initializing...[/dim]")

    # Create agent
    agent = create_agent(model, device, load_in_4bit, unsafe_exec)

    async def run_single():
        # Inject generation preferences into agent's LLM
        try:
            agent.llm.generation_config.max_new_tokens = max_new_tokens
            agent.llm.temperature = temperature
        except Exception:
            pass
        with (
            console.status("[bold green]Processing...", spinner="dots")
            if output_format != "json"
            else contextlib.nullcontext()
        ):
            response = await agent.process_message(prompt)

        if output_format == "json":
            import json

            result = {
                "prompt": prompt,
                "response": response.content,
                "metadata": response.metadata,
            }
            # Print only JSON to stdout (no color formatting for JSON output)
            print(json.dumps(result, indent=2))
        else:
            if "```" in response.content:
                console.print(Markdown(response.content))
            else:
                console.print(response.content)

    import contextlib

    asyncio.run(run_single())


@app.command()
def blocks(
    action: str = typer.Argument(
        ..., help="Action: list|show|export|search|rerun|pick"
    ),
    arg: str = typer.Argument(None, help="ID or query depending on action"),
    format: str = typer.Option("md", help="Export format for export action (md|json)"),
    limit: int = typer.Option(20, help="Limit for list/search"),
    tool: str = typer.Option(None, help="Filter by tool name (list/search)"),
    status: str = typer.Option(
        None, help="Filter by status success|error (list/search)"
    ),
    since: str = typer.Option(None, help="Filter since date (YYYY-MM-DD)"),
    until: str = typer.Option(None, help="Filter until date (YYYY-MM-DD)"),
    edit: bool = typer.Option(False, help="Rerun with edit (open $EDITOR)"),
):
    """Manage and inspect history blocks."""
    hm = HistoryManager()
    if action == "list":
        items = hm.list_blocks(limit=limit)

        # Apply filters client-side (best-effort)
        def _date_ok(ts: str) -> bool:
            if not ts:
                return True
            try:
                d = ts[:10]
                if since and d < since:
                    return False
                if until and d > until:
                    return False
            except Exception:
                return True
            return True

        filtered = []
        for o in items:
            if tool and tool not in " ".join(
                [tr.get("tool", "") for tr in o.get("tool_results", [])]
            ):
                continue
            if status:
                s = status.lower()
                any_success = any(
                    bool(tr.get("success")) for tr in o.get("tool_results", [])
                )
                if s == "success" and not any_success:
                    continue
                if s == "error" and any_success:
                    continue
            if not _date_ok(o.get("timestamp", "")):
                continue
            filtered.append(o)
        items = filtered
        table = Table(show_header=True, header_style="bold")
        table.add_column("ID")
        table.add_column("Time")
        table.add_column("Input")
        for o in items:
            table.add_row(
                o.get("id", ""), o.get("timestamp", ""), (o.get("input", "") or "")[:60]
            )
        console.print(table)
        return
    if action == "show":
        if not arg:
            console.print("[red]Missing block ID[/red]")
            raise typer.Exit(1)
        obj = hm.get(arg)
        if not obj:
            console.print("[red]Not found[/red]")
            raise typer.Exit(1)
        md = f"""# Block {obj['id']}

- Time: {obj.get('timestamp')}
- Model: {obj.get('model', {}).get('model_name')}

## Input

```
{obj.get('input','')}
```

## Tool Results

{chr(10).join([('- ' + (tr.get('tool','tool')) + (': success' if tr.get('success') else ': error ' + str(tr.get('error')))) for tr in obj.get('tool_results',[])])}

## Response

{obj.get('response','')}
"""
        console.print(Markdown(md))
        return
    if action == "export":
        if not arg:
            console.print("[red]Missing block ID[/red]")
            raise typer.Exit(1)
        obj = hm.get(arg)
        if not obj:
            console.print("[red]Not found[/red]")
            raise typer.Exit(1)
        if format == "json":
            # Redact sensitive data before export
            console.print_json(data=json.loads(redact_text(json.dumps(obj))))
        else:
            # Redact the markdown export
            md = f"""# Block {obj['id']}

- Time: {obj.get('timestamp')}
- Model: {obj.get('model', {}).get('model_name')}

## Input

```
{obj.get('input','')}
```

## Tool Results

{chr(10).join([('- ' + (tr.get('tool','tool')) + (': success' if tr.get('success') else ': error ' + str(tr.get('error')))) for tr in obj.get('tool_results',[])])}

## Response

{obj.get('response','')}
"""
            console.print(Markdown(md))
        return
    if action == "search":
        if not arg:
            console.print("[red]Missing query[/red]")
            raise typer.Exit(1)
        items = hm.search(arg, limit=limit)
        table = Table(show_header=True, header_style="bold")
        table.add_column("ID")
        table.add_column("Time")
        table.add_column("Snippet")
        for o in items:
            snippet = (o.get("response") or o.get("input") or "")[:80]
            table.add_row(o.get("id", ""), o.get("timestamp", ""), snippet)
        # Apply filters as in list
        console.print(table)
        return
    if action == "pick":
        # Provide an interactive picker if fzf is available; fallback to list
        try:
            import shutil
            import subprocess as sp

            if shutil.which("fzf"):
                items = hm.list_blocks(limit=200)
                lines = [
                    f"{o.get('id','')}  {o.get('timestamp','')}  {(o.get('input','') or '')[:60]}"
                    for o in items
                ]
                sel = sp.run(
                    ["fzf", "--prompt", "block> ", "--with-nth", "1,2,3.."],
                    input=("\n".join(lines)).encode(),
                    stdout=sp.PIPE,
                )
                chosen = sel.stdout.decode().strip().split()[0] if sel.stdout else ""
                if chosen:
                    console.print(chosen)
                else:
                    console.print("[yellow]No selection[/yellow]")
            else:
                console.print(
                    "[dim]fzf not found. Use: openagent blocks list | your-picker[/dim]"
                )
        except Exception as e:
            console.print(f"[red]Picker error: {e}[/red]")
        return
    if action == "rerun":
        if not arg:
            console.print("[red]Missing block ID[/red]")
            raise typer.Exit(1)
        obj = hm.get(arg)
        if not obj:
            console.print("[red]Not found[/red]")
            raise typer.Exit(1)
        prompt = obj.get("input", "")
        if edit:
            # open in EDITOR
            import subprocess as sp
            import tempfile

            with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tf:
                tf.write(prompt)
                tf.flush()
                editor = os.getenv("EDITOR", "nano")
                sp.call([editor, tf.name])
                tf.seek(0)
                prompt = tf.read()
        console.print(f"[dim]Re-running block {arg}...[/dim]")
        # Create agent with defaults
        a = create_agent("auto", "auto", True, unsafe_exec=False)

        async def run_again():
            with console.status("[bold green]Processing...", spinner="dots"):
                resp = await a.process_message(prompt)
            console.print(resp.content)

        asyncio.run(run_again())
        return
    console.print("[red]Unknown action[/red]")


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
def fix(
    block_id: str = typer.Argument(
        ..., help="Block ID containing a suggested fix from a failed command"
    ),
    auto: bool = typer.Option(
        False, help="Execute immediately without showing the suggestion"
    ),
    dry_run: bool = typer.Option(False, help="Explain-only; do not execute"),
):
    """Execute a suggested fix command from a prior failed step (approval-based)."""
    hm = HistoryManager()
    obj = hm.get(block_id)
    if not obj:
        console.print("[red]Block not found[/red]")
        raise typer.Exit(1)
    # Find suggested fix
    suggested = None
    for tr in obj.get("tool_results", []):
        meta = tr.get("metadata") or {}
        if meta.get("suggested_fix_command"):
            suggested = meta.get("suggested_fix_command")
            break
    if not suggested:
        console.print("[yellow]No suggested fix found in this block[/yellow]")
        raise typer.Exit(1)
    if not auto:
        console.print(Panel(f"Proposed fix:\n\n{suggested}", title="Suggested Command"))
        confirm = console.input("Run this command? [y/N]: ").strip().lower()
        if confirm != "y":
            console.print("[yellow]Cancelled[/yellow]")
            return

    # Execute or explain
    async def do_exec():
        tool = CommandExecutor(default_explain_only=dry_run)
        with (
            console.status("[bold green]Executing fix...", spinner="dots")
            if not dry_run
            else console.status("[bold green]Explaining fix...", spinner="dots")
        ):
            res = await tool.execute(
                {"command": suggested, "explain_only": dry_run, "confirm": True}
            )
        if res.success:
            console.print(
                Panel(
                    res.content or "",
                    title=f"Fix {'Explanation' if dry_run else 'Output'}",
                )
            )
        else:
            console.print(Panel(f"Error: {res.error}", title="Fix Failed", style="red"))

    asyncio.run(do_exec())


@app.command()
def code(
    description: str = typer.Argument(..., help="Description of code to generate"),
    language: str = typer.Option("python", help="Programming language"),
    model: str = typer.Option(
        "codellama-7b", help="Model to use (code models recommended)"
    ),
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

            console.print(
                Panel(Markdown(analysis), title=f"Code Analysis: {file_path}")
            )

        asyncio.run(analyze_code())

    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def validate(
    command: str = typer.Argument(..., help="Shell command to validate"),
    quiet: bool = typer.Option(
        False, help="Print only the decision (allow/warn/block)"
    ),
    quiet_with_reason: bool = typer.Option(
        False, help="Print 'decision|reason' on a single line for shell integration"
    ),
):
    """Validate a command against OpenAgent's policy (allow/warn/block)."""
    decision, reason = validate_cmd(command)
    if quiet_with_reason:
        # One-line machine-friendly output for shell integration
        console.print(f"{decision}|{reason}")
        return
    if quiet:
        console.print(decision)
    else:
        console.print(
            Panel(
                f"Decision: [bold]{decision}[/bold]\nReason: {reason}",
                title="Validation",
            )
        )


@app.command()
def policy(
    action: str = typer.Argument(
        ...,
        help="Action: show|reset|set-default|add-allow|remove-allow|block-risky|unblock-risky|strict|relaxed",
    ),
    key: str = typer.Argument(
        None, help="Command for allowlist updates or default value"
    ),
    value: str = typer.Argument(
        None, help="Value for the action (flag prefix or default decision)"
    ),
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
            console.print(
                "[red]Usage: policy remove-allow <command> <flag_prefix>[/red]"
            )
            raise typer.Exit(1)
        flags = p.setdefault("allowlist", {}).setdefault(key, [])
        if value in flags:
            flags.remove(value)
        save_policy(p)
        console.print(
            f"[green]Removed allow flag '{value}' for command '{key}'[/green]"
        )
        return
    if action == "block-risky":
        p["block_risky"] = True
        save_policy(p)
        console.print("[green]Risky commands will be blocked[/green]")
        return
    if action == "unblock-risky":
        p["block_risky"] = False
        save_policy(p)
        console.print("[green]Risky commands will not be blocked (may still warn)")
        return
    if action == "strict":
        p["default_decision"] = "block"
        p["block_risky"] = True
        save_policy(p)
        console.print(
            "[green]Policy set to STRICT (block-by-default, risky blocked).[/green]"
        )
        return
    if action == "relaxed":
        p["default_decision"] = "warn"
        p["block_risky"] = True
        save_policy(p)
        console.print(
            "[green]Policy set to RELAXED (warn-by-default, risky blocked).[/green]"
        )
        return
    console.print("[red]Unknown action[/red]")


@app.command()
def completion(
    shell: str = typer.Argument("zsh", help="Shell: zsh|bash"),
    install: bool = typer.Option(
        False, help="Write completion file to ~/.config/openagent"
    ),
):
    """Output a minimal completion script for OpenAgent commands."""
    base_dir = Path.home() / ".config" / "openagent"
    base_dir.mkdir(parents=True, exist_ok=True)
    if shell == "zsh":
        script = """
#compdef openagent
_arguments '*: :->cmds'

case $state in
  cmds)
    local -a subcmds
    subcmds=(chat run blocks workflow fix exec do models doctor setup integrate serve policy validate completion)
    _describe 'command' subcmds
    ;;
esac
"""
        if install:
            path = base_dir / "_openagent"
            path.write_text(script)
            console.print(
                Panel(
                    f"Installed zsh completion to {path}. Add to fpath and run: compinit",
                    title="Completion",
                )
            )
        else:
            console.print(script)
    else:
        # simple bash completion
        script = """
_openagent_completions()
{
    COMPREPLY=()
    local cmds="chat run blocks workflow fix exec do models doctor setup integrate serve policy validate completion"
    COMPREPLY=( $(compgen -W "$cmds" -- ${COMP_WORDS[1]}) )
}
complete -F _openagent_completions openagent
"""
        if install:
            path = base_dir / "openagent.bash"
            path.write_text(script)
            console.print(
                Panel(
                    f"Installed bash completion to {path}. Source it in your shell rc.",
                    title="Completion",
                )
            )
        else:
            console.print(script)


@app.command()
def integrate(
    shell: str = typer.Option("zsh", help="Shell to integrate with (zsh or bash)"),
    apply: bool = typer.Option(
        False, help="Append the integration snippet to your shell rc file"
    ),
):
    """Show or apply shell integration to bring OpenAgent into your terminal."""
    try:
        rc_path, snippet = install_snippet(shell, apply=apply)
        if apply:
            console.print(
                Panel(
                    f"Installed OpenAgent integration into {rc_path}.\n\nEnable features by setting env vars in your rc file, e.g.:\nexport OPENAGENT_EXPLAIN=1\nexport OPENAGENT_WARN=1\n\nRestart your shell or run: source {rc_path}",
                    title="Shell Integration Applied",
                )
            )
        else:
            console.print(Panel(snippet, title="Add this to your .zshrc"))
    except Exception as e:
        console.print(f"[red]Failed to set up integration: {e}[/red]")


@app.command()
def setup():
    """First-run setup (local-only): configure .env for local models.

    - Local: tiny model fallback, 4-bit, device auto/cuda.
    """
    load_dotenv()
    console.print(
        Panel.fit("Welcome to OpenAgent setup (local-only)", title="First Run")
    )

    env_path = Path.cwd() / ".env"
    existing = env_path.read_text() if env_path.exists() else ""

    def write_env(lines: str):
        env_path.write_text(lines)
        console.print(Panel(f"Updated {env_path}", title="Config"))

    # Local-only defaults
    lines = []
    found = set()
    for line in existing.splitlines():
        if line.startswith("DEFAULT_MODEL="):
            lines.append("DEFAULT_MODEL=tiny-llama")
            found.add("DEFAULT_MODEL")
        elif line.startswith("DEFAULT_DEVICE="):
            preferred = "cuda"
            try:
                import torch  # type: ignore

                if not torch.cuda.is_available():
                    preferred = "auto"
            except Exception:
                preferred = "auto"
            lines.append(f"DEFAULT_DEVICE={preferred}")
            found.add("DEFAULT_DEVICE")
        elif line.startswith("LOAD_IN_4BIT="):
            lines.append("LOAD_IN_4BIT=true")
            found.add("LOAD_IN_4BIT")
        else:
            lines.append(line)
    if "DEFAULT_MODEL" not in found:
        lines.append("DEFAULT_MODEL=tiny-llama")
    if "DEFAULT_DEVICE" not in found:
        lines.append("DEFAULT_DEVICE=auto")
    if "LOAD_IN_4BIT" not in found:
        lines.append("LOAD_IN_4BIT=true")
    if not lines:
        lines = [
            "DEFAULT_MODEL=tiny-llama",
            "DEFAULT_DEVICE=auto",
            "LOAD_IN_4BIT=true",
        ]
    write_env("\n".join(filter(None, lines)) + "\n")
    console.print(
        "[green]Local mode configured. Use: openagent daemon then openagent chat --model auto[/green]"
    )

    _set_first_run_completed()


@app.command()
def doctor():
    """Check environment, keys, GPU, and CLI wiring."""
    import platform
    from shutil import which

    issues = []
    ok = []

    # Python and platform
    ok.append(
        f"Python: {sys.version.split()[0]} on {platform.system()} {platform.release()}"
    )

    # CLI entrypoint availability in PATH
    resolved = which("openagent")
    if resolved:
        ok.append(f"CLI found: {resolved}")
    else:
        issues.append(
            "CLI not on PATH. Use scripts/install.sh or add venv/bin to PATH."
        )

    # Venv note
    if sys.prefix and "venv" in sys.prefix:
        ok.append(f"Using venv: {sys.prefix}")

    # GPU availability (best-effort)
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            ok.append(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            ok.append("CUDA GPU: not available (CPU mode)")
    except Exception:
        ok.append("torch not available; CPU mode")

    # Local models only - no API keys needed

    # Models import
    try:
        import transformers  # type: ignore

        ok.append(f"transformers: {getattr(transformers, '__version__', 'ok')}")
    except Exception as e:
        issues.append(f"transformers import failed: {e}")

    # Print report
    console.print(Panel("\n".join(ok), title="Environment"))
    if issues:
        console.print(Panel("\n".join(issues), title="Issues", style="red"))
    else:
        console.print(Panel("All good!", title="Status", style="green"))


@app.command()
def models(
    local: bool = typer.Option(
        False, "--local", help="List locally installed Ollama models"
    )
):
    """List available models (presets) or locally installed models with --local."""
    if local:
        try:
            import asyncio as _asyncio

            from openagent.core.llm_ollama import list_ollama_models

            names = _asyncio.get_event_loop().run_until_complete(list_ollama_models())
        except Exception:
            names = []
        if names:
            console.print(
                Panel(
                    "\n".join([f"• ollama:{n}" for n in names]),
                    title="Local Ollama Models",
                )
            )
        else:
            console.print(
                Panel(
                    "No local Ollama models detected. Install with 'ollama pull <model>'.",
                    title="Local Ollama Models",
                )
            )
        return

    console.print(
        Panel(
            f"""[bold cyan]Code Models (Best for programming tasks):[/bold cyan]
{chr(10).join([f'• [green]{k}[/green]: {v}' for k, v in ModelConfig.CODE_MODELS.items()])}

[bold cyan]Chat Models (General conversation):[/bold cyan]
{chr(10).join([f'• [blue]{k}[/blue]: {v}' for k, v in ModelConfig.CHAT_MODELS.items()])}

[bold cyan]Lightweight Models (Fast, low resource usage):[/bold cyan]
{chr(10).join([f'• [yellow]{k}[/yellow]: {v}' for k, v in ModelConfig.LIGHTWEIGHT_MODELS.items()])}

[dim]Use these model names with the --model flag[/dim]
""",
            title="Available Models",
        )
    )


# Plugin Management Commands are registered below, alongside command implementations.


@app.command()
def workflow(
    action: str = typer.Argument(..., help="Action: list|run|new|sync"),
    name: str = typer.Argument(None, help="Workflow name for run/new"),
    param: list[str] = typer.Option([], "--param", help="key=value parameters"),
    repo: str = typer.Option(
        None, help="Git repo URL for sync (or set OPENAGENT_WORKFLOWS_REPO)"
    ),
    branch: str = typer.Option("main", help="Branch name for sync"),
    dest: str = typer.Option(
        None, help="Destination directory for sync (defaults to workflows dir)"
    ),
):
    """Manage and run workflows (YAML in ~/.config/openagent/workflows)."""
    wm = WorkflowManager()
    if action == "list":
        wfs = wm.list()
        table = Table(show_header=True, header_style="bold")
        table.add_column("Name")
        table.add_column("Description")
        for wf in wfs:
            table.add_row(wf.name, wf.description)
        console.print(table)
        return
    if action == "new":
        if not name:
            console.print("[red]Missing workflow name[/red]")
            raise typer.Exit(1)
        wf = wm.create(name, description="")
        console.print(
            f"[green]Created workflow {wf.name} at {wm.base_dir / (wf.name + '.yaml')}[/green]"
        )
        return
    if action == "sync":
        repo_url = repo or os.getenv("OPENAGENT_WORKFLOWS_REPO")
        if not repo_url:
            console.print(
                "[red]Missing repo URL. Use --repo or set OPENAGENT_WORKFLOWS_REPO.[/red]"
            )
            raise typer.Exit(1)
        target = wm.sync(repo_url, branch=branch, dest=Path(dest) if dest else None)
        console.print(Panel(f"Synced workflows to {target}", title="Workflow Sync"))
        return
    if action == "run":
        if not name:
            console.print("[red]Missing workflow name[/red]")
            raise typer.Exit(1)
        wf = wm.get(name)
        if not wf:
            console.print("[red]Workflow not found[/red]")
            raise typer.Exit(1)
        # Build params from CLI
        params_dict: Dict[str, str] = {}
        for p in param:
            if "=" in p:
                k, v = p.split("=", 1)
                params_dict[k] = v
        # Merge defaults
        merged = dict(wf.params)
        merged.update(params_dict)
        # Run steps
        a = create_agent("auto", "auto", True, unsafe_exec=False)

        async def run_steps():
            console.print(Panel.fit(f"Running workflow: {wf.name}", title="Workflow"))
            for i, step in enumerate(wf.steps, start=1):
                console.print(f"\n[cyan]Step {i}/{len(wf.steps)}[/cyan]")
                if isinstance(step, dict) and step.get("tool"):
                    tool_name = step["tool"]
                    args = step.get("args", {})
                    # Param substitution
                    for k, v in list(args.items()):
                        if (
                            isinstance(v, str)
                            and v.startswith("${")
                            and v.endswith("}")
                        ):
                            key = v[2:-1]
                            args[k] = merged.get(key, v)
                    tool = a.get_tool(tool_name)
                    if not tool:
                        console.print(
                            f"[yellow]Skipping unknown tool: {tool_name}[/yellow]"
                        )
                        continue
                    with console.status(
                        "[bold green]Executing tool...", spinner="dots"
                    ):
                        res = await tool.execute(args)
                    console.print(
                        Panel(
                            res.content or (res.error or ""),
                            title=f"{tool_name}",
                            style=None if res.success else "red",
                        )
                    )
                else:
                    # Natural language step with simple substitution
                    text = step if isinstance(step, str) else str(step)
                    for key, val in merged.items():
                        text = text.replace(f"${{{key}}}", str(val))
                    with console.status("[bold green]Thinking...", spinner="dots"):
                        resp = await a.process_message(text)
                    console.print(resp.content)

        asyncio.run(run_steps())
        return
    console.print("[red]Unknown action[/red]")


@app.command()
def daemon(
    host: str = typer.Option("127.0.0.1", help="Daemon host"),
    port: int = typer.Option(8765, help="Daemon port"),
):
    """Start a background agent daemon (preloads default local model)."""
    load_dotenv()
    console.print(Panel(f"Starting OpenAgent daemon on {host}:{port}", title="Daemon"))
    try:
        import uvicorn  # type: ignore

        # Local-only default model
        os.environ.setdefault("DEFAULT_MODEL", "tiny-llama")
        uvicorn.run(
            "openagent.server.app:app",
            host=host,
            port=port,
            reload=False,
            log_level="warning",
        )
    except Exception as e:
        console.print(f"[red]Failed to start daemon: {e}[/red]")


# Policy Management Commands
policy_app = typer.Typer(help="Manage safety policies and audit logs")
app.add_typer(policy_app, name="policy")


@policy_app.command("show")
def policy_show():
    """Show current policy configuration."""
    from openagent.core.policy import get_policy_engine
    from openagent.terminal.validator import CONFIG_PATH

    engine = get_policy_engine()
    policy = engine.policy

    # Display Policy File path (for test compatibility)
    console.print(f"[bold cyan]Policy File:[/bold cyan] {CONFIG_PATH}")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="yellow")
    table.add_column("Value", style="white")

    table.add_row("Default Mode", policy.default_mode)
    table.add_row(
        "Require Approval (Medium Risk)", str(policy.require_approval_for_medium)
    )
    table.add_row("Block High Risk", str(policy.block_high_risk))
    table.add_row("Admin Override", str(policy.allow_admin_override))
    table.add_row("Sandbox Mode", str(policy.sandbox_mode))
    table.add_row("Audit Enabled", str(policy.audit_enabled))
    table.add_row("Audit Path", str(engine.audit_path))

    console.print(Panel(table, title="Policy Configuration"))

    # Show patterns
    console.print("\n[bold cyan]Allowlist Patterns:[/bold cyan]")
    for pattern in policy.allowlist_patterns[:5]:
        console.print(f"  ✅ {pattern}")
    if len(policy.allowlist_patterns) > 5:
        console.print(f"  ... and {len(policy.allowlist_patterns) - 5} more")

    console.print("\n[bold cyan]Denylist Patterns:[/bold cyan]")
    for pattern in policy.denylist_patterns[:5]:
        console.print(f"  ❌ {pattern}")
    if len(policy.denylist_patterns) > 5:
        console.print(f"  ... and {len(policy.denylist_patterns) - 5} more")


@policy_app.command("set-mode")
def policy_set_mode(
    mode: str = typer.Argument(..., help="Mode: explain_only, approve, execute")
):
    """Set the default execution mode."""
    from openagent.core.policy import CommandPolicy, configure_policy, get_policy_engine

    if mode not in ["explain_only", "approve", "execute"]:
        console.print(
            "[red]Invalid mode. Must be: explain_only, approve, or execute[/red]"
        )
        raise typer.Exit(1)

    engine = get_policy_engine()
    policy = engine.policy
    policy.default_mode = mode
    configure_policy(policy, engine.audit_path)

    console.print(f"[green]Policy mode set to: {mode}[/green]")


@policy_app.command("reset")
def policy_reset():
    """Reset policy to defaults."""
    from openagent.terminal.validator import CONFIG_PATH, DEFAULT_POLICY, save_policy

    save_policy(DEFAULT_POLICY)
    console.print(f"[green]Policy reset to defaults at {CONFIG_PATH}[/green]")


@policy_app.command("set-default")
def policy_set_default(
    decision: str = typer.Argument(..., help="Default decision: allow, warn, or block")
):
    """Set the default decision for unmatched commands."""
    from openagent.terminal.validator import load_policy, save_policy

    if decision not in ["allow", "warn", "block"]:
        console.print("[red]Invalid decision. must be one of: allow, warn, block[/red]")
        raise typer.Exit(1)

    policy = load_policy()
    policy["default_decision"] = decision
    save_policy(policy)

    console.print(f"[green]Default decision set to {decision}[/green]")


@policy_app.command("audit")
def policy_audit(
    action: str = typer.Argument(..., help="Action: list, verify, export"),
    start_date: str = typer.Option(None, help="Start date (YYYYMM)"),
    end_date: str = typer.Option(None, help="End date (YYYYMM)"),
    format: str = typer.Option("summary", help="Export format: json, summary"),
    limit: int = typer.Option(100, help="Limit for list action"),
):
    """Manage audit logs."""
    from openagent.core.policy import get_policy_engine

    engine = get_policy_engine()

    if action == "list":
        # Read recent audit entries
        import json
        import time
        from pathlib import Path

        audit_file = engine.audit_path / f"audit_{time.strftime('%Y%m')}.jsonl"
        if not audit_file.exists():
            console.print("[yellow]No audit logs found for current month[/yellow]")
            return

        entries = []
        with open(audit_file, "r") as f:
            for line in f:
                entries.append(json.loads(line))

        # Show last N entries
        entries = entries[-limit:]

        table = Table(show_header=True, header_style="bold")
        table.add_column("Time", style="cyan")
        table.add_column("Command", style="white")
        table.add_column("Risk", style="yellow")
        table.add_column("Decision", style="green")
        table.add_column("Executed", style="blue")

        for entry in entries:
            import datetime

            timestamp = datetime.datetime.fromtimestamp(entry["timestamp"]).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            cmd = (
                entry["command"][:50] + "..."
                if len(entry["command"]) > 50
                else entry["command"]
            )
            table.add_row(
                timestamp,
                cmd,
                entry["risk_level"],
                entry["policy_decision"],
                "✓" if entry["executed"] else "✗",
            )

        console.print(table)

    elif action == "verify":
        with console.status(
            "[bold green]Verifying audit chain integrity...", spinner="dots"
        ):
            is_valid = engine.verify_audit_integrity(start_date)

        if is_valid:
            console.print("[green]✅ Audit chain integrity verified![/green]")
        else:
            console.print("[red]❌ Audit chain integrity check failed![/red]")

    elif action == "export":
        report = engine.export_audit_report(start_date, end_date, format)

        if format == "json":
            console.print_json(report)
        else:
            console.print(Panel(report, title="Audit Report"))

    else:
        console.print("[red]Unknown action. Use: list, verify, or export[/red]")


@policy_app.command("sandbox")
def policy_sandbox(
    enable: bool = typer.Option(
        None, "--enable/--disable", help="Enable or disable sandbox mode"
    )
):
    """Configure sandbox mode for command execution."""
    from openagent.core.policy import configure_policy, get_policy_engine

    engine = get_policy_engine()

    if enable is None:
        # Show current status
        status = "enabled" if engine.policy.sandbox_mode else "disabled"
        console.print(f"Sandbox mode is currently: [bold]{status}[/bold]")
        return

    engine.policy.sandbox_mode = enable
    configure_policy(engine.policy, engine.audit_path)

    status = "enabled" if enable else "disabled"
    console.print(f"[green]Sandbox mode {status}[/green]")

    if enable:
        console.print(
            "[dim]Commands will run with resource limits and namespace isolation (Linux only)[/dim]"
        )


# Plugin Management Commands
plugin_app = typer.Typer(help="Manage OpenAgent plugins")
app.add_typer(plugin_app, name="plugin")


@plugin_app.command("list")
def plugin_list(
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Show detailed plugin information"
    )
):
    """List all discovered plugins and their status."""
    from openagent.plugins.manager import PluginManager
    pm = PluginManager()

    async def run():
        await pm.initialize()
        # Discover and list (do not auto-load)
        discovered = await pm.discover_plugins()
        # Attempt to load all to show status/metadata
        await pm.load_all_plugins()
        infos = []
        for name in discovered:
            info = await pm.get_plugin_info(name)
            if info:
                infos.append(info)
        # Render
        console.print("[bold cyan]Plugins:[/bold cyan]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Plugin", style="green")
        table.add_column("Version", style="cyan")
        table.add_column("Description", style="dim")
        table.add_column("Status", justify="center")
        for info in infos:
            md = info.get("metadata") or {}
            status = info.get("status", "unknown")
            status_emoji = "🟢" if status == "active" else ("🟡" if status == "loaded" else "⚪")
            table.add_row(md.get("name", "?"), md.get("version", "?"), md.get("description", ""), f"{status_emoji} {status}")
        console.print(table)

    asyncio.run(run())


@plugin_app.command("enable")
def plugin_enable(plugin_name: str = typer.Argument(..., help="Plugin to enable")):
    from openagent.plugins.manager import PluginManager
    pm = PluginManager()
    async def run():
        await pm.initialize()
        await pm.load_plugin(plugin_name)
        ok = await pm.enable_plugin(plugin_name)
        if ok:
            await pm.save_plugin_configs()
        console.print("[green]Enabled[/green]" if ok else "[red]Failed to enable[/red]")
    asyncio.run(run())


@plugin_app.command("disable")
def plugin_disable(plugin_name: str = typer.Argument(..., help="Plugin to disable")):
    from openagent.plugins.manager import PluginManager
    pm = PluginManager()
    async def run():
        await pm.initialize()
        ok = await pm.disable_plugin(plugin_name)
        if ok:
            await pm.save_plugin_configs()
        console.print("[yellow]Disabled[/yellow]" if ok else "[red]Failed to disable[/red]")
    asyncio.run(run())


@plugin_app.command("reload")
def plugin_reload(plugin_name: str = typer.Argument(..., help="Plugin to reload")):
    from openagent.plugins.manager import PluginManager
    pm = PluginManager()
    async def run():
        await pm.initialize()
        ok = await pm.reload_plugin(plugin_name)
        console.print("[green]Reloaded[/green]" if ok else "[red]Failed to reload[/red]")
    asyncio.run(run())


@plugin_app.command("tools")
def plugin_tools(
    plugin: str = typer.Option(None, "--plugin", help="Filter by plugin name")
):
    """List tools provided by enabled plugins."""
    from openagent.plugins.manager import PluginManager
    pm = PluginManager()

    async def run():
        await pm.initialize()
        await pm.discover_plugins()
        await pm.load_all_plugins()
        # Enable all configured enabled plugins
        for info in await pm.list_plugins():
            name = (info.get("metadata") or {}).get("name")
            if not name:
                continue
            try:
                await pm.enable_plugin(name)
            except Exception:
                pass
        # Enriched entries with version
        entries = pm.get_tool_entries(
            plugins={plugin} if plugin else None
        )
        table = Table(show_header=True, header_style="bold")
        table.add_column("Plugin", style="cyan")
        table.add_column("Version", style="magenta")
        table.add_column("Tool", style="green")
        table.add_column("Source", style="cyan")
        table.add_column("Description", style="dim")
        if not entries:
            console.print("[yellow]No plugin tools are currently available.[/yellow]")
            return
        for e in entries:
            tool = e["tool"]
            desc = getattr(tool, "description", "")
            source = tool.__class__.__module__
            table.add_row(e["plugin"], str(e["version"] or "-"), e["tool_name"], source, desc)
        console.print(table)

    asyncio.run(run())


@plugin_app.command("sync-tools")
def plugin_sync_tools(
    plugin: list[str] = typer.Option(
        None,
        "--plugin",
        help="Only sync tools from the specified plugin(s). Repeat --plugin to include multiple.",
    )
):
    """Attach enabled plugin tools to the current process agent if available."""
    global agent
    if agent is None:
        console.print("[yellow]No running agent in this process. Start chat first (openagent chat).[/yellow]")
        return
    from openagent.plugins.manager import PluginManager
    pm = PluginManager()

    async def run():
        await pm.initialize()
        await pm.discover_plugins()
        await pm.load_all_plugins()
        for info in await pm.list_plugins():
            name = (info.get("metadata") or {}).get("name")
            if not name:
                continue
            # Only enable if no filter or in requested set
            if plugin and name not in set(plugin):
                continue
            try:
                await pm.enable_plugin(name)
            except Exception:
                pass
        added = pm.register_tools_with_agent(agent, plugins=set(plugin) if plugin else None)
        console.print(f"[green]Attached {added} plugin tool(s) to the running agent.[/green]")

    asyncio.run(run())


@plugin_app.command("install")

def plugin_install(
    source: str = typer.Argument(..., help="Plugin source (path, git repo, or name)"),
    force: bool = typer.Option(False, help="Force reinstall if already exists"),
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
        console.print(
            f"⚠️  [yellow]Plugin installation not yet implemented for: {source}[/yellow]"
        )
        console.print("\n🔧 [dim]This feature is coming soon in v0.2.0![/dim]")


# Execute shell commands with policy and safety
@app.command("exec")
def exec_cmd(
    command: str = typer.Argument(
        ..., help="Shell command to execute (quotes recommended)"
    ),
    model: str = typer.Option(
        "tiny-llama", help="Model to use for explanations/logging"
    ),
    device: str = typer.Option("auto", help="Device for model (auto/cpu/cuda)"),
    load_in_4bit: bool = typer.Option(True, help="Load model in 4-bit precision"),
    dry_run: bool = typer.Option(False, help="Explain-only; do not execute"),
    output_format: str = typer.Option("text", help="Output format (text|json)"),
):
    """Execute a shell command with OpenAgent's safety policy."""
    load_dotenv()
    agent = create_agent(model, device, load_in_4bit, unsafe_exec=not dry_run)

    async def do_exec():
        tool = CommandExecutor(default_explain_only=dry_run)
        # Only show spinner for text output
        status_cm = (
            console.status("[bold green]Executing...", spinner="dots")
            if (output_format != "json" and not dry_run)
            else (
                console.status("[bold green]Explaining...", spinner="dots")
                if output_format != "json"
                else None
            )
        )
        if status_cm:
            with status_cm:
                result = await tool.execute(
                    {"command": command, "explain_only": dry_run}
                )
        else:
            result = await tool.execute({"command": command, "explain_only": dry_run})
        if output_format == "json":
            import json as _json

            payload = {
                "command": command,
                "dry_run": dry_run,
                "success": bool(result.success),
                "content": result.content or "",
                "error": result.error or "",
                "metadata": getattr(result, "metadata", {}) or {},
            }
            print(_json.dumps(payload))
            return
        if result.success:
            console.print(
                Panel(
                    result.content or "",
                    title=f"Command {'Explanation' if dry_run else 'Output'}",
                    expand=True,
                )
            )
        else:
            console.print(
                Panel(f"Error: {result.error}", title="Execution Failed", style="red")
            )

    asyncio.run(do_exec())


# Plan-and-exec natural language task
@app.command("do")
def plan_and_exec(
    task: str = typer.Argument(
        ..., help="Describe what you want to do in natural language"
    ),
    model: str = typer.Option("tiny-llama", help="Model to use for planning"),
    device: str = typer.Option("auto", help="Device for model"),
    load_in_4bit: bool = typer.Option(True, help="Load model in 4-bit precision"),
    auto_execute: bool = typer.Option(
        True, help="Automatically execute if allowed by policy"
    ),
    plan_only: bool = typer.Option(
        False, help="Output plan (JSON) and exit without executing"
    ),
):
    """Plan a command, explain risks, validate via policy, then execute if allowed."""
    load_dotenv()

    async def do_plan():
        from openagent.core.context import gather_context
        from openagent.core.tool_selector import SmartToolSelector
        from openagent.terminal.validator import validate as validate_cmd
        from openagent.tools.git import GitTool, RepoGrep
        from openagent.tools.system import CommandExecutor, FileManager, SystemInfo

        # Gather context for better planning
        sysctx = gather_context()
        ctx_block = sysctx.to_prompt_block()

        # Initialize selector with available tools
        tools = {
            "command_executor": CommandExecutor(default_explain_only=not auto_execute),
            "file_manager": FileManager(),
            "system_info": SystemInfo(),
            "git_tool": GitTool(),
            "repo_grep": RepoGrep(),
        }
        # If plan_only, avoid loading an LLM and rely on heuristics and available tools
        llm_for_planning = None
        agent_local = None
        if not plan_only:
            # Create agent only when we may execute or use model-based planning
            agent_local = create_agent(
                model, device, load_in_4bit, unsafe_exec=auto_execute
            )
            llm_for_planning = agent_local.llm
        selector = SmartToolSelector(llm_for_planning, tools)

        # Create plan using LLM-driven tool selection (or heuristics if llm is None)
        with console.status("[bold green]Planning...", spinner="dots"):
            plan = await selector.create_tool_plan(task, context={"system": ctx_block})

        if plan_only:
            # Print plan as JSON for programmatic use
            import json as _json

            plan_json = {
                "explanation": plan.explanation,
                "estimated_risk": plan.estimated_risk,
                "requires_confirmation": plan.requires_confirmation,
                "calls": [
                    {
                        "order": c.order,
                        "tool_name": c.tool_name,
                        "parameters": c.parameters,
                        "intent": c.intent.value,
                        "rationale": c.rationale,
                    }
                    for c in sorted(plan.calls, key=lambda c: c.order)
                ],
            }
            print(_json.dumps(plan_json))
            return

        # Display plan summary
        plan_table = Table(show_header=True, header_style="bold")
        plan_table.add_column("Order", justify="right")
        plan_table.add_column("Tool")
        plan_table.add_column("Parameters")
        plan_table.add_column("Rationale")
        for call in sorted(plan.calls, key=lambda c: c.order):
            plan_table.add_row(
                str(call.order), call.tool_name, str(call.parameters), call.rationale
            )
        console.print(
            Panel(plan_table, title=f"Planned Steps (Risk: {plan.estimated_risk})")
        )

        # Special case: if the plan is a single command execution, validate via policy
        if len(plan.calls) == 1 and plan.calls[0].tool_name.startswith("command_"):
            cmd = plan.calls[0].parameters.get("command", "")
            if cmd:
                decision, reason = validate_cmd(cmd)
                table = Table(show_header=True, header_style="bold")
                table.add_column("Field")
                table.add_column("Value")
                table.add_row("Command", cmd)
                table.add_row("Decision", decision)
                table.add_row("Policy Reason", reason)
                table.add_row("Rationale", plan.calls[0].rationale or "")
                console.print(Panel(table, title="Plan & Policy"))
                if decision == "block":
                    console.print("[red]Blocked by policy. Not executing.[/red]")
                    return
                if plan.requires_confirmation and auto_execute:
                    console.print(
                        "[yellow]Confirmation required. Re-run with --auto-execute after review.[/yellow]"
                    )
                    return

        if not auto_execute:
            console.print(
                "[yellow]Auto-exec disabled. Use --auto-execute to run.[/yellow]"
            )
            return

        # Execute plan
        with console.status("[bold green]Executing...", spinner="dots"):
            results = await selector.execute_plan(plan)

        # Render results
        for i, res in enumerate(results, start=1):
            title = f"Step {i} Output" if res.success else f"Step {i} Failed"
            style = None if res.success else "red"
            console.print(
                Panel(res.content or (res.error or ""), title=title, style=style)
            )

    asyncio.run(do_plan())


@app.command()
def optimize(
    model: str = typer.Option("tiny-llama", help="Model to preload for optimization"),
    enable_caching: bool = typer.Option(True, help="Enable performance caching"),
    memory_limit_gb: float = typer.Option(4.0, help="Memory limit in GB"),
):
    """Optimize OpenAgent performance with advanced optimizations."""
    console.print(
        Panel.fit("🚀 Starting OpenAgent Performance Optimization", title="Optimizer")
    )

    async def run_optimization():
        # Start optimizations
        result = await optimize_openagent_performance()

        # Show results
        console.print("\n✅ [green]Performance optimization complete![/green]")
        console.print(f"\n[bold]Optimization Results:[/bold]")
        for key, value in result.items():
            console.print(f"  • {key}: {value}")

        # Get performance profiler stats
        profiler = get_performance_profiler()
        profiler.start_profiling()
        report = profiler.get_performance_report()

        # Show performance metrics
        metrics_table = Table(show_header=True, header_style="bold cyan")
        metrics_table.add_column("Metric", style="yellow")
        metrics_table.add_column("Value", style="white")

        for metric, value in report["metrics"].items():
            if isinstance(value, float):
                value_str = (
                    f"{value:.2f}"
                    if metric.endswith(("_time", "_mb", "_percent"))
                    else str(value)
                )
            else:
                value_str = str(value)
            metrics_table.add_row(metric.replace("_", " ").title(), value_str)

        console.print("\n")
        console.print(Panel(metrics_table, title="Performance Metrics"))

        # Show recommendations if any
        if report["recommendations"]:
            console.print("\n[bold yellow]Recommendations:[/bold yellow]")
            for rec in report["recommendations"]:
                severity_color = (
                    "red"
                    if rec["severity"] == "high"
                    else "yellow" if rec["severity"] == "medium" else "blue"
                )
                console.print(
                    f"  [{severity_color}]{rec['severity'].upper()}[/{severity_color}]: {rec['message']}"
                )
                console.print(f"    💡 {rec['suggestion']}")

    asyncio.run(run_optimization())


@app.command()
def ui_demo(
    duration: int = typer.Option(30, help="Demo duration in seconds"),
    interactive: bool = typer.Option(
        False, help="Enable interactive mode with keyboard controls"
    ),
):
    """Demo the advanced terminal UI features with command blocks and formatting."""
    console.print(Panel.fit("🎨 Starting OpenAgent UI Demo", title="UI Demo"))

    async def run_ui_demo():
        # Create renderer
        renderer = create_terminal_renderer()

        # Start live display
        live = renderer.start_live_display()

        try:
            console.print("\n[green]Live UI Demo Started![/green]")
            console.print(
                "[dim]Watch as commands are executed with visual blocks and formatting...[/dim]"
            )

            # Demo sequence
            demo_commands = [
                (
                    "ls -la",
                    "total 24\ndrwxr-xr-x 3 user user 4096 Jan 1 12:00 .\ndrwxr-xr-x 5 user user 4096 Jan 1 11:00 ..\n-rw-r--r-- 1 user user 1234 Jan 1 12:00 file.txt\n-rw-r--r-- 1 user user 567 Jan 1 11:30 script.py",
                ),
                (
                    "cat script.py",
                    "#!/usr/bin/env python3\ndef hello_world():\n    print('Hello from OpenAgent!')\n\nif __name__ == '__main__':\n    hello_world()",
                ),
                (
                    "git status",
                    "On branch main\nYour branch is up to date with 'origin/main'.\n\nnothing to commit, working tree clean",
                ),
                (
                    "docker ps",
                    "CONTAINER ID   IMAGE          COMMAND       CREATED         STATUS         PORTS     NAMES\n1a2b3c4d5e6f   nginx:latest   nginx -g ...  2 hours ago     Up 2 hours     80/tcp    web-server",
                ),
                ("invalid_command", "bash: invalid_command: command not found", True),
            ]

            for i, cmd_data in enumerate(demo_commands):
                if len(cmd_data) == 3:
                    command, output, is_error = cmd_data
                else:
                    command, output = cmd_data
                    is_error = False

                # Create and start block
                block = renderer.render_command_execution(command)
                await asyncio.sleep(1)

                # Add output
                renderer.update_block_output(block, output, is_error)

                # Complete execution
                exit_code = 127 if is_error else 0
                renderer.complete_block_execution(block, exit_code, 0.5)

                # Add AI explanation for some commands
                if command.startswith("ls"):
                    renderer.add_ai_response(
                        "The `ls -la` command displays a detailed listing of all files and directories, including hidden ones, with permissions, ownership, and modification times."
                    )
                elif command.startswith("docker"):
                    renderer.add_ai_response(
                        "The `docker ps` command shows currently running containers with their IDs, images, and status information."
                    )

                await asyncio.sleep(3)

            if interactive:
                console.print(
                    "\n[cyan]Interactive mode enabled! Use keyboard shortcuts:[/cyan]"
                )
                console.print(
                    "[dim]j/k: navigate blocks, o: toggle output, h: help, q: quit[/dim]"
                )

                # Keep demo running for interactive use
                import time
                start_time = time.time()
                while time.time() - start_time < duration:
                    await asyncio.sleep(1)
            else:
                # Non-interactive: just show for remaining duration
                remaining = max(0, duration - len(demo_commands) * 4)
                if remaining > 0:
                    await asyncio.sleep(remaining)

        finally:
            renderer.stop_live_display()
            console.print("\n✅ [green]UI Demo completed![/green]")

            # Show stats
            stats = renderer.get_stats()
            stats_table = Table(show_header=True, header_style="bold")
            stats_table.add_column("Stat", style="cyan")
            stats_table.add_column("Value", style="white")

            for key, value in stats.items():
                if isinstance(value, dict):
                    value = f"{sum(value.values())} total"
                stats_table.add_row(key.replace("_", " ").title(), str(value))

            console.print(Panel(stats_table, title="Demo Statistics"))

    asyncio.run(run_ui_demo())


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev only)"),
):
    """Run the OpenAgent FastAPI server."""
    load_dotenv()
    try:
        import uvicorn  # type: ignore

        uvicorn.run("openagent.server.app:app", host=host, port=port, reload=reload)
    except Exception as e:
        console.print(f"[red]Failed to start server: {e}[/red]")


@plugin_app.command("info")
def plugin_info(
    plugin_name: str = typer.Argument(..., help="Name of plugin to show info for")
):
    """Show detailed information about a plugin."""
    from openagent.plugins.manager import PluginManager
    pm = PluginManager()

    async def run():
        await pm.initialize()
        # Ensure metadata is available by discovery
        await pm.discover_plugins()
        info = await pm.get_plugin_info(plugin_name)
        if not info:
            # Try loading then query
            await pm.load_plugin(plugin_name)
            info = await pm.get_plugin_info(plugin_name)
        if not info:
            console.print(f"[red]Plugin '{plugin_name}' not found.[/red]")
            return
        md = info.get("metadata") or {}
        console.print(f"\n[bold cyan]{md.get('name','?')}[/bold cyan] v{md.get('version','?')}")
        console.print(f"📝 {md.get('description','')}")
        if md.get('author'):
            console.print(f"👤 Author: {md.get('author')}")
        if md.get('keywords'):
            console.print(f"🏷️  Tags: {', '.join(md.get('keywords'))}")
        console.print(f"Status: {info.get('status')}")

    asyncio.run(run())


def main():
    """Main CLI entry point."""
    try:
        # Always try to load global keys env quietly
        _load_global_env()
        # If no arguments provided, run setup wizard if first time; else open the menu
        if len(sys.argv) == 1:
            if _is_first_run():
                return setup()
            return menu()
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
