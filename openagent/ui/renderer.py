"""
Terminal Renderer - Integrated block and formatting system.

This module provides the main terminal rendering system that combines
command blocks, output folding, and advanced formatting.
"""

import asyncio
import threading
import time
from typing import Any, Dict, List, Optional, Union

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .blocks import BlockManager, BlockRenderer, BlockStatus, BlockType, CommandBlock
from .formatting import AdvancedFormatter, OutputFolder, OutputType, ProgressTracker


class TerminalRenderer:
    """Main terminal renderer integrating blocks and formatting."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the terminal renderer."""
        self.console = console or Console()
        self.block_manager = BlockManager()
        self.block_renderer = BlockRenderer(self.console)
        self.formatter = AdvancedFormatter(self.console)
        self.output_folder = OutputFolder(self.formatter)
        self.progress_tracker = ProgressTracker(self.console)

        # UI state
        self.live_display: Optional[Live] = None
        self.show_block_list = False
        self.show_help = False
        self.current_mode = "normal"  # normal, block_list, help

        # Keyboard shortcuts
        self.shortcuts = {
            "j": self._next_block,
            "k": self._previous_block,
            "o": self._toggle_output_fold,
            "b": self._toggle_bookmark,
            "d": self._delete_current_block,
            "c": self._clear_all_blocks,
            "l": self._toggle_block_list,
            "h": self._toggle_help,
            "f": self._fold_all_outputs,
            "u": self._unfold_all_outputs,
            "s": self._save_session,
            "r": self._reload_session,
            "q": self._quit_renderer,
            "/": self._search_blocks,
            "n": self._next_search_result,
            "p": self._previous_search_result,
            "e": self._export_blocks,
            "t": self._add_tag_to_block,
            "g": self._go_to_block,
            "1-9": self._select_block_by_number,
        }

    def render_command_execution(
        self, command: str, working_dir: Optional[str] = None
    ) -> CommandBlock:
        """Render a command execution with a new block."""
        block = self.block_manager.create_command_block(command, working_dir)
        block.start_execution()

        if self.live_display:
            self._update_display()

        return block

    def update_block_output(
        self, block: CommandBlock, output: str, is_error: bool = False
    ):
        """Update a block with output or error."""
        if is_error:
            block.error = output
            block.status = BlockStatus.ERROR
        else:
            block.output = output

        # Create foldable sections for long output
        if output and len(output.splitlines()) > 10:
            self.output_folder.clear()
            sections = self.output_folder.add_content(
                output, "Command Output" if not is_error else "Error Output"
            )

            # Auto-fold long outputs
            for section in sections:
                if section.line_count and section.line_count > 20:
                    section.folded = True

        if self.live_display:
            self._update_display()

    def complete_block_execution(
        self, block: CommandBlock, exit_code: int = 0, duration: Optional[float] = None
    ):
        """Complete a block execution."""
        block.complete_execution(exit_code, duration)

        if self.live_display:
            self._update_display()

    def add_ai_response(
        self, response: str, metadata: Optional[Dict] = None
    ) -> CommandBlock:
        """Add an AI response block."""
        block = self.block_manager.create_ai_response_block(response, metadata)

        if self.live_display:
            self._update_display()

        return block

    def start_live_display(self) -> Live:
        """Start live terminal display."""
        if self.live_display is None:
            self.live_display = Live(
                self._build_display(),
                console=self.console,
                refresh_per_second=10,
                auto_refresh=True,
            )
            self.live_display.start()

        return self.live_display

    def stop_live_display(self):
        """Stop live terminal display."""
        if self.live_display:
            self.live_display.stop()
            self.live_display = None

    def _build_display(self) -> Union[Panel, Layout, Group]:
        """Build the main display layout."""
        if self.current_mode == "help":
            return self._build_help_display()
        elif self.current_mode == "block_list":
            return self._build_block_list_display()
        else:
            return self._build_normal_display()

    def _build_normal_display(self) -> Group:
        """Build the normal display with command blocks."""
        elements = []

        # Show recent blocks (last 5)
        recent_blocks = (
            self.block_manager.blocks[-5:] if self.block_manager.blocks else []
        )

        for block in recent_blocks:
            panel = self.block_renderer.render_block(block)
            elements.append(panel)

        # Show status bar
        status_bar = self._build_status_bar()
        elements.append(status_bar)

        return Group(*elements)

    def _build_block_list_display(self) -> Panel:
        """Build the block list display."""
        if not self.block_manager.blocks:
            content = Text("No blocks available", style="dim italic")
        else:
            tree = self.block_renderer.render_block_list(self.block_manager.blocks)
            content = tree

        return Panel(
            content,
            title="Command Blocks",
            subtitle="Press 'l' to return to normal view",
            border_style="cyan",
        )

    def _build_help_display(self) -> Panel:
        """Build the help display."""
        help_table = Table(show_header=True, header_style="bold cyan")
        help_table.add_column("Key", style="yellow", width=8)
        help_table.add_column("Action", style="white")

        shortcuts = [
            ("j/k", "Navigate blocks up/down"),
            ("o", "Toggle output folding"),
            ("b", "Toggle bookmark"),
            ("d", "Delete current block"),
            ("c", "Clear all blocks"),
            ("l", "Toggle block list view"),
            ("h", "Toggle this help"),
            ("f/u", "Fold/unfold all outputs"),
            ("s/r", "Save/reload session"),
            ("/", "Search blocks"),
            ("n/p", "Next/previous search result"),
            ("e", "Export blocks"),
            ("t", "Add tag to block"),
            ("g", "Go to block by ID"),
            ("1-9", "Select block by number"),
            ("q", "Quit renderer"),
        ]

        for key, action in shortcuts:
            help_table.add_row(key, action)

        return Panel(
            help_table,
            title="OpenAgent Terminal UI - Keyboard Shortcuts",
            subtitle="Press 'h' to close help",
            border_style="green",
        )

    def _build_status_bar(self) -> Panel:
        """Build the status bar."""
        selected_block = self.block_manager.get_selected_block()
        total_blocks = len(self.block_manager.blocks)

        status_parts = [
            f"Blocks: {total_blocks}",
        ]

        if selected_block:
            status_parts.append(f"Selected: {selected_block.id}")
            status_parts.append(f"Status: {selected_block.status.value}")

        status_parts.append("Press 'h' for help")

        status_text = " | ".join(status_parts)

        return Panel(Text(status_text, style="dim"), height=3, border_style="dim")

    def _update_display(self):
        """Update the live display."""
        if self.live_display:
            self.live_display.update(self._build_display())

    # Keyboard shortcut handlers
    def _next_block(self):
        """Select next block."""
        self.block_manager.select_next_block()
        self._update_display()

    def _previous_block(self):
        """Select previous block."""
        self.block_manager.select_previous_block()
        self._update_display()

    def _toggle_output_fold(self):
        """Toggle output folding for current block."""
        block = self.block_manager.get_selected_block()
        if block:
            self.block_manager.toggle_block_collapsed()
            self._update_display()

    def _toggle_bookmark(self):
        """Toggle bookmark for current block."""
        block = self.block_manager.get_selected_block()
        if block:
            self.block_manager.bookmark_block()
            self._update_display()

    def _delete_current_block(self):
        """Delete the current block."""
        block = self.block_manager.get_selected_block()
        if block:
            self.block_manager.delete_block(block.id)
            self._update_display()

    def _clear_all_blocks(self):
        """Clear all blocks."""
        self.block_manager.clear_blocks()
        self.output_folder.clear()
        self._update_display()

    def _toggle_block_list(self):
        """Toggle block list view."""
        if self.current_mode == "block_list":
            self.current_mode = "normal"
        else:
            self.current_mode = "block_list"
        self._update_display()

    def _toggle_help(self):
        """Toggle help display."""
        if self.current_mode == "help":
            self.current_mode = "normal"
        else:
            self.current_mode = "help"
        self._update_display()

    def _fold_all_outputs(self):
        """Fold all output sections."""
        for block in self.block_manager.blocks:
            block.collapsed = True
        self.output_folder.fold_all()
        self._update_display()

    def _unfold_all_outputs(self):
        """Unfold all output sections."""
        for block in self.block_manager.blocks:
            block.collapsed = False
        self.output_folder.unfold_all()
        self._update_display()

    def _save_session(self):
        """Save current session."""
        # TODO: Implement session saving
        pass

    def _reload_session(self):
        """Reload saved session."""
        # TODO: Implement session loading
        pass

    def _quit_renderer(self):
        """Quit the renderer."""
        self.stop_live_display()

    def _search_blocks(self):
        """Search blocks."""
        # TODO: Implement block search
        pass

    def _next_search_result(self):
        """Go to next search result."""
        # TODO: Implement search navigation
        pass

    def _previous_search_result(self):
        """Go to previous search result."""
        # TODO: Implement search navigation
        pass

    def _export_blocks(self):
        """Export blocks."""
        # TODO: Implement block export
        pass

    def _add_tag_to_block(self):
        """Add tag to current block."""
        # TODO: Implement tag addition
        pass

    def _go_to_block(self):
        """Go to block by ID."""
        # TODO: Implement block navigation by ID
        pass

    def _select_block_by_number(self, number: int):
        """Select block by number."""
        if 1 <= number <= len(self.block_manager.blocks):
            self.block_manager.select_block(number - 1)
            self._update_display()

    def handle_keypress(self, key: str) -> bool:
        """Handle keyboard input."""
        if key in self.shortcuts:
            self.shortcuts[key]()
            return True
        elif key.isdigit():
            self._select_block_by_number(int(key))
            return True
        return False

    def render_progress(self, description: str, total: Optional[int] = None) -> str:
        """Start rendering progress for a long operation."""
        task_name = self.progress_tracker.add_task(description, total)
        return task_name

    def update_progress(
        self, task_name: str, advance: int = 1, description: Optional[str] = None
    ):
        """Update progress for a task."""
        self.progress_tracker.update_task(task_name, advance, description)

    def complete_progress(self, task_name: str):
        """Complete a progress task."""
        self.progress_tracker.complete_task(task_name)

    def render_table(self, data: List[Dict[str, Any]], title: str = "Data") -> Panel:
        """Render data as a formatted table."""
        if not data:
            return Panel(Text("No data to display", style="dim"), title=title)

        # Get column names from first row
        columns = list(data[0].keys())

        table = Table(show_header=True, header_style="bold")
        for col in columns:
            table.add_column(col, style="white")

        # Add rows
        for row in data:
            table.add_row(*[str(row.get(col, "")) for col in columns])

        return Panel(table, title=title, border_style="cyan")

    def render_tree_data(self, data: Dict[str, Any], title: str = "Tree") -> Panel:
        """Render hierarchical data as a tree."""
        from rich.tree import Tree

        def build_tree(node_data: Dict[str, Any], tree_node: Tree):
            for key, value in node_data.items():
                if isinstance(value, dict):
                    sub_node = tree_node.add(f"📁 {key}")
                    build_tree(value, sub_node)
                elif isinstance(value, list):
                    sub_node = tree_node.add(f"📋 {key} ({len(value)} items)")
                    for i, item in enumerate(value[:10]):  # Limit to first 10 items
                        if isinstance(item, dict):
                            item_node = sub_node.add(f"[{i}]")
                            build_tree(item, item_node)
                        else:
                            sub_node.add(f"[{i}] {str(item)[:50]}")
                    if len(value) > 10:
                        sub_node.add(f"... and {len(value) - 10} more items")
                else:
                    tree_node.add(f"📄 {key}: {str(value)[:100]}")

        tree = Tree(title)
        build_tree(data, tree)

        return Panel(tree, title=title, border_style="green")

    def render_diff(
        self, old_content: str, new_content: str, title: str = "Diff"
    ) -> Panel:
        """Render a diff between two text contents."""
        import difflib

        diff_lines = list(
            difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile="old",
                tofile="new",
                lineterm="",
            )
        )

        diff_content = "".join(diff_lines)
        formatted_diff = self.formatter.format_content(diff_content, OutputType.DIFF)

        return Panel(formatted_diff, title=title, border_style="magenta")

    def get_stats(self) -> Dict[str, Any]:
        """Get renderer statistics."""
        return {
            "total_blocks": len(self.block_manager.blocks),
            "selected_block": self.block_manager.selected_index,
            "current_mode": self.current_mode,
            "live_display_active": self.live_display is not None,
            "blocks_by_status": {
                status.value: len(self.block_manager.get_blocks_by_status(status))
                for status in BlockStatus
            },
            "blocks_by_type": {
                block_type.value: len(self.block_manager.get_blocks_by_type(block_type))
                for block_type in BlockType
            },
        }


class InteractiveRenderer:
    """Interactive terminal renderer with keyboard input handling."""

    def __init__(self, terminal_renderer: Optional[TerminalRenderer] = None):
        """Initialize interactive renderer."""
        self.renderer = terminal_renderer or TerminalRenderer()
        self.running = False
        self.input_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the interactive renderer."""
        self.running = True
        self.renderer.start_live_display()

        # Start input handling in a separate thread
        self.input_thread = threading.Thread(target=self._handle_input, daemon=True)
        self.input_thread.start()

        return self.renderer.live_display

    def stop(self):
        """Stop the interactive renderer."""
        self.running = False
        self.renderer.stop_live_display()

        if self.input_thread:
            self.input_thread.join(timeout=1.0)

    def _handle_input(self):
        """Handle keyboard input in a separate thread."""
        import sys
        import termios
        import tty

        # Set terminal to raw mode for single character input
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setraw(sys.stdin.fileno())

            while self.running:
                try:
                    char = sys.stdin.read(1)
                    if char:
                        # Handle special keys
                        if ord(char) == 3:  # Ctrl+C
                            break
                        elif ord(char) == 27:  # Escape sequence
                            # Read the rest of the escape sequence
                            char += sys.stdin.read(2)
                            if char == "\x1b[A":  # Up arrow
                                self.renderer.handle_keypress("k")
                            elif char == "\x1b[B":  # Down arrow
                                self.renderer.handle_keypress("j")
                        else:
                            # Regular character
                            self.renderer.handle_keypress(char)

                    time.sleep(0.1)  # Small delay to prevent high CPU usage

                except (KeyboardInterrupt, EOFError):
                    break

        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.stop()


# Convenience functions for easy usage
def create_terminal_renderer(console: Optional[Console] = None) -> TerminalRenderer:
    """Create a new terminal renderer."""
    return TerminalRenderer(console)


def create_interactive_renderer(
    console: Optional[Console] = None,
) -> InteractiveRenderer:
    """Create a new interactive terminal renderer."""
    renderer = TerminalRenderer(console)
    return InteractiveRenderer(renderer)


async def demo_renderer():
    """Demo function showing the renderer capabilities."""
    renderer = create_terminal_renderer()

    # Start live display
    live = renderer.start_live_display()

    try:
        # Simulate some command executions
        block1 = renderer.render_command_execution("ls -la")
        await asyncio.sleep(1)

        renderer.update_block_output(
            block1,
            "total 24\ndrwxr-xr-x 3 user user 4096 Jan 1 12:00 .\ndrwxr-xr-x 5 user user 4096 Jan 1 11:00 ..\n-rw-r--r-- 1 user user 1234 Jan 1 12:00 file.txt",
        )
        renderer.complete_block_execution(block1, 0, 0.5)

        await asyncio.sleep(2)

        # Add an AI response
        ai_block = renderer.add_ai_response(
            "The `ls -la` command lists all files and directories in the current directory with detailed information including permissions, ownership, size, and modification time."
        )

        await asyncio.sleep(3)

        # Simulate an error command
        error_block = renderer.render_command_execution("invalid_command")
        await asyncio.sleep(1)

        renderer.update_block_output(
            error_block, "bash: invalid_command: command not found", is_error=True
        )
        renderer.complete_block_execution(error_block, 127, 0.1)

        await asyncio.sleep(5)

    finally:
        renderer.stop_live_display()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_renderer())
