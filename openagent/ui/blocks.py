"""
Command Block System - Warp-style visual command blocks.

This module implements a visual block-based terminal interface
that groups commands and outputs into manageable blocks.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich.console import Console, ConsoleOptions, RenderResult
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree


class BlockType(Enum):
    """Types of command blocks."""

    COMMAND = "command"
    OUTPUT = "output"
    ERROR = "error"
    AI_RESPONSE = "ai_response"
    SYSTEM = "system"
    WORKFLOW = "workflow"


class BlockStatus(Enum):
    """Status of a command block."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class CommandBlock:
    """A visual command block containing command, output, and metadata."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    block_type: BlockType = BlockType.COMMAND
    status: BlockStatus = BlockStatus.PENDING
    command: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None
    ai_explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration: Optional[float] = None
    working_directory: Optional[str] = None
    exit_code: Optional[int] = None
    collapsed: bool = False
    selected: bool = False
    bookmarked: bool = False
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize block after creation."""
        if self.working_directory is None:
            self.working_directory = str(Path.cwd())

    def start_execution(self):
        """Mark block as running."""
        self.status = BlockStatus.RUNNING
        self.timestamp = time.time()

    def complete_execution(self, exit_code: int = 0, duration: Optional[float] = None):
        """Mark block as completed."""
        self.exit_code = exit_code
        self.status = BlockStatus.SUCCESS if exit_code == 0 else BlockStatus.ERROR
        if duration is not None:
            self.duration = duration
        elif self.timestamp:
            self.duration = time.time() - self.timestamp

    def toggle_collapsed(self):
        """Toggle output collapsed state."""
        self.collapsed = not self.collapsed

    def add_tag(self, tag: str):
        """Add a tag to the block."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str):
        """Remove a tag from the block."""
        if tag in self.tags:
            self.tags.remove(tag)

    def get_summary(self) -> str:
        """Get a one-line summary of the block."""
        if self.command:
            cmd_preview = (
                self.command[:50] + "..." if len(self.command) > 50 else self.command
            )
            return f"[{self.status.value}] {cmd_preview}"
        elif self.block_type == BlockType.AI_RESPONSE:
            return f"[AI] Response block"
        else:
            return f"[{self.block_type.value}] Block {self.id}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary for serialization."""
        return {
            "id": self.id,
            "block_type": self.block_type.value,
            "status": self.status.value,
            "command": self.command,
            "output": self.output,
            "error": self.error,
            "ai_explanation": self.ai_explanation,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "working_directory": self.working_directory,
            "exit_code": self.exit_code,
            "collapsed": self.collapsed,
            "selected": self.selected,
            "bookmarked": self.bookmarked,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommandBlock":
        """Create block from dictionary."""
        block = cls()
        block.id = data.get("id", block.id)
        block.block_type = BlockType(data.get("block_type", "command"))
        block.status = BlockStatus(data.get("status", "pending"))
        block.command = data.get("command")
        block.output = data.get("output")
        block.error = data.get("error")
        block.ai_explanation = data.get("ai_explanation")
        block.metadata = data.get("metadata", {})
        block.timestamp = data.get("timestamp", time.time())
        block.duration = data.get("duration")
        block.working_directory = data.get("working_directory")
        block.exit_code = data.get("exit_code")
        block.collapsed = data.get("collapsed", False)
        block.selected = data.get("selected", False)
        block.bookmarked = data.get("bookmarked", False)
        block.tags = data.get("tags", [])
        return block


class BlockManager:
    """Manages command blocks and provides block operations."""

    def __init__(self, max_blocks: int = 1000):
        """Initialize block manager."""
        self.blocks: List[CommandBlock] = []
        self.max_blocks = max_blocks
        self.selected_index: Optional[int] = None
        self.console = Console()
        self._callbacks: Dict[str, List[callable]] = {
            "block_added": [],
            "block_updated": [],
            "block_removed": [],
            "selection_changed": [],
        }

    def add_block(self, block: CommandBlock) -> CommandBlock:
        """Add a new block."""
        self.blocks.append(block)

        # Trim old blocks if we exceed max
        if len(self.blocks) > self.max_blocks:
            removed_blocks = self.blocks[: -self.max_blocks]
            self.blocks = self.blocks[-self.max_blocks :]
            for removed_block in removed_blocks:
                self._trigger_callback("block_removed", removed_block)

        # Auto-select the new block
        self.selected_index = len(self.blocks) - 1
        self._trigger_callback("block_added", block)
        self._trigger_callback("selection_changed", self.selected_index)

        return block

    def create_command_block(
        self, command: str, working_dir: Optional[str] = None
    ) -> CommandBlock:
        """Create and add a new command block."""
        block = CommandBlock(
            block_type=BlockType.COMMAND,
            command=command,
            working_directory=working_dir or str(Path.cwd()),
        )
        return self.add_block(block)

    def create_ai_response_block(
        self, response: str, metadata: Optional[Dict] = None
    ) -> CommandBlock:
        """Create and add a new AI response block."""
        block = CommandBlock(
            block_type=BlockType.AI_RESPONSE, output=response, metadata=metadata or {}
        )
        return self.add_block(block)

    def get_block(self, block_id: str) -> Optional[CommandBlock]:
        """Get block by ID."""
        for block in self.blocks:
            if block.id == block_id:
                return block
        return None

    def get_selected_block(self) -> Optional[CommandBlock]:
        """Get currently selected block."""
        if self.selected_index is not None and 0 <= self.selected_index < len(
            self.blocks
        ):
            return self.blocks[self.selected_index]
        return None

    def select_block(self, index: int) -> bool:
        """Select block by index."""
        if 0 <= index < len(self.blocks):
            # Deselect previous block
            if self.selected_index is not None:
                self.blocks[self.selected_index].selected = False

            # Select new block
            self.selected_index = index
            self.blocks[index].selected = True
            self._trigger_callback("selection_changed", index)
            return True
        return False

    def select_next_block(self) -> bool:
        """Select next block."""
        if not self.blocks:
            return False

        if self.selected_index is None:
            return self.select_block(0)

        next_index = (self.selected_index + 1) % len(self.blocks)
        return self.select_block(next_index)

    def select_previous_block(self) -> bool:
        """Select previous block."""
        if not self.blocks:
            return False

        if self.selected_index is None:
            return self.select_block(len(self.blocks) - 1)

        prev_index = (self.selected_index - 1) % len(self.blocks)
        return self.select_block(prev_index)

    def toggle_block_collapsed(self, block_id: Optional[str] = None) -> bool:
        """Toggle collapsed state of block."""
        block = None
        if block_id:
            block = self.get_block(block_id)
        else:
            block = self.get_selected_block()

        if block:
            block.toggle_collapsed()
            self._trigger_callback("block_updated", block)
            return True
        return False

    def bookmark_block(self, block_id: Optional[str] = None) -> bool:
        """Toggle bookmark state of block."""
        block = None
        if block_id:
            block = self.get_block(block_id)
        else:
            block = self.get_selected_block()

        if block:
            block.bookmarked = not block.bookmarked
            self._trigger_callback("block_updated", block)
            return True
        return False

    def delete_block(self, block_id: str) -> bool:
        """Delete a block."""
        for i, block in enumerate(self.blocks):
            if block.id == block_id:
                removed_block = self.blocks.pop(i)

                # Update selection if needed
                if self.selected_index == i:
                    if i < len(self.blocks):
                        self.select_block(i)
                    elif len(self.blocks) > 0:
                        self.select_block(len(self.blocks) - 1)
                    else:
                        self.selected_index = None
                elif self.selected_index is not None and self.selected_index > i:
                    self.selected_index -= 1

                self._trigger_callback("block_removed", removed_block)
                return True
        return False

    def clear_blocks(self):
        """Clear all blocks."""
        removed_blocks = self.blocks.copy()
        self.blocks.clear()
        self.selected_index = None

        for block in removed_blocks:
            self._trigger_callback("block_removed", block)

    def get_blocks_by_tag(self, tag: str) -> List[CommandBlock]:
        """Get all blocks with a specific tag."""
        return [block for block in self.blocks if tag in block.tags]

    def get_blocks_by_status(self, status: BlockStatus) -> List[CommandBlock]:
        """Get all blocks with a specific status."""
        return [block for block in self.blocks if block.status == status]

    def get_blocks_by_type(self, block_type: BlockType) -> List[CommandBlock]:
        """Get all blocks of a specific type."""
        return [block for block in self.blocks if block.block_type == block_type]

    def search_blocks(self, query: str) -> List[CommandBlock]:
        """Search blocks by command or output content."""
        query_lower = query.lower()
        results = []

        for block in self.blocks:
            if (
                (block.command and query_lower in block.command.lower())
                or (block.output and query_lower in block.output.lower())
                or (block.error and query_lower in block.error.lower())
            ):
                results.append(block)

        return results

    def export_blocks(self, format: str = "json") -> Union[str, Dict]:
        """Export blocks in specified format."""
        if format == "json":
            import json

            return json.dumps([block.to_dict() for block in self.blocks], indent=2)
        elif format == "markdown":
            return self._export_to_markdown()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_to_markdown(self) -> str:
        """Export blocks to markdown format."""
        lines = ["# OpenAgent Command History\n"]

        for block in self.blocks:
            lines.append(f"## Block {block.id}")
            lines.append(f"**Type:** {block.block_type.value}")
            lines.append(f"**Status:** {block.status.value}")
            lines.append(f"**Timestamp:** {time.ctime(block.timestamp)}")

            if block.command:
                lines.append(f"**Command:**")
                lines.append(f"```bash")
                lines.append(block.command)
                lines.append(f"```")

            if block.output:
                lines.append(f"**Output:**")
                lines.append(f"```")
                lines.append(block.output)
                lines.append(f"```")

            if block.error:
                lines.append(f"**Error:**")
                lines.append(f"```")
                lines.append(block.error)
                lines.append(f"```")

            if block.tags:
                lines.append(f"**Tags:** {', '.join(block.tags)}")

            lines.append("")  # Empty line between blocks

        return "\n".join(lines)

    def add_callback(self, event: str, callback: callable):
        """Add event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def remove_callback(self, event: str, callback: callable):
        """Remove event callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def _trigger_callback(self, event: str, *args):
        """Trigger event callbacks."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(*args)
                except Exception as e:
                    # Log error but don't crash
                    print(f"Callback error: {e}")


class BlockRenderer:
    """Renders command blocks with Rich formatting."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize block renderer."""
        self.console = console or Console()

    def render_block(self, block: CommandBlock, width: Optional[int] = None) -> Panel:
        """Render a single command block with enhanced Warp-style visual formatting."""
        from rich.layout import Layout
        from rich.table import Table
        from rich.console import Group
        
        # Create main content group
        content_elements = []
        
        # 1. Command header with status indicator and metadata
        if block.command:
            cmd_header = self._render_command_header(block)
            content_elements.append(cmd_header)
            content_elements.append(Text(""))  # Spacing
        
        # 2. Output sections with smart folding
        if not block.collapsed:
            if block.output:
                output_section = self._render_output_section(block.output, "Output", "white")
                content_elements.append(output_section)
                
            if block.error:
                error_section = self._render_output_section(block.error, "Error", "red")
                content_elements.append(error_section)
                
            if block.ai_explanation:
                ai_section = self._render_ai_section(block.ai_explanation)
                content_elements.append(ai_section)
        else:
            # Collapsed state with summary
            if block.output or block.error or block.ai_explanation:
                collapse_info = self._render_collapsed_info(block)
                content_elements.append(collapse_info)

        # Combine all content
        if content_elements:
            content = Group(*content_elements)
        else:
            content = Text("(empty block)", style="dim italic")

        # Enhanced border styling with status-aware colors
        border_style, title_style = self._get_block_styling(block)
        
        # Enhanced title with rich metadata
        title = self._create_enhanced_title(block, title_style)

        # Create panel with enhanced styling
        panel = Panel(
            content,
            title=title,
            border_style=border_style,
            expand=False,
            width=width,
            padding=(0, 1),
            style="on black" if block.selected else None
        )

        return panel
    
    def _render_command_header(self, block: CommandBlock) -> Table:
        """Render an enhanced command header with status and timing info."""
        table = Table.grid(expand=True)
        table.add_column("command", ratio=1)
        table.add_column("meta", justify="right", style="dim")
        
        # Status indicator
        status_indicators = {
            BlockStatus.PENDING: ("⏳", "yellow"),
            BlockStatus.RUNNING: ("🔄", "blue"),
            BlockStatus.SUCCESS: ("✅", "green"),
            BlockStatus.ERROR: ("❌", "red"),
            BlockStatus.CANCELLED: ("🚫", "red"),
        }
        
        indicator, color = status_indicators.get(block.status, ("❓", "white"))
        
        cmd_text = Text(f"{indicator} {block.command}", style=f"bold {color}")
        
        # Metadata (timing, directory, etc.)
        meta_parts = []
        if block.duration:
            meta_parts.append(f"{block.duration:.2f}s")
        if block.working_directory:
            meta_parts.append(f"📁 {block.working_directory.split('/')[-1]}")
        if block.exit_code is not None:
            meta_parts.append(f"exit:{block.exit_code}")
            
        meta_text = " | ".join(meta_parts) if meta_parts else ""
        
        table.add_row(cmd_text, Text(meta_text, style="dim"))
        return table
    
    def _render_output_section(self, content: str, section_name: str, style: str) -> Panel:
        """Render an output section with smart truncation and folding."""
        lines = content.splitlines()
        
        # Smart truncation for very long output
        if len(lines) > 50:
            display_lines = lines[:25] + ["... (truncated, press 'o' to see full output)"] + lines[-10:]
            display_content = "\n".join(display_lines)
        else:
            display_content = content
            
        # Syntax highlighting for common patterns
        if section_name == "Error":
            # Highlight common error patterns
            display_content = self._highlight_error_patterns(display_content)
            
        return Panel(
            Text(display_content, style=style),
            title=f"📄 {section_name} ({len(lines)} lines)",
            title_align="left",
            border_style="dim",
            expand=False,
            padding=(0, 1)
        )
    
    def _render_ai_section(self, explanation: str) -> Panel:
        """Render AI explanation section with special styling."""
        return Panel(
            Text(explanation, style="italic cyan"),
            title="🤖 AI Assistant",
            title_align="left",
            border_style="cyan",
            expand=False,
            padding=(0, 1)
        )
    
    def _render_collapsed_info(self, block: CommandBlock) -> Text:
        """Render collapsed state summary."""
        info_parts = []
        if block.output:
            lines = len(block.output.splitlines())
            info_parts.append(f"📄 {lines} lines output")
        if block.error:
            error_lines = len(block.error.splitlines())
            info_parts.append(f"❌ {error_lines} lines error")
        if block.ai_explanation:
            info_parts.append("🤖 AI explanation")
            
        summary = " | ".join(info_parts)
        return Text(f"▶️ {summary} (press 'o' to expand)", style="dim italic")
    
    def _get_block_styling(self, block: CommandBlock) -> tuple[str, str]:
        """Get enhanced border and title styling based on block state."""
        base_styles = {
            BlockStatus.PENDING: ("blue", "blue"),
            BlockStatus.RUNNING: ("yellow", "yellow"),
            BlockStatus.SUCCESS: ("green", "green"),
            BlockStatus.ERROR: ("red", "red"),
            BlockStatus.CANCELLED: ("magenta", "magenta"),
        }
        
        border_style, title_style = base_styles.get(block.status, ("white", "white"))
        
        # Enhanced styling for selected blocks
        if block.selected:
            border_style = f"bold {border_style}"
            title_style = f"bold {title_style}"
            
        return border_style, title_style
    
    def _create_enhanced_title(self, block: CommandBlock, title_style: str) -> Text:
        """Create an enhanced title with rich metadata."""
        # Base title with block ID
        title_parts = [f"#{block.id}"]
        
        # Add timing if available
        if block.duration:
            title_parts.append(f"⏱️ {block.duration:.2f}s")
            
        # Add bookmark indicator
        if block.bookmarked:
            title_parts.append("📌")
            
        # Add tags
        if block.tags:
            tags_str = ", ".join(block.tags)
            title_parts.append(f"🏷️ [{tags_str}]")
            
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.fromtimestamp(block.timestamp).strftime("%H:%M:%S")
        title_parts.append(f"🕐 {timestamp}")
        
        title_text = " | ".join(title_parts)
        return Text(title_text, style=title_style)
    
    def _highlight_error_patterns(self, error_text: str) -> str:
        """Add basic error pattern highlighting."""
        # This is a simple implementation - could be enhanced with regex patterns
        highlighted = error_text
        
        # Common error patterns
        patterns = [
            ("Error:", "[red]Error:[/red]"),
            ("Exception:", "[red]Exception:[/red]"),
            ("Failed", "[red]Failed[/red]"),
            ("command not found", "[yellow]command not found[/yellow]"),
            ("Permission denied", "[red]Permission denied[/red]"),
        ]
        
        for pattern, replacement in patterns:
            highlighted = highlighted.replace(pattern, replacement)
            
        return highlighted

    def render_block_list(
        self, blocks: List[CommandBlock], width: Optional[int] = None
    ) -> Tree:
        """Render a list of blocks as a tree."""
        tree = Tree("Command Blocks")

        for i, block in enumerate(blocks):
            node_text = block.get_summary()
            if block.selected:
                node_text = f"► {node_text}"

            node = tree.add(node_text)

            # Add metadata as sub-nodes
            if block.working_directory:
                node.add(f"📁 {block.working_directory}")

            if block.duration:
                node.add(f"⏱️ {block.duration:.2f}s")

            if block.exit_code is not None:
                node.add(f"🔢 Exit code: {block.exit_code}")

        return tree

    def render_block_summary(self, block: CommandBlock) -> Text:
        """Render a one-line summary of a block."""
        # Status indicator
        status_indicators = {
            BlockStatus.PENDING: "⏳",
            BlockStatus.RUNNING: "🔄",
            BlockStatus.SUCCESS: "✅",
            BlockStatus.ERROR: "❌",
            BlockStatus.CANCELLED: "🚫",
        }

        indicator = status_indicators.get(block.status, "❓")

        # Build summary text
        summary_parts = [
            indicator,
            f"[{block.id}]",
        ]

        if block.command:
            cmd_preview = (
                block.command[:40] + "..." if len(block.command) > 40 else block.command
            )
            summary_parts.append(f"`{cmd_preview}`")

        if block.bookmarked:
            summary_parts.append("📌")

        summary_text = " ".join(summary_parts)

        # Style based on status
        style = (
            "green"
            if block.status == BlockStatus.SUCCESS
            else (
                "red"
                if block.status == BlockStatus.ERROR
                else "yellow" if block.status == BlockStatus.RUNNING else "white"
            )
        )

        if block.selected:
            style = f"bold {style}"

        return Text(summary_text, style=style)
