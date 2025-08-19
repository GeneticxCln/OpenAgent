"""
Advanced Terminal Formatting and Output Folding.

This module provides enhanced text formatting, syntax highlighting,
and output folding capabilities for better terminal experience.
"""

import re
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from rich.console import Console, ConsoleOptions, RenderResult
from rich.highlighter import ReprHighlighter
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


class OutputType(Enum):
    """Types of output content for different formatting."""

    PLAIN = "plain"
    CODE = "code"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    LOG = "log"
    ERROR = "error"
    DIFF = "diff"
    MARKDOWN = "markdown"
    SQL = "sql"
    SHELL = "shell"


@dataclass
class FoldableSection:
    """A section of output that can be folded/expanded."""

    title: str
    content: str
    output_type: OutputType = OutputType.PLAIN
    folded: bool = True
    line_count: Optional[int] = None
    max_preview_lines: int = 3

    def __post_init__(self):
        """Calculate line count after initialization."""
        if self.line_count is None:
            self.line_count = len(self.content.splitlines())

    def get_preview(self) -> str:
        """Get a preview of the folded content."""
        lines = self.content.splitlines()
        if len(lines) <= self.max_preview_lines:
            return self.content

        preview_lines = lines[: self.max_preview_lines]
        remaining = len(lines) - self.max_preview_lines
        preview_lines.append(f"... and {remaining} more lines (press 'o' to expand)")

        return "\n".join(preview_lines)

    def toggle_fold(self):
        """Toggle the folded state."""
        self.folded = not self.folded


class AdvancedFormatter:
    """Advanced text formatter with syntax highlighting and folding."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the formatter."""
        self.console = console or Console()
        self.highlighter = ReprHighlighter()

        # Language detection patterns
        self.language_patterns = {
            "python": [
                r"def\s+\w+",
                r"import\s+\w+",
                r"from\s+\w+\s+import",
                r"class\s+\w+",
            ],
            "javascript": [
                r"function\s+\w+",
                r"var\s+\w+",
                r"let\s+\w+",
                r"const\s+\w+",
                r"=>",
            ],
            "json": [r"^\s*[{\[]", r'"\w+":\s*'],
            "yaml": [r"^\w+:", r"^\s+-\s+"],
            "xml": [r"<\w+.*?>", r"</\w+>"],
            "sql": [
                r"\bSELECT\b",
                r"\bFROM\b",
                r"\bWHERE\b",
                r"\bINSERT\b",
                r"\bUPDATE\b",
            ],
            "shell": [r"^\$", r"#!/bin/(bash|sh)", r"\|\s*\w+", r"&&\s*\w+"],
            "diff": [r"^[\+\-@]", r"^diff\s+", r"^index\s+"],
            "log": [r"\d{4}-\d{2}-\d{2}", r"ERROR", r"WARNING", r"INFO", r"DEBUG"],
        }

    def detect_output_type(self, content: str) -> OutputType:
        """Detect the type of output content."""
        content_lower = content.lower()
        content_lines = content.splitlines()

        # Check for specific patterns
        if any(
            re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            for pattern in self.language_patterns["json"]
        ):
            return OutputType.JSON

        if any(
            re.search(pattern, content, re.MULTILINE)
            for pattern in self.language_patterns["xml"]
        ):
            return OutputType.XML

        if any(
            re.search(pattern, content, re.MULTILINE)
            for pattern in self.language_patterns["yaml"]
        ):
            return OutputType.YAML

        if any(
            re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            for pattern in self.language_patterns["sql"]
        ):
            return OutputType.SQL

        if any(
            re.search(pattern, content, re.MULTILINE)
            for pattern in self.language_patterns["diff"]
        ):
            return OutputType.DIFF

        if any(
            re.search(pattern, content, re.MULTILINE)
            for pattern in self.language_patterns["shell"]
        ):
            return OutputType.SHELL

        if any(
            re.search(pattern, content, re.MULTILINE)
            for pattern in self.language_patterns["log"]
        ):
            return OutputType.LOG

        # Check for common programming languages
        for lang, patterns in self.language_patterns.items():
            if lang in ["python", "javascript"]:
                if any(
                    re.search(pattern, content, re.MULTILINE) for pattern in patterns
                ):
                    return OutputType.CODE

        # Check for error patterns
        error_indicators = ["error", "exception", "traceback", "failed", "fatal"]
        if any(indicator in content_lower for indicator in error_indicators):
            return OutputType.ERROR

        # Check for markdown
        markdown_patterns = [r"^#+\s+", r"\*\*\w+\*\*", r"`\w+`", r"^\s*[-*+]\s+"]
        if any(
            re.search(pattern, content, re.MULTILINE) for pattern in markdown_patterns
        ):
            return OutputType.MARKDOWN

        return OutputType.PLAIN

    def format_content(
        self,
        content: str,
        output_type: Optional[OutputType] = None,
        width: Optional[int] = None,
    ) -> Union[Text, Syntax, Markdown]:
        """Format content based on its type."""
        if output_type is None:
            output_type = self.detect_output_type(content)

        # Get terminal width if not specified
        if width is None:
            width = shutil.get_terminal_size().columns - 4  # Account for panel borders

        if output_type == OutputType.JSON:
            return self._format_json(content, width)
        elif output_type == OutputType.CODE:
            return self._format_code(content, width)
        elif output_type == OutputType.PYTHON:
            return Syntax(
                content, "python", theme="monokai", line_numbers=True, word_wrap=True
            )
        elif output_type == OutputType.JAVASCRIPT:
            return Syntax(
                content,
                "javascript",
                theme="monokai",
                line_numbers=True,
                word_wrap=True,
            )
        elif output_type == OutputType.SQL:
            return Syntax(
                content, "sql", theme="monokai", line_numbers=True, word_wrap=True
            )
        elif output_type == OutputType.YAML:
            return Syntax(
                content, "yaml", theme="monokai", line_numbers=True, word_wrap=True
            )
        elif output_type == OutputType.XML:
            return Syntax(
                content, "xml", theme="monokai", line_numbers=True, word_wrap=True
            )
        elif output_type == OutputType.SHELL:
            return Syntax(
                content, "bash", theme="monokai", line_numbers=False, word_wrap=True
            )
        elif output_type == OutputType.DIFF:
            return self._format_diff(content, width)
        elif output_type == OutputType.LOG:
            return self._format_log(content, width)
        elif output_type == OutputType.ERROR:
            return self._format_error(content, width)
        elif output_type == OutputType.MARKDOWN:
            return Markdown(content)
        else:
            # Plain text with basic highlighting
            return self._format_plain(content, width)

    def _format_json(self, content: str, width: int) -> Union[Syntax, Text]:
        """Format JSON content with syntax highlighting."""
        try:
            import json

            # Try to pretty-print JSON
            parsed = json.loads(content)
            formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
            return Syntax(
                formatted, "json", theme="monokai", line_numbers=True, word_wrap=True
            )
        except (json.JSONDecodeError, ImportError):
            # Fallback to plain text if JSON is invalid
            return Text(content)

    def _format_code(self, content: str, width: int) -> Union[Syntax, Text]:
        """Format code content with language detection."""
        # Try to detect specific language
        for lang, patterns in self.language_patterns.items():
            if lang in ["python", "javascript"] and any(
                re.search(pattern, content, re.MULTILINE) for pattern in patterns
            ):
                return Syntax(
                    content, lang, theme="monokai", line_numbers=True, word_wrap=True
                )

        # Fallback to generic code highlighting
        return Syntax(
            content, "text", theme="monokai", line_numbers=True, word_wrap=True
        )

    def _format_diff(self, content: str, width: int) -> Text:
        """Format diff output with colors."""
        text = Text()

        for line in content.splitlines():
            if line.startswith("+"):
                text.append(line + "\n", style="green")
            elif line.startswith("-"):
                text.append(line + "\n", style="red")
            elif line.startswith("@@"):
                text.append(line + "\n", style="cyan")
            elif line.startswith("diff"):
                text.append(line + "\n", style="bold blue")
            else:
                text.append(line + "\n", style="white")

        return text

    def _format_log(self, content: str, width: int) -> Text:
        """Format log output with level-based colors."""
        text = Text()

        for line in content.splitlines():
            line_lower = line.lower()
            if "error" in line_lower or "fatal" in line_lower:
                text.append(line + "\n", style="bold red")
            elif "warning" in line_lower or "warn" in line_lower:
                text.append(line + "\n", style="yellow")
            elif "info" in line_lower:
                text.append(line + "\n", style="cyan")
            elif "debug" in line_lower:
                text.append(line + "\n", style="dim")
            else:
                text.append(line + "\n", style="white")

        return text

    def _format_error(self, content: str, width: int) -> Text:
        """Format error output with highlighting."""
        text = Text()

        for line in content.splitlines():
            # Highlight file paths and line numbers
            if re.search(r'File\s+"[^"]+",\s+line\s+\d+', line):
                text.append(line + "\n", style="bold cyan")
            # Highlight exception names
            elif re.search(r"\w+Error:", line) or re.search(r"\w+Exception:", line):
                text.append(line + "\n", style="bold red")
            # Highlight stack trace arrows
            elif line.strip().startswith("^"):
                text.append(line + "\n", style="red")
            else:
                text.append(line + "\n", style="white")

        return text

    def _format_plain(self, content: str, width: int) -> Text:
        """Format plain text with basic highlighting."""
        # Apply basic highlighting for URLs, file paths, etc.
        text = Text(content)

        # Highlight URLs
        url_pattern = r"https?://[^\s]+"
        for match in re.finditer(url_pattern, content):
            start, end = match.span()
            text.stylize("blue underline", start, end)

        # Highlight file paths
        path_pattern = r"/[^\s]+"
        for match in re.finditer(path_pattern, content):
            start, end = match.span()
            if Path(match.group()).exists():
                text.stylize("green", start, end)

        return text

    def create_foldable_sections(
        self, content: str, max_section_lines: int = 50
    ) -> List[FoldableSection]:
        """Create foldable sections from content."""
        lines = content.splitlines()
        sections = []

        if len(lines) <= max_section_lines:
            # Content is short enough, return as single section
            output_type = self.detect_output_type(content)
            section = FoldableSection(
                title="Output", content=content, output_type=output_type, folded=False
            )
            return [section]

        # Split into logical sections
        current_section_lines = []
        current_title = "Output"
        section_count = 1

        for i, line in enumerate(lines):
            current_section_lines.append(line)

            # Check if we should start a new section
            if len(current_section_lines) >= max_section_lines:
                section_content = "\n".join(current_section_lines)
                output_type = self.detect_output_type(section_content)

                section = FoldableSection(
                    title=f"{current_title} (Part {section_count})",
                    content=section_content,
                    output_type=output_type,
                    folded=True,  # Auto-fold long sections
                )
                sections.append(section)

                # Reset for next section
                current_section_lines = []
                section_count += 1

        # Add remaining lines as final section
        if current_section_lines:
            section_content = "\n".join(current_section_lines)
            output_type = self.detect_output_type(section_content)

            section = FoldableSection(
                title=f"{current_title} (Part {section_count})",
                content=section_content,
                output_type=output_type,
                folded=len(current_section_lines) > 10,  # Fold if more than 10 lines
            )
            sections.append(section)

        return sections


class OutputFolder:
    """Manages folding and unfolding of output sections."""

    def __init__(self, formatter: Optional[AdvancedFormatter] = None):
        """Initialize output folder."""
        self.formatter = formatter or AdvancedFormatter()
        self.sections: List[FoldableSection] = []

    def add_content(
        self,
        content: str,
        title: str = "Output",
        output_type: Optional[OutputType] = None,
    ) -> List[FoldableSection]:
        """Add content and create foldable sections."""
        if output_type is None:
            output_type = self.formatter.detect_output_type(content)

        # Create sections based on content size and type
        if (
            output_type in [OutputType.LOG, OutputType.ERROR]
            or len(content.splitlines()) > 20
        ):
            sections = self.formatter.create_foldable_sections(content)
        else:
            # Small content, single section
            section = FoldableSection(
                title=title, content=content, output_type=output_type, folded=False
            )
            sections = [section]

        self.sections.extend(sections)
        return sections

    def render_sections(self, width: Optional[int] = None) -> List[Panel]:
        """Render all sections as panels."""
        panels = []

        for i, section in enumerate(self.sections):
            # Determine content to show
            if (
                section.folded
                and section.line_count
                and section.line_count > section.max_preview_lines
            ):
                content_to_show = section.get_preview()
                title = f"{section.title} ({section.line_count} lines, folded)"
            else:
                content_to_show = section.content
                title = section.title

            # Format the content
            formatted_content = self.formatter.format_content(
                content_to_show, section.output_type, width
            )

            # Create panel with appropriate styling
            panel_style = self._get_panel_style(section.output_type)

            panel = Panel(
                formatted_content,
                title=f"[{i+1}] {title}",
                border_style=panel_style,
                expand=False,
                width=width,
            )

            panels.append(panel)

        return panels

    def _get_panel_style(self, output_type: OutputType) -> str:
        """Get panel border style based on output type."""
        style_map = {
            OutputType.ERROR: "red",
            OutputType.LOG: "yellow",
            OutputType.CODE: "green",
            OutputType.JSON: "cyan",
            OutputType.DIFF: "magenta",
            OutputType.MARKDOWN: "blue",
            OutputType.PLAIN: "white",
        }
        return style_map.get(output_type, "white")

    def toggle_section(self, index: int) -> bool:
        """Toggle folding state of a section."""
        if 0 <= index < len(self.sections):
            self.sections[index].toggle_fold()
            return True
        return False

    def fold_all(self):
        """Fold all sections."""
        for section in self.sections:
            section.folded = True

    def unfold_all(self):
        """Unfold all sections."""
        for section in self.sections:
            section.folded = False

    def clear(self):
        """Clear all sections."""
        self.sections.clear()

    def get_section_summary(self) -> Table:
        """Get a summary table of all sections."""
        table = Table(show_header=True, header_style="bold")
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="white")
        table.add_column("Type", style="yellow")
        table.add_column("Lines", justify="right", style="green")
        table.add_column("State", justify="center", style="blue")

        for i, section in enumerate(self.sections):
            state = "📁 Folded" if section.folded else "📖 Expanded"
            table.add_row(
                str(i + 1),
                section.title,
                section.output_type.value,
                str(section.line_count or 0),
                state,
            )

        return table


class ProgressTracker:
    """Enhanced progress tracking for long-running operations."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize progress tracker."""
        self.console = console or Console()
        self.progress = None
        self.tasks: Dict[str, Any] = {}

    def start_progress(self, title: str = "Processing...") -> Progress:
        """Start a progress display."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            transient=True,
        )
        self.progress.start()
        return self.progress

    def add_task(self, description: str, total: Optional[int] = None) -> str:
        """Add a new task to track."""
        if self.progress is None:
            self.start_progress()

        task_id = self.progress.add_task(description, total=total)
        task_name = f"task_{len(self.tasks)}"
        self.tasks[task_name] = task_id
        return task_name

    def update_task(
        self, task_name: str, advance: int = 1, description: Optional[str] = None
    ):
        """Update task progress."""
        if self.progress and task_name in self.tasks:
            task_id = self.tasks[task_name]
            self.progress.update(task_id, advance=advance, description=description)

    def complete_task(self, task_name: str):
        """Mark task as completed."""
        if self.progress and task_name in self.tasks:
            task_id = self.tasks[task_name]
            self.progress.update(task_id, completed=True)

    def stop_progress(self):
        """Stop the progress display."""
        if self.progress:
            self.progress.stop()
            self.progress = None
            self.tasks.clear()
