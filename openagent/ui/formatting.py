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
            "typescript": [
                r"interface\s+\w+",
                r"type\s+\w+\s*=",
                r"export\s+(interface|type|class|const|function)",
                r"import\s+\{?\w+",
            ],
            "go": [
                r"package\s+\w+",
                r"func\s+\w+\(",
                r"import\s+\(",
            ],
            "rust": [
                r"fn\s+\w+\(",
                r"let\s+mut\s+\w+",
                r"use\s+\w+::",
                r"impl\s+\w+",
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

        # Check for markdown (headers, bold/italic, code fences, tables, lists)
        markdown_patterns = [
            r"^#+\s+",  # headers
            r"\*\*.+?\*\*",  # bold
            r"`{1,3}.+?`{1,3}",  # inline or fenced markers
            r"^\s*[-*+]\s+",  # lists
            r"^\|.+\|\s*$",  # table row
            r"^\s{0,3}```",  # fenced block start
        ]
        if any(
            re.search(pattern, content, re.MULTILINE) for pattern in markdown_patterns
        ):
            return OutputType.MARKDOWN

        return OutputType.PLAIN

    def format_content(
        self,
        content: str,
        output_type: Optional[Union[OutputType, str]] = None,
        width: Optional[int] = None,
    ) -> Union[Text, Syntax, Markdown]:
        """Format content based on its type."""
        # Accept string output_type for compatibility
        if isinstance(output_type, str):
            try:
                output_type = OutputType(output_type.lower())
            except Exception:
                output_type = None

        if output_type is None:
            output_type = self.detect_output_type(content)

        # Get terminal width if not specified
        if width is None:
            width = shutil.get_terminal_size().columns - 4  # Account for panel borders

        if output_type == OutputType.JSON:
            return self._format_json(content, width)
        elif output_type == OutputType.CODE:
            return self._format_code(content, width)
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
            return Markdown(content, code_theme="monokai")
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
        language_map = {
            "python": "python",
            "javascript": "javascript",
            "typescript": "typescript",
            "go": "go",
            "rust": "rust",
        }
        for lang, patterns in self.language_patterns.items():
            if lang in language_map and any(
                re.search(pattern, content, re.MULTILINE) for pattern in patterns
            ):
                return Syntax(
                    content,
                    language_map[lang],
                    theme="monokai",
                    line_numbers=True,
                    word_wrap=True,
                )

        # Fallback to generic code highlighting
        return Syntax(
            content, "text", theme="monokai", line_numbers=True, word_wrap=True
        )

    def _format_diff(self, content: str, width: int) -> Union[Syntax, Text]:
        """Format diff output with colors or Syntax diff lexer."""
        # Prefer Syntax 'diff' lexer for better highlighting if content resembles a diff
        try:
            if re.search(r"^diff\s|^@@|^\+|^\-", content, re.MULTILINE):
                return Syntax(
                    content, "diff", theme="monokai", line_numbers=False, word_wrap=True
                )
        except Exception:
            pass
        # Fallback manual coloring
        text = Text()
        for line in content.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                text.append(line + "\n", style="bold blue")
            elif line.startswith("+"):
                text.append(line + "\n", style="green")
            elif line.startswith("-"):
                text.append(line + "\n", style="red")
            elif line.startswith("@@"):
                text.append(line + "\n", style="cyan")
            elif line.startswith("index "):
                text.append(line + "\n", style="magenta")
            else:
                text.append(line + "\n")
        return text

    def _format_log(self, content: str, width: int) -> Text:
        """Format log output with timestamp and level highlighting, including common formats (JSON, Nginx/Apache)."""
        text = Text()
        ts_pattern = re.compile(
            r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
        )
        level_styles = {
            "CRITICAL": "bold red",
            "FATAL": "bold red",
            "ERROR": "red",
            "WARNING": "yellow",
            "WARN": "yellow",
            "INFO": "cyan",
            "DEBUG": "dim",
            "TRACE": "dim",
        }
        level_pattern = re.compile(
            r"\b(CRITICAL|FATAL|ERROR|WARNING|WARN|INFO|DEBUG|TRACE)\b"
        )

        nginx_apache = re.compile(
            r"^(\S+)\s+(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+\"(\w+)\s+([^\"]+)\s+HTTP\/[^\"]+\"\s+(\d{3})\s+(\S+)"
        )
        for line in content.splitlines():
            # Structured JSON logs (single-line JSON objects)
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    import json

                    obj = json.loads(stripped)
                    # Build a concise formatted line
                    lvl = str(
                        obj.get("level")
                        or obj.get("lvl")
                        or obj.get("severity")
                        or "INFO"
                    ).upper()
                    msg = str(
                        obj.get("msg") or obj.get("message") or obj.get("event") or ""
                    )
                    method = obj.get("method")
                    path = obj.get("path") or obj.get("url")
                    status = obj.get("status") or obj.get("status_code")
                    base = Text(f"{lvl}: ")
                    base.stylize(level_styles.get(lvl, ""), 0, len(lvl) + 2)
                    base.append(msg)
                    if method and path:
                        base.append(" ")
                        base.append(f"{method} {path}", style="cyan")
                    if status is not None:
                        s = int(status)
                        sstyle = (
                            "green"
                            if 200 <= s < 400
                            else ("yellow" if 400 <= s < 500 else "red")
                        )
                        base.append(" ")
                        base.append(str(s), style=sstyle)
                    text.append(base)
                    text.append("\n")
                    continue
                except Exception:
                    pass
            # Nginx/Apache logs
            m = nginx_apache.match(line)
            if m:
                method = m.group(5)
                path = m.group(6)
                status = int(m.group(7))
                sstyle = (
                    "green"
                    if 200 <= status < 400
                    else ("yellow" if 400 <= status < 500 else "red")
                )
                ln = Text(line)
                # Method + path highlight
                idx = line.find(f'"{method} ')
                if idx != -1:
                    ln.stylize("cyan", idx + 1, idx + 1 + len(method) + 1 + len(path))
                # Status highlight
                s_idx = line.rfind(str(status))
                if s_idx != -1:
                    ln.stylize(sstyle, s_idx, s_idx + len(str(status)))
                text.append(ln)
                text.append("\n")
                continue
            # Generic highlighting
            line_text = Text(line)
            # Highlight timestamp(s)
            for m in ts_pattern.finditer(line):
                start, end = m.span()
                line_text.stylize("dim", start, end)
            # Highlight level token(s)
            for m in level_pattern.finditer(line):
                start, end = m.span()
                style = level_styles.get(m.group(1), "")
                if style:
                    line_text.stylize(style, start, end)
            text.append(line_text)
            text.append("\n")
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
        """Create intelligent foldable sections from content."""
        lines = content.splitlines()

        # If content is short, return as single section
        if len(lines) <= 10:
            output_type = self.detect_output_type(content)
            section = FoldableSection(
                title="Output", content=content, output_type=output_type, folded=False
            )
            return [section]

        # Try to detect logical sections based on content patterns
        sections = self._detect_logical_sections(lines)

        if not sections:
            # Fallback to size-based sectioning
            sections = self._create_size_based_sections(lines, max_section_lines)

        return sections

    def _detect_logical_sections(self, lines: List[str]) -> List[FoldableSection]:
        """Detect logical sections in output based on patterns."""
        sections = []
        current_section_lines = []
        current_title = "Output"

        i = 0
        while i < len(lines):
            line = lines[i]

            # Detect different section types
            section_info = self._detect_section_start(
                line, lines[i : i + 5] if i + 5 <= len(lines) else lines[i:]
            )

            if section_info:
                # Save current section if it has content
                if current_section_lines:
                    section_content = "\n".join(current_section_lines)
                    output_type = self.detect_output_type(section_content)
                    section = FoldableSection(
                        title=current_title,
                        content=section_content,
                        output_type=output_type,
                        folded=len(current_section_lines) > 15,
                    )
                    sections.append(section)

                # Start new section
                current_section_lines = []
                current_title = section_info["title"]

                # Collect lines for this section
                section_end = self._find_section_end(lines[i:], section_info["type"])
                section_lines = lines[i : i + section_end]
                current_section_lines.extend(section_lines)
                i += section_end
            else:
                current_section_lines.append(line)
                i += 1

        # Add final section
        if current_section_lines:
            section_content = "\n".join(current_section_lines)
            output_type = self.detect_output_type(section_content)
            section = FoldableSection(
                title=current_title,
                content=section_content,
                output_type=output_type,
                folded=len(current_section_lines) > 15,
            )
            sections.append(section)

        return sections

    def _detect_section_start(
        self, line: str, context_lines: List[str]
    ) -> Optional[Dict[str, str]]:
        """Detect if a line starts a new logical section."""
        line_stripped = line.strip()

        # Stack trace section
        if re.match(r"Traceback \(most recent call last\):", line):
            return {"title": "🚨 Stack Trace", "type": "traceback"}

        # Error sections
        if re.match(r"\w+Error:|\w+Exception:", line):
            return {"title": "❌ Error Details", "type": "error"}

        # Compilation/Build output
        if any(
            keyword in line.lower() for keyword in ["compiling", "building", "linking"]
        ):
            return {"title": "🔨 Build Output", "type": "build"}

        # Test output (pytest/unittest)
        if (
            re.match(r".*test.*\.\.\.", line, re.IGNORECASE)
            or re.search(r"=+\s*test session starts\s*=+", line, re.IGNORECASE)
            or "PASSED" in line
            or "FAILED" in line
            or "ERROR" in line
        ):
            return {"title": "🧪 Test Results", "type": "test"}
        # Individual test case lines (pytest nodeid or dot/letter status lines)
        if re.match(
            r"^tests?\/.*::.*\s+(PASSED|FAILED|ERROR|SKIPPED)", line
        ) or re.match(r"^tests?\/.*\.py\s+[.FEskx]+$", line):
            return {"title": "🧪 Test Case", "type": "testcase"}

        # Package installation
        if any(
            keyword in line.lower()
            for keyword in ["installing", "downloading", "collecting"]
        ):
            return {"title": "📦 Package Installation", "type": "package"}

        # Git output
        if line.startswith(("commit", "Author:", "Date:", "diff --git")):
            return {"title": "📝 Git Output", "type": "git"}

        # Docker output
        if (
            line.startswith(("STEP", "FROM", "RUN", "COPY", "ADD"))
            or "docker" in line.lower()
        ):
            return {"title": "🐳 Docker Output", "type": "docker"}

        # Network/HTTP requests
        if re.match(r"\d{3} (GET|POST|PUT|DELETE)", line) or "HTTP/" in line:
            return {"title": "🌐 Network Activity", "type": "network"}

        # Log entries with timestamps
        if re.match(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", line):
            return {"title": "📋 Application Logs", "type": "logs"}

        return None

    def _find_section_end(self, lines: List[str], section_type: str) -> int:
        """Find where a logical section ends."""
        if section_type == "traceback":
            # Stack trace ends at the exception line
            for i, line in enumerate(lines[1:], 1):
                if re.match(r"\w+Error:|\w+Exception:", line):
                    return i + 1
            return min(20, len(lines))  # Max 20 lines for traceback

        elif section_type == "error":
            # Error details are usually just a few lines
            return min(5, len(lines))

        elif section_type == "test":
            # Test sections end when no more test-related lines
            for i, line in enumerate(lines[1:], 1):
                if not any(
                    keyword in line.lower()
                    for keyword in [
                        "test",
                        "passed",
                        "failed",
                        "ok",
                        "error",
                        "collected",
                        "session",
                    ]
                ):
                    if i > 3:  # Ensure we get at least a few lines
                        return i
            return min(30, len(lines))

        elif section_type == "testcase":
            # Single test case lines are short; include up to 3 lines around
            return min(3, len(lines))

        elif section_type in ["build", "package", "docker"]:
            # These sections can be quite long
            return min(50, len(lines))

        elif section_type == "git":
            # Git commits and diffs can be long; stop at next commit/diff marker
            for i, line in enumerate(lines[1:], 1):
                if line.startswith("commit") or line.startswith("diff --git"):
                    return i
            return min(80, len(lines))

        elif section_type == "logs":
            # Log sections - group consecutive log entries by timestamp pattern
            for i, line in enumerate(lines[1:], 1):
                if not re.match(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", line):
                    if i > 5:  # At least 5 log entries
                        return i
            return min(50, len(lines))

        # Default: section split on blank-line runs or small cap
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "" and (
                i + 1 < len(lines) and lines[i + 1].strip() == ""
            ):
                return i + 1
        return min(10, len(lines))

    def _create_size_based_sections(
        self, lines: List[str], max_section_lines: int
    ) -> List[FoldableSection]:
        """Create sections based purely on size when logical detection fails."""
        sections = []
        current_section_lines = []
        section_count = 1

        for line in lines:
            current_section_lines.append(line)

            if len(current_section_lines) >= max_section_lines:
                section_content = "\n".join(current_section_lines)
                output_type = self.detect_output_type(section_content)

                section = FoldableSection(
                    title=f"📄 Output (Part {section_count})",
                    content=section_content,
                    output_type=output_type,
                    folded=True,  # Auto-fold large sections
                )
                sections.append(section)

                current_section_lines = []
                section_count += 1

        # Add remaining lines
        if current_section_lines:
            section_content = "\n".join(current_section_lines)
            output_type = self.detect_output_type(section_content)

            section = FoldableSection(
                title=(
                    f"📄 Output (Part {section_count})"
                    if section_count > 1
                    else "📄 Output"
                ),
                content=section_content,
                output_type=output_type,
                folded=len(current_section_lines) > 15,
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
