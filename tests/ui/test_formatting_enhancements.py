import re
from typing import List

from rich.console import Console

from openagent.ui.formatting import AdvancedFormatter, OutputFolder, OutputType


def test_markdown_detection_and_rendering():
    console = Console(record=True, color_system=None, width=100)
    fmt = AdvancedFormatter(console)
    md = """
# Header

| col1 | col2 |
|------|------|
|  a   |  b   |

```python
print("hello")
```
"""
    t = fmt.detect_output_type(md)
    assert t == OutputType.MARKDOWN
    renderable = fmt.format_content(md, t)
    # Should be a Markdown renderable
    from rich.markdown import Markdown

    assert isinstance(renderable, Markdown)


def test_diff_detection_and_formatting():
    console = Console(record=True, color_system=None, width=100)
    fmt = AdvancedFormatter(console)
    diff = """
diff --git a/foo.txt b/foo.txt
index e69de29..4b825dc 100644
--- a/foo.txt
+++ b/foo.txt
@@ -0,0 +1,2 @@
+hello
+world
"""
    t = fmt.detect_output_type(diff)
    assert t == OutputType.DIFF
    renderable = fmt.format_content(diff, t)
    # Render to text to ensure no exceptions and content visible
    console.print(renderable)
    out = console.export_text()
    assert "hello" in out and "world" in out


def test_log_detection_and_highlighting():
    console = Console(record=True, color_system=None, width=100)
    fmt = AdvancedFormatter(console)
    log = """
2025-08-28T09:00:00Z INFO server Started
2025-08-28T09:00:01Z WARNING disk High usage
2025-08-28T09:00:02Z ERROR api Failure
"""
    t = fmt.detect_output_type(log)
    assert t == OutputType.LOG
    renderable = fmt.format_content(log, t)
    console.print(renderable)
    out = console.export_text()
    # Plain text should include log lines
    assert "Started" in out and "High usage" in out and "Failure" in out


def test_folding_heuristics_logical_sections():
    console = Console(record=True, color_system=None, width=100)
    fmt = AdvancedFormatter(console)
    content = """
Traceback (most recent call last):
  File "main.py", line 1, in <module>
ValueError: bad
==== test session starts ====
collected 2 items

test_a.py .

test_b.py F
2025-08-28 09:01:00 INFO run done
"""
    sections = fmt.create_foldable_sections(content)
    assert isinstance(sections, list)
    # Expect multiple sections for traceback, tests, and logs
    assert len(sections) >= 2
    titles = [s.title for s in sections]
    joined = " ".join(titles)
    assert (
        "Stack Trace" in joined
        or "Test Results" in joined
        or "Application Logs" in joined
    )
