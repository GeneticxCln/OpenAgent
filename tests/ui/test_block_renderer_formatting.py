from rich.console import Console

from openagent.ui.blocks import BlockRenderer, BlockType, CommandBlock


def render_text(renderable) -> str:
    console = Console(record=True, color_system=None, width=100)
    console.print(renderable)
    return console.export_text()


def test_ai_markdown_rendering():
    console = Console(record=True, color_system=None, width=100)
    renderer = BlockRenderer(console)
    md_content = """# Title\n\n- item 1\n- item 2\n\n```python\nprint(\"hi\")\n```\n"""
    block = CommandBlock(block_type=BlockType.AI_RESPONSE, output=md_content)
    panel = renderer.render_block(block, width=80)
    text = render_text(panel)
    # Expect markdown content to be present in some form
    assert "Title" in text or "item 1" in text
    assert "print(" in text


def test_error_output_presence():
    console = Console(record=True, color_system=None, width=100)
    renderer = BlockRenderer(console)
    error_text = (
        "Error: build failed\nTraceback (most recent call last):\nException: boom"
    )
    block = CommandBlock(command="make build", error=error_text)
    panel = renderer.render_block(block, width=80)
    text = render_text(panel)
    # Ensure error content made it into the rendering
    assert "Error:" in text or "Exception" in text or "Traceback" in text


def test_output_truncation_behavior():
    console = Console(record=True, color_system=None, width=100)
    renderer = BlockRenderer(console)
    # Create long output (250 lines)
    long_output = "\n".join([f"line {i}" for i in range(250)])
    block = CommandBlock(command="echo long", output=long_output)
    panel = renderer.render_block(block, width=80)
    text = render_text(panel)
    # Truncation hint should be present
    assert "lines truncated; press 'o' to expand" in text
    # Head and tail context should be present
    assert "line 0" in text
    assert "line 249" in text
