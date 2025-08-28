from rich.console import Console

from openagent.ui.formatting import AdvancedFormatter, OutputType


def test_language_detection_code_rendering():
    console = Console(record=True, color_system=None, width=100)
    fmt = AdvancedFormatter(console)

    samples = {
        "go": """package main\nimport (\n\"fmt\"\n)\nfunc main() { fmt.Println(\"hi\") }""",
        "typescript": """export interface Foo { bar: string }\nconst x: Foo = { bar: 'b' }""",
        "rust": """fn main() { let mut x = 1; println!(\"{}\", x); }""",
    }

    for lang, code in samples.items():
        t = fmt.detect_output_type(code)
        assert t == OutputType.CODE
        renderable = fmt.format_content(code, t)
        console.print(renderable)
        out = console.export_text()
        assert "hi" in out or "interface" in out or "println" in out


def test_structured_logs_json_and_nginx():
    console = Console(record=True, color_system=None, width=120)
    fmt = AdvancedFormatter(console)

    json_log = '{"time":"2025-08-28T09:00:00Z","level":"ERROR","msg":"failed","method":"GET","path":"/api","status":500}'
    nginx_log = (
        '127.0.0.1 - - [28/Aug/2025:09:00:01 +0000] "GET /healthz HTTP/1.1" 404 123'
    )

    t1 = fmt.detect_output_type(json_log)
    t2 = fmt.detect_output_type(nginx_log)
    assert t1 == OutputType.LOG and t2 == OutputType.LOG

    r1 = fmt.format_content(json_log, t1)
    r2 = fmt.format_content(nginx_log, t2)
    console.print(r1)
    console.print(r2)
    out = console.export_text()
    assert "ERROR" in out and "failed" in out
    assert "/healthz" in out and "404" in out


def test_per_test_case_folding_sections():
    console = Console(record=True, color_system=None, width=100)
    fmt = AdvancedFormatter(console)

    content = """
collected 2 items

tests/test_a.py::TestX::test_y PASSED

tests/test_b.py::test_z FAILED
"""
    sections = fmt.create_foldable_sections(content)
    titles = [s.title for s in sections]
    assert any("Test Case" in t for t in titles) or any(
        "Test Results" in t for t in titles
    )
