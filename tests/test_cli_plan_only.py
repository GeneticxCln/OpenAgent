import json

import pytest
from typer.testing import CliRunner

from openagent.cli import app

runner = CliRunner()


def _invoke_do(task: str, monkeypatch):
    # Stub create_agent to avoid loading real models or background tasks
    class _Stub:
        llm = None

    monkeypatch.setattr("openagent.cli.create_agent", lambda *args, **kwargs: _Stub())

    # Force heuristic fallback in planner by making intent UNKNOWN
    from openagent.core.tool_selector import SmartToolSelector, ToolIntent

    async def _fake_analyze(*args, **kwargs):
        return ToolIntent.UNKNOWN

    monkeypatch.setattr(SmartToolSelector, "analyze_intent", _fake_analyze)

    result = runner.invoke(
        app, ["do", task, "--plan-only", "--auto-execute"], catch_exceptions=False
    )
    assert result.exit_code == 0, f"CLI exited with {result.exit_code}: {result.output}"
    # Output should be JSON
    data = json.loads(result.stdout.strip())
    assert isinstance(data, dict)
    assert "calls" in data
    return data


def test_cli_plan_only_lists_and_disk_usage(monkeypatch):
    data = _invoke_do("list files and check disk usage", monkeypatch)

    calls = data.get("calls", [])
    assert len(calls) >= 2

    names = [c["tool_name"] for c in calls]
    assert "file_manager" in names
    assert "command_executor" in names

    # Ensure df -h appears
    ce = [c for c in calls if c["tool_name"] == "command_executor"]
    assert any(c.get("parameters", {}).get("command") == "df -h" for c in ce)


def test_cli_plan_only_search_and_overview(monkeypatch):
    data = _invoke_do('search for "openagent" and show system info', monkeypatch)

    calls = data.get("calls", [])
    assert len(calls) >= 2

    names = [c["tool_name"] for c in calls]
    assert "repo_grep" in names
    assert "system_info" in names

    grep_calls = [c for c in calls if c["tool_name"] == "repo_grep"]
    assert any(c.get("parameters", {}).get("pattern") for c in grep_calls)
