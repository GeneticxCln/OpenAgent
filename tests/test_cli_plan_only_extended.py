import json
import pytest
from typer.testing import CliRunner

from openagent.cli import app

runner = CliRunner()


def _plan(task: str):
    res = runner.invoke(app, ["do", task, "--plan-only", "--auto-execute"], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    return json.loads(res.stdout.strip())


def _names(plan_json):
    return [c["tool_name"] for c in plan_json.get("calls", [])]


def test_cli_plan_only_network_diagnostics():
    data = _plan("diagnose network connectivity issues and check ports")
    names = _names(data)
    assert names.count("command_executor") >= 2
    # Check for ss and ping planned
    ce_cmds = [c.get("parameters", {}).get("command", "") for c in data.get("calls", []) if c["tool_name"] == "command_executor"]
    assert any("ss -tulpn" in cmd for cmd in ce_cmds)
    assert any("ping -c 3 8.8.8.8" in cmd for cmd in ce_cmds)


def test_cli_plan_only_package_overview():
    data = _plan("show python packages and dependencies")
    names = _names(data)
    assert "command_executor" in names
    ce = [c for c in data.get("calls", []) if c["tool_name"] == "command_executor"]
    assert any("pip list" in c.get("parameters", {}).get("command", "") for c in ce)


def test_cli_plan_only_git_troubleshoot():
    data = _plan("git has a conflict issue, what's wrong")
    names = _names(data)
    # Should include git status and git log
    assert names.count("git_tool") >= 2


def test_cli_plan_only_error_recovery():
    data = _plan("the build failed with an error timeout, help")
    names = _names(data)
    # Should include a repo_grep step
    assert "repo_grep" in names

