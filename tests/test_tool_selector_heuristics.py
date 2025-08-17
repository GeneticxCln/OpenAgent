import pytest

from openagent.core.tool_selector import SmartToolSelector, ToolPlan
from openagent.tools.git import RepoGrep
from openagent.tools.system import CommandExecutor, FileManager, SystemInfo

pytestmark = pytest.mark.asyncio


async def test_heuristic_creates_multi_step_plan_list_and_disk(monkeypatch):
    """Heuristic should create multi-step plan for listing files and checking disk usage."""
    tools = {
        "command_executor": CommandExecutor(default_explain_only=True),
        "file_manager": FileManager(),
        "system_info": SystemInfo(),
        "repo_grep": RepoGrep(),
    }
    selector = SmartToolSelector(llm=None, available_tools=tools)

    # Force intent UNKNOWN so fallback heuristic path is used
    from openagent.core.tool_selector import ToolIntent

    async def _fake_analyze(*args, **kwargs):
        return ToolIntent.UNKNOWN

    monkeypatch.setattr(SmartToolSelector, "analyze_intent", _fake_analyze)
    plan: ToolPlan = await selector.create_tool_plan("show files and free space")

    # Should create at least two calls based on heuristics
    assert len(plan.calls) >= 2

    # Verify we have a file listing step and a disk usage command step, ordered
    names = [c.tool_name for c in plan.calls]
    assert "file_manager" in names
    assert "command_executor" in names

    # Ensure the disk usage step uses df -h in explain-only mode
    ce_calls = [c for c in plan.calls if c.tool_name == "command_executor"]
    assert any(c.parameters.get("command") == "df -h" for c in ce_calls)

    # Ensure ordering is strictly increasing
    orders = [c.order for c in plan.calls]
    assert orders == sorted(orders)


async def test_heuristic_adds_search_and_system_info(monkeypatch):
    """Heuristic should add search and system overview when requested."""
    tools = {
        "command_executor": CommandExecutor(default_explain_only=True),
        "file_manager": FileManager(),
        "system_info": SystemInfo(),
        "repo_grep": RepoGrep(),
    }
    selector = SmartToolSelector(llm=None, available_tools=tools)

    # Force intent UNKNOWN so we don't shortcut to single-step code_search or system_info
    from openagent.core.tool_selector import ToolIntent

    async def _fake_analyze(*args, **kwargs):
        return ToolIntent.UNKNOWN

    monkeypatch.setattr(SmartToolSelector, "analyze_intent", _fake_analyze)
    plan: ToolPlan = await selector.create_tool_plan(
        'grep for "openagent" and show overview'
    )

    names = [c.tool_name for c in plan.calls]
    assert "repo_grep" in names
    assert "system_info" in names

    # Check that the search pattern was extracted
    grep_calls = [c for c in plan.calls if c.tool_name == "repo_grep"]
    assert any(c.parameters.get("pattern") for c in grep_calls)
