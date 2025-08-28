from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple


@dataclass
class ToolCall:
    name: str
    args: Any


@dataclass
class ToolPlan:
    calls: List[ToolCall]


def validate_tool_plan(
    plan: ToolPlan, valid_tool_names: Iterable[str]
) -> Tuple[bool, Optional[str]]:
    names = set(valid_tool_names)
    for call in plan.calls:
        if call.name not in names:
            return False, f"Unknown tool: {call.name}"
        # Basic args validation: allow dict, str, or None
        if not isinstance(call.args, (dict, str, type(None))):
            return False, f"Invalid args type for {call.name}"
    return True, None
