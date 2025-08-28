from pathlib import Path

import pytest

from openagent.core.policy import CommandPolicy, configure_policy, get_policy_engine
from openagent.tools.system import FileManager


@pytest.mark.asyncio
async def test_file_write_denied_outside_safe(tmp_path):
    policy = CommandPolicy()
    policy.safe_paths = [str(tmp_path)]
    policy.restricted_paths = ["/etc", "/boot"]
    configure_policy(policy)

    fm = FileManager()
    outside = Path("/etc/should_not_write")
    res = await fm.execute({"operation": "write", "path": str(outside), "content": "x"})
    assert not res.success
    assert "not in safe paths" in (res.error or "").lower()


@pytest.mark.asyncio
async def test_file_move_requires_approval(tmp_path):
    policy = CommandPolicy()
    policy.safe_paths = [str(tmp_path)]
    configure_policy(policy)

    src = tmp_path / "a.txt"
    dst = tmp_path / "b.txt"
    src.write_text("x")

    fm = FileManager()
    res = await fm.execute(
        {"operation": "move", "path": str(src), "destination": str(dst)}
    )
    assert not res.success
    assert "requires approval" in (res.error or "").lower()

    res2 = await fm.execute(
        {
            "operation": "move",
            "path": str(src),
            "destination": str(dst),
            "confirm": True,
        }
    )
    assert isinstance(res2.success, bool)
