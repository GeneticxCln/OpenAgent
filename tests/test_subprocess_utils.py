import pytest
from openagent.utils.subprocess_utils import run_exec, SubprocessError


@pytest.mark.asyncio
async def test_run_exec_success():
    res = await run_exec(["echo", "hello"])
    assert res["success"]
    assert "hello" in res["stdout"]


@pytest.mark.asyncio
async def test_run_exec_timeout():
    with pytest.raises(SubprocessError):
        await run_exec(["sleep", "2"], timeout=0.1)
