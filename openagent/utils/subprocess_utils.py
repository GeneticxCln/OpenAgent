from __future__ import annotations

import asyncio
from typing import Dict, List


class SubprocessError(Exception):
    pass


async def run_exec(argv: List[str], timeout: float = 30.0) -> Dict[str, object]:
    """
    Safe exec-style subprocess wrapper:
    - No shell=True
    - Controlled timeout
    - Captures stdout/stderr
    """
    if not argv or not isinstance(argv, list) or not isinstance(argv[0], str):
        raise ValueError("argv must be a non-empty list of strings")

    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError as e:
        proc.terminate()
        await proc.wait()
        raise SubprocessError(f"process timeout after {timeout}s") from e

    out = stdout.decode("utf-8", errors="replace")
    err = stderr.decode("utf-8", errors="replace")
    return {
        "success": proc.returncode == 0,
        "stdout": out,
        "stderr": err,
        "exit_code": proc.returncode,
    }
