"""
GitHub read-only integration tool for OpenAgent.

Provides operations to list PRs/issues and fetch PR details/diffs without
exposing tokens. Uses $GITHUB_TOKEN if present. All operations are read-only.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx

from openagent.core.base import BaseTool, ToolResult
from openagent.core.redact import redact_text


class GitHubTool(BaseTool):
    """
    Read-only GitHub integration.

    Input schema examples:
    - {"operation":"list_prs", "owner":"org", "repo":"name", "state":"open", "per_page":30}
    - {"operation":"get_pr", "owner":"org", "repo":"name", "number":123}
    - {"operation":"list_issues", "owner":"org", "repo":"name", "state":"open"}
    - {"operation":"get_pr_files", "owner":"org", "repo":"name", "number":123}
    - {"operation":"get_pr_diff", "owner":"org", "repo":"name", "number":123}
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="github_tool",
            description="Read-only GitHub queries: PRs, issues, PR details and diffs",
            **kwargs,
        )

    async def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        op = (input_data or {}).get("operation")
        if not op:
            return ToolResult(success=False, content="", error="Missing 'operation'")

        owner = input_data.get("owner")
        repo = input_data.get("repo")
        if not owner or not repo:
            # allow 'owner/repo' combined via 'repo_full'
            repo_full = input_data.get("repo_full")
            if repo_full and "/" in repo_full:
                owner, repo = repo_full.split("/", 1)
            else:
                return ToolResult(success=False, content="", error="Provide 'owner' and 'repo' or 'repo_full' as 'owner/repo'")

        base = f"https://api.github.com/repos/{owner}/{repo}"

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                if op == "list_prs":
                    params = {
                        "state": input_data.get("state", "open"),
                        "per_page": int(input_data.get("per_page", 30)),
                    }
                    resp = await self._request(client, "GET", f"{base}/pulls", params=params)
                    if not resp["ok"]:
                        return ToolResult(success=False, content="", error=resp["error"]) 
                    content = json.dumps(resp["json"], indent=2)
                    return ToolResult(success=True, content=redact_text(content), metadata={"endpoint": f"{base}/pulls"})

                if op == "get_pr":
                    number = input_data.get("number")
                    if not number:
                        return ToolResult(success=False, content="", error="Missing 'number'")
                    resp = await self._request(client, "GET", f"{base}/pulls/{number}")
                    if not resp["ok"]:
                        return ToolResult(success=False, content="", error=resp["error"]) 
                    content = json.dumps(resp["json"], indent=2)
                    return ToolResult(success=True, content=redact_text(content), metadata={"endpoint": f"{base}/pulls/{number}"})

                if op == "list_issues":
                    params = {
                        "state": input_data.get("state", "open"),
                        "per_page": int(input_data.get("per_page", 30)),
                    }
                    resp = await self._request(client, "GET", f"{base}/issues", params=params)
                    if not resp["ok"]:
                        return ToolResult(success=False, content="", error=resp["error"]) 
                    content = json.dumps(resp["json"], indent=2)
                    return ToolResult(success=True, content=redact_text(content), metadata={"endpoint": f"{base}/issues"})

                if op == "get_pr_files":
                    number = input_data.get("number")
                    if not number:
                        return ToolResult(success=False, content="", error="Missing 'number'")
                    resp = await self._request(client, "GET", f"{base}/pulls/{number}/files")
                    if not resp["ok"]:
                        return ToolResult(success=False, content="", error=resp["error"]) 
                    content = json.dumps(resp["json"], indent=2)
                    return ToolResult(success=True, content=redact_text(content), metadata={"endpoint": f"{base}/pulls/{number}/files"})

                if op == "get_pr_diff":
                    number = input_data.get("number")
                    if not number:
                        return ToolResult(success=False, content="", error="Missing 'number'")
                    resp = await self._request(client, "GET", f"{base}/pulls/{number}", accept="application/vnd.github.v3.diff")
                    if not resp["ok"]:
                        return ToolResult(success=False, content="", error=resp["error"]) 
                    # content is textual diff
                    return ToolResult(success=True, content=resp["text"], metadata={"endpoint": f"{base}/pulls/{number}", "format": "diff"})

                return ToolResult(success=False, content="", error=f"Unsupported operation: {op}")
        except Exception as e:
            return ToolResult(success=False, content="", error=f"GitHubTool failed: {e}")

    async def _request(self, client: httpx.AsyncClient, method: str, url: str, params: Optional[Dict[str, Any]] = None, accept: Optional[str] = None) -> Dict[str, Any]:
        headers = {"Accept": accept or "application/vnd.github+json"}
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
            headers["X-GitHub-Api-Version"] = "2022-11-28"
        r = await client.request(method, url, params=params, headers=headers)
        if r.status_code >= 400:
            # Provide a succinct error; do not include token
            msg = r.text
            # handle rate limit hint
            if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
                msg = "Rate limit exceeded. Provide a token or wait before retrying."
            return {"ok": False, "error": f"HTTP {r.status_code}: {msg[:200]}"}
        try:
            if accept == "application/vnd.github.v3.diff":
                return {"ok": True, "text": r.text}
            return {"ok": True, "json": r.json()}
        except Exception:
            return {"ok": True, "text": r.text}

