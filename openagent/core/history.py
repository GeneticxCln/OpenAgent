"""
History management for OpenAgent.

Provides persistent storage of interaction blocks (request, plan, tool results, response)
similar to Warp's command blocks. Stores JSONL files under ~/.openagent/history/.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

HISTORY_DIR = Path.home() / ".openagent" / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Block:
    id: str
    timestamp: str
    input: str
    plan: Optional[Dict[str, Any]] = None
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    response: str = ""
    model: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class HistoryManager:
    """Manage persistent history blocks in JSONL files per day."""

    def __init__(self, base_dir: Path = HISTORY_DIR) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _file_for_today(self) -> Path:
        fname = time.strftime("%Y%m%d.jsonl", time.localtime())
        return self.base_dir / fname

    def append(self, block: Block) -> None:
        f = self._file_for_today()
        with f.open("a", encoding="utf-8") as fp:
            fp.write(block.to_json() + "\n")

    def list_blocks(self, limit: int = 50) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("*.jsonl"), reverse=True):
            try:
                with path.open("r", encoding="utf-8") as fp:
                    # Read all lines from the file
                    file_items = []
                    for line in fp:
                        try:
                            obj = json.loads(line)
                            file_items.append(obj)
                        except Exception:
                            continue
                    # Add in reverse order (newest first)
                    items.extend(reversed(file_items))
            except Exception:
                continue
            if len(items) >= limit:
                break
        return items[:limit]

    def iter_all(self):
        for path in sorted(self.base_dir.glob("*.jsonl"), reverse=True):
            with path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue

    def get(self, block_id: str) -> Optional[Dict[str, Any]]:
        for obj in self.iter_all():
            if obj.get("id") == block_id:
                return obj
        return None

    def search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        q = query.lower()
        out: List[Dict[str, Any]] = []
        for obj in self.iter_all():
            hay = json.dumps(obj, ensure_ascii=False).lower()
            if q in hay:
                out.append(obj)
                if len(out) >= limit:
                    break
        return out

    def export(self, block_id: str, format: str = "json") -> str:
        """Export a history block in specified format.

        Args:
            block_id: ID of the block to export
            format: Export format ('json' or 'md')

        Returns:
            Exported content as a string
        """
        block_data = self.get(block_id)
        if not block_data:
            raise ValueError(f"Block {block_id} not found")

        if format == "json":
            return json.dumps(block_data, indent=2, ensure_ascii=False)
        elif format == "md":
            # Markdown export
            model_info = block_data.get("model") or {}
            md = f"""# Block {block_data['id']}

- Time: {block_data.get('timestamp', 'Unknown')}
- Model: {model_info.get('model_name', 'Unknown') if model_info else 'Unknown'}

## Input

```
{block_data.get('input', '')}
```

## Tool Results

"""
            for tr in block_data.get("tool_results", []):
                success_str = "Success" if tr.get("success") else "Failed"
                md += f"- {tr.get('tool', 'Unknown')}: {success_str}\n"
                if tr.get("content"):
                    md += (
                        f"  - Content: {tr['content'][:100]}...\n"
                        if len(tr.get("content", "")) > 100
                        else f"  - Content: {tr['content']}\n"
                    )

            md += f"""
## Response

{block_data.get('response', '')}
"""
            return md
        else:
            raise ValueError(f"Unsupported export format: {format}")

    @staticmethod
    def new_block(
        input_text: str,
        response: str,
        plan: Optional[Dict[str, Any]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        model: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Block:
        return Block(
            id=str(uuid.uuid4())[:8],
            timestamp=_now_iso(),
            input=input_text,
            plan=plan,
            tool_results=tool_results or [],
            response=response,
            model=model,
            context=context,
        )
