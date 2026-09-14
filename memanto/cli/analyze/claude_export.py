"""
Export a Claude (Anthropic) data export (``conversations.json``) to a
provider-style migration JSON for ``memanto migrate claude``.

Companion to ``chatgpt_export``. Like ChatGPT, Claude has no public "export my
memory" API — the user requests their data from
https://claude.ai/settings/data-privacy ("Export data") and Anthropic emails a
zip. So this adapter is also pure local file I/O: no key, no network, fully
offline and reproducible.

Claude export shape (as of 2026). The format has drifted across versions, so
this parser is deliberately *tolerant* of the variations seen in the wild:

    [
      {
        "uuid": "<id>",
        "name": "Trip to Chengdu",
        "created_at": "2025-01-01T00:00:00.000000Z",
        "updated_at": "2025-01-02T00:00:00.000000Z",
        "chat_messages": [
          {
            "uuid": "<id>",
            "sender": "human" | "assistant",
            "text": "...",
            "created_at": "...",
            // newer exports: "content": [{"type": "text", "text": "..."}]
          },
          ...
        ],
      },
      ...
    ]

Some exports nest messages under ``messages`` instead of ``chat_messages``,
use ``role`` instead of ``sender``, or put text in ``content`` blocks or
``parts`` instead of a flat ``text`` string. All of those are accepted.

Each emitted row is one memory candidate:

    {
      "id": "claude:<conversation_uuid>:<message_uuid>",
      "memory": "<message text>",
      "role": "user" | "assistant",
      "created_at": "<iso8601 string>",
      "conversation_id": "<uuid>",
      "conversation_title": "<name>",
    }
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCE_NAME = "claude"
DEFAULT_EXPORT_FILENAME = "claude_export.json"

_SKIP_ROLES = {"system", "tool"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_conversations(source: Path) -> list[dict[str, Any]]:
    path = source
    if path.is_dir():
        candidate = path / "conversations.json"
        if candidate.exists():
            path = candidate
        else:
            matches = sorted(path.rglob("conversations.json"))
            if not matches:
                raise ValueError(
                    f"No conversations.json found under {source}. "
                    "Unzip your Claude data export first, or point at the file directly."
                )
            path = matches[0]
    if not path.exists():
        raise ValueError(f"Claude export not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}")
    if not isinstance(data, list):
        raise ValueError(
            f"{path} should be a JSON list of conversations; got {type(data).__name__}."
        )
    return [c for c in data if isinstance(c, dict)]


def _messages_of(conv: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("chat_messages", "messages", "conversation"):
        batch = conv.get(key)
        if isinstance(batch, list):
            return [m for m in batch if isinstance(m, dict)]
    return []


def _role_of(message: dict[str, Any]) -> str:
    raw = message.get("sender") or message.get("role") or ""
    raw = str(raw).lower()
    if raw in ("human", "user"):
        return "user"
    if raw in ("assistant", "claude", "ai"):
        return "assistant"
    return raw


def _text_of(message: dict[str, Any]) -> str:
    # 1) flat text field
    text = message.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    # 2) content blocks: [{"type": "text", "text": "..."}] or ["..."]
    content = message.get("content")
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                btext = block.get("text")
                if isinstance(btext, str) and btext.strip():
                    parts.append(btext.strip())
            elif isinstance(block, str) and block.strip():
                parts.append(block.strip())
    # 3) ChatGPT-style parts
    for part in message.get("parts") or []:
        if isinstance(part, str) and part.strip():
            parts.append(part.strip())
    return "\n\n".join(parts).strip()


def _iter_memory_rows(
    conv: dict[str, Any], *, include_assistant: bool
) -> list[dict[str, Any]]:
    conv_id = str(conv.get("uuid") or conv.get("id") or "")
    title = (conv.get("name") or conv.get("title") or "").strip() or "Untitled conversation"
    rows: list[dict[str, Any]] = []

    for message in _messages_of(conv):
        role = _role_of(message)
        if role in _SKIP_ROLES:
            continue
        if role == "assistant" and not include_assistant:
            continue
        text = _text_of(message)
        if not text:
            continue
        msg_id = str(message.get("uuid") or message.get("id") or "")
        rows.append(
            {
                "id": f"claude:{conv_id}:{msg_id}",
                "memory": text,
                "role": role,
                "created_at": message.get("created_at"),
                "conversation_id": conv_id,
                "conversation_title": title,
            }
        )
    return rows


def run_claude_export(
    source: str | Path,
    run_dir: Path,
    *,
    include_assistant: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Transform a Claude ``conversations.json`` into a migration export.

    Same ``(export_path, export_dict)`` contract as the other exporters so the
    migration runner consumes it unchanged.
    """
    progress = on_progress or (lambda _msg: None)
    source_path = Path(source).expanduser()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    progress(f"Reading Claude export from {source_path}")
    conversations = _load_conversations(source_path)

    memories: list[dict[str, Any]] = []
    conv_summaries: list[dict[str, Any]] = []
    for conv in conversations:
        rows = _iter_memory_rows(conv, include_assistant=include_assistant)
        memories.extend(rows)
        conv_summaries.append(
            {
                "id": str(conv.get("uuid") or conv.get("id") or ""),
                "title": (conv.get("name") or conv.get("title") or "").strip()
                or "Untitled conversation",
                "create_time": conv.get("created_at"),
                "update_time": conv.get("updated_at"),
                "message_count": len(rows),
            }
        )

    export: dict[str, Any] = {
        "source": SOURCE_NAME,
        "exported_at": _now_iso(),
        "include_assistant": include_assistant,
        "conversation_count": len(conversations),
        "memory_count": len(memories),
        "conversations": conv_summaries,
        "memories": memories,
    }

    export_path = run_dir / DEFAULT_EXPORT_FILENAME
    export_path.write_text(
        json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    progress(
        f"Wrote {len(memories)} memory rows from "
        f"{len(conversations)} conversations -> {export_path}"
    )
    return export_path, export
