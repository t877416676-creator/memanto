"""
Export a ChatGPT data export (``conversations.json``) to a provider-style
migration JSON for ``memanto migrate chatgpt``.

Unlike the API-backed exporters (``mem0``/``letta``/``supermemory``), there is
no ChatGPT API to call: the user downloads their history once from
https://chatgpt.com/#settings/DataControls ("Export data") and hands us the
resulting file. So this adapter is pure local file I/O — no key, no network,
fully offline and reproducible.

ChatGPT export shape (as of 2026):

    [
      {
        "title": "Refactor plan",
        "create_time": 1735689600.0,
        "update_time": 1735693200.0,
        "mapping": {
          "<node_id>": {
            "id": "<node_id>",
            "message": {
              "id": "<msg_id>",
              "author": {"role": "user" | "assistant" | "system" | "tool"},
              "create_time": 1735689605.0,
              "content": {"content_type": "text", "parts": ["..."]},
            },
            "parent": "<node_id> | null",
            "children": ["<node_id>", ...],
          },
          ...
        },
        "current_node": "<node_id>",
        "conversation_id": "<id>",
      },
      ...
    ]

``mapping`` is a small tree of message nodes, not a list. We walk the thread
that leads to ``current_node`` (the branch the user was last on) and, when a
node has no ``current_node``, fall back to every text-bearing message in
insertion order so nothing is silently dropped.

Each emitted row is one memory candidate:

    {
      "id": "chatgpt:<conversation_id>:<message_id>",
      "memory": "<message text>",
      "role": "user" | "assistant",
      "created_at": <epoch seconds>,
      "conversation_id": "<id>",
      "conversation_title": "<title>",
    }

Non-text parts (images, tool calls, code-interpreter blobs, hidden system
messages) are skipped — they carry no portable "memory" value and only bloat
the migration. Set ``include_assistant=False`` to migrate only what *you*
said (the closest thing to ChatGPT's saved "memory"/facts about you).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCE_NAME = "chatgpt"
DEFAULT_EXPORT_FILENAME = "chatgpt_export.json"

# Roles that never carry user memory worth migrating.
_SKIP_ROLES = {"system", "tool"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_conversations(source: Path) -> list[dict[str, Any]]:
    """Read ``conversations.json`` from a file or a directory containing one."""
    path = source
    if path.is_dir():
        candidate = path / "conversations.json"
        if candidate.exists():
            path = candidate
        else:
            # ChatGPT's zip unpacks to a folder that may nest the file one level.
            matches = sorted(path.rglob("conversations.json"))
            if not matches:
                raise ValueError(
                    f"No conversations.json found under {source}. "
                    "Unzip your ChatGPT data export first, or point at the file directly."
                )
            path = matches[0]
    if not path.exists():
        raise ValueError(f"ChatGPT export not found: {path}")
    try:
        # ``utf-8-sig`` so a UTF-8 BOM (common in exports and Windows-written
        # files) doesn't trip the JSON parser.
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}")
    if not isinstance(data, list):
        raise ValueError(
            f"{path} should be a JSON list of conversations; got {type(data).__name__}. "
            "This file does not look like a ChatGPT conversations.json export."
        )
    return [c for c in data if isinstance(c, dict)]


def _text_parts(message: dict[str, Any]) -> list[str]:
    """Pull the human-readable text parts out of a message node."""
    content = message.get("content") or {}
    parts = content.get("parts") or []
    out: list[str] = []
    for part in parts:
        if isinstance(part, str) and part.strip():
            out.append(part.strip())
    return out


def _node_has_text(node: dict[str, Any]) -> bool:
    message = node.get("message") or {}
    if not message:
        return False
    return bool(_text_parts(message))


def _thread_nodes(conv: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the conversation's message nodes in thread order.

    Follow the parent chain back from ``current_node`` when present (the live
    branch). Otherwise fall back to every node that carries text, preserving
    the export's own ordering, so branched/edited histories are not lost.
    """
    mapping = conv.get("mapping") or {}
    if not isinstance(mapping, dict) or not mapping:
        return []

    current = conv.get("current_node")
    if current and current in mapping:
        # Walk current -> root via parents, then reverse for chronological order.
        chain: list[dict[str, Any]] = []
        seen: set[str] = set()
        node_id: str | None = current
        while node_id and node_id in mapping and node_id not in seen:
            seen.add(node_id)
            chain.append(mapping[node_id])
            node_id = mapping[node_id].get("parent")
        chain.reverse()
        return chain

    # No current_node (older exports): every text node, in export order.
    return [n for n in mapping.values() if isinstance(n, dict) and _node_has_text(n)]


def _iter_memory_rows(
    conv: dict[str, Any], *, include_assistant: bool
) -> list[dict[str, Any]]:
    conv_id = str(conv.get("conversation_id") or conv.get("id") or "")
    title = (conv.get("title") or "").strip() or "Untitled conversation"
    rows: list[dict[str, Any]] = []

    for node in _thread_nodes(conv):
        message = node.get("message") or {}
        if not message:
            continue
        author = (message.get("author") or {}).get("role") or ""
        role = str(author).lower()
        if role in _SKIP_ROLES:
            continue
        if role == "assistant" and not include_assistant:
            continue

        text = "\n\n".join(_text_parts(message)).strip()
        if not text:
            continue

        msg_id = str(message.get("id") or node.get("id") or "")
        rows.append(
            {
                "id": f"chatgpt:{conv_id}:{msg_id}",
                "memory": text,
                "role": "user" if role == "user" else "assistant",
                "created_at": message.get("create_time"),
                "conversation_id": conv_id,
                "conversation_title": title,
            }
        )
    return rows


def run_chatgpt_export(
    source: str | Path,
    run_dir: Path,
    *,
    include_assistant: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Transform a ChatGPT ``conversations.json`` into a migration export.

    Returns ``(export_path, export_dict)`` — the same ``(Path, dict)`` contract
    the API-backed exporters honour, so ``memanto migrate chatgpt`` plugs into
    the existing runner unchanged.
    """
    progress = on_progress or (lambda _msg: None)
    source_path = Path(source).expanduser()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    progress(f"Reading ChatGPT export from {source_path}")
    conversations = _load_conversations(source_path)

    memories: list[dict[str, Any]] = []
    conv_summaries: list[dict[str, Any]] = []
    for conv in conversations:
        rows = _iter_memory_rows(conv, include_assistant=include_assistant)
        memories.extend(rows)
        conv_summaries.append(
            {
                "id": str(conv.get("conversation_id") or conv.get("id") or ""),
                "title": (conv.get("title") or "").strip() or "Untitled conversation",
                "create_time": conv.get("create_time"),
                "update_time": conv.get("update_time"),
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
