"""Tests for the Claude -> Memanto migration adapter (companion to chatgpt)."""

import json

import pytest

from memanto.cli.analyze.claude_export import (
    _load_conversations,
    run_claude_export,
)
from memanto.cli.migrate.mappers import map_claude
from memanto.cli.migrate.runner import run_migration


def _sample_claude() -> list[dict]:
    return [
        {
            "uuid": "conv-claude-1",
            "name": "Chengdu food spots",
            "created_at": "2025-01-01T00:00:00.000000Z",
            "updated_at": "2025-01-02T00:00:00.000000Z",
            "chat_messages": [
                {
                    "uuid": "msg-1",
                    "sender": "human",
                    "text": "I live in Chengdu and I prefer concise answers.",
                    "created_at": "2025-01-01T00:00:05.000000Z",
                },
                {
                    "uuid": "msg-2",
                    "sender": "assistant",
                    "text": "Noted. I'll keep it brief.",
                    "created_at": "2025-01-01T00:00:06.000000Z",
                },
                {
                    "uuid": "msg-3",
                    "sender": "human",
                    "text": "My favorite hotpot place is on Fucheng Avenue.",
                    "created_at": "2025-01-01T00:00:10.000000Z",
                },
            ],
        },
        {
            # Newer-style export: content blocks + "messages" key + "role".
            "uuid": "conv-claude-2",
            "name": "Tooling",
            "created_at": "2025-01-03T00:00:00.000000Z",
            "updated_at": "2025-01-03T00:00:00.000000Z",
            "messages": [
                {
                    "uuid": "msg-9",
                    "role": "user",
                    "content": [{"type": "text", "text": "My editor is Neovim."}],
                    "created_at": "2025-01-03T00:00:01.000000Z",
                }
            ],
        },
    ]


def _write(tmp_path, data):
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestClaudeLoad:
    def test_reads_file(self, tmp_path):
        path = _write(tmp_path, _sample_claude())
        assert len(_load_conversations(path)) == 2

    def test_rejects_non_list(self, tmp_path):
        path = tmp_path / "conversations.json"
        path.write_text('{"x": 1}', encoding="utf-8")
        with pytest.raises(ValueError, match="JSON list"):
            _load_conversations(path)


class TestClaudeExport:
    def test_export_counts_and_roles(self, tmp_path):
        src = _write(tmp_path, _sample_claude())
        _, export = run_claude_export(src, tmp_path / "out")
        assert export["source"] == "claude"
        assert export["conversation_count"] == 2
        # msg-1(human) msg-2(assistant) msg-3(human) msg-9(user) = 4
        assert export["memory_count"] == 4
        roles = {m["role"] for m in export["memories"]}
        assert roles == {"user", "assistant"}
        assert all(m["id"].startswith("claude:conv-claude-") for m in export["memories"])

    def test_tolerates_format_variations(self, tmp_path):
        # conv-claude-2 uses "messages" key + "role" + content blocks
        src = _write(tmp_path, _sample_claude())
        _, export = run_claude_export(src, tmp_path / "out")
        texts = [m["memory"] for m in export["memories"]]
        assert any("Neovim" in t for t in texts)

    def test_user_only(self, tmp_path):
        src = _write(tmp_path, _sample_claude())
        _, export = run_claude_export(src, tmp_path / "out", include_assistant=False)
        assert all(m["role"] == "user" for m in export["memories"])


class TestMapClaude:
    def test_valid_payloads(self, tmp_path):
        src = _write(tmp_path, _sample_claude())
        _, export = run_claude_export(src, tmp_path / "out")
        rows = map_claude(export)
        assert len(rows) == export["memory_count"]
        for row in rows:
            assert row["title"]
            assert row["content"]
            assert row["source"] == "claude"
            assert row["provenance"] == "imported"
            assert isinstance(row["tags"], list)
            assert row["source_ref"]

    def test_source_label_is_claude_not_chatgpt(self, tmp_path):
        src = _write(tmp_path, _sample_claude())
        _, export = run_claude_export(src, tmp_path / "out")
        rows = map_claude(export)
        assert all(r["source"] == "claude" for r in rows)


class TestClaudeMigrationDryRun:
    def test_dry_run(self, tmp_path):
        src = _write(tmp_path, _sample_claude())
        _, export = run_claude_export(src, tmp_path / "out")
        summary, rows = run_migration(
            provider="claude",
            export=export,
            client=None,
            agent_id="",
            dry_run=True,
        )
        assert summary.mapped_count == export["memory_count"]
        assert summary.imported == 0
        assert summary.failed == 0
