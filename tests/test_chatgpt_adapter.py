"""Tests for the ChatGPT -> Memanto migration adapter (issue #1609 Path B).

Covers the full loop: read a ChatGPT ``conversations.json`` -> transform into a
provider export -> map onto the Memanto schema -> dry-run migration summary.
"""

import json

import pytest

from memanto.cli.analyze.chatgpt_export import (
    _load_conversations,
    run_chatgpt_export,
)
from memanto.cli.migrate.mappers import map_chatgpt
from memanto.cli.migrate.runner import run_migration


def _sample_conversations() -> list[dict]:
    """A small but realistic ChatGPT export: a tree `mapping` with a live
    current_node branch, plus system/tool/assistant/user roles to filter."""
    return [
        {
            "title": "Trip to Chengdu",
            "create_time": 1735689600.0,
            "update_time": 1735693200.0,
            "conversation_id": "conv-abc",
            "current_node": "n4",
            "mapping": {
                "root": {
                    "id": "root",
                    "message": None,
                    "parent": None,
                    "children": ["n1"],
                },
                "n1": {
                    "id": "n1",
                    "message": {
                        "id": "m1",
                        "author": {"role": "system"},
                        "create_time": 1735689601.0,
                        "content": {"content_type": "text", "parts": ["You are ChatGPT."]},
                    },
                    "parent": "root",
                    "children": ["n2"],
                },
                "n2": {
                    "id": "n2",
                    "message": {
                        "id": "m2",
                        "author": {"role": "user"},
                        "create_time": 1735689605.0,
                        "content": {
                            "content_type": "text",
                            "parts": ["I live in Chengdu and I prefer concise answers."],
                        },
                    },
                    "parent": "n1",
                    "children": ["n3"],
                },
                "n3": {
                    "id": "n3",
                    "message": {
                        "id": "m3",
                        "author": {"role": "assistant"},
                        "create_time": 1735689606.0,
                        "content": {"content_type": "text", "parts": ["Noted. I'll keep it brief."]},
                    },
                    "parent": "n2",
                    "children": ["n4"],
                },
                "n4": {
                    "id": "n4",
                    "message": {
                        "id": "m4",
                        "author": {"role": "user"},
                        "create_time": 1735689610.0,
                        "content": {
                            "content_type": "text",
                            "parts": ["My favorite hotpot place is on Fucheng Avenue."],
                        },
                    },
                    "parent": "n3",
                    "children": [],
                },
            },
        },
        {
            # No current_node: exporter must fall back to all text nodes.
            "title": "Legacy chat",
            "create_time": 1735000000.0,
            "update_time": 1735000100.0,
            "conversation_id": "conv-xyz",
            "mapping": {
                "a": {
                    "id": "a",
                    "message": {
                        "id": "ma",
                        "author": {"role": "user"},
                        "create_time": 1735000001.0,
                        "content": {"content_type": "text", "parts": ["Remember my editor is Neovim."]},
                    },
                    "parent": None,
                    "children": [],
                },
            },
        },
    ]


def _write_export(tmp_path, conversations) -> "object":
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps(conversations), encoding="utf-8")
    return path


class TestLoadConversations:
    def test_reads_file_directly(self, tmp_path):
        path = _write_export(tmp_path, _sample_conversations())
        convs = _load_conversations(path)
        assert len(convs) == 2

    def test_reads_from_directory(self, tmp_path):
        _write_export(tmp_path, _sample_conversations())
        convs = _load_conversations(tmp_path)
        assert len(convs) == 2

    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            _load_conversations(tmp_path / "nope.json")

    def test_rejects_non_list_json(self, tmp_path):
        path = tmp_path / "conversations.json"
        path.write_text('{"not": "a list"}', encoding="utf-8")
        with pytest.raises(ValueError, match="JSON list"):
            _load_conversations(path)


class TestChatgptExport:
    def test_export_shape_and_counts(self, tmp_path):
        src = _write_export(tmp_path, _sample_conversations())
        run_dir = tmp_path / "out"
        export_path, export = run_chatgpt_export(src, run_dir)

        assert export_path.exists()
        assert export["source"] == "chatgpt"
        assert export["conversation_count"] == 2
        # user m2, assistant m3, user m4 (live branch) + legacy user ma = 4.
        assert export["memory_count"] == 4
        assert len(export["memories"]) == 4
        assert len(export["conversations"]) == 2

        roles = {m["role"] for m in export["memories"]}
        assert roles == {"user", "assistant"}
        # system message must be filtered out
        assert all("You are ChatGPT." not in m["memory"] for m in export["memories"])
        # ids are stable and traceable
        assert all(m["id"].startswith("chatgpt:conv-") for m in export["memories"])

    def test_user_only_excludes_assistant(self, tmp_path):
        src = _write_export(tmp_path, _sample_conversations())
        _, export = run_chatgpt_export(src, tmp_path / "out", include_assistant=False)
        assert all(m["role"] == "user" for m in export["memories"])
        # m2, m4, ma remain (3 user messages)
        assert export["memory_count"] == 3

    def test_export_file_roundtrips_as_json(self, tmp_path):
        src = _write_export(tmp_path, _sample_conversations())
        export_path, export = run_chatgpt_export(src, tmp_path / "out")
        on_disk = json.loads(export_path.read_text(encoding="utf-8"))
        assert on_disk["memory_count"] == export["memory_count"]


class TestMapChatgpt:
    def _export(self, tmp_path):
        src = _write_export(tmp_path, _sample_conversations())
        _, export = run_chatgpt_export(src, tmp_path / "out")
        return export

    def test_maps_to_valid_memanto_payloads(self, tmp_path):
        export = self._export(tmp_path)
        rows = map_chatgpt(export)
        assert len(rows) == export["memory_count"]
        for row in rows:
            # required schema slots
            assert row["title"]
            assert row["content"]
            assert row["source"] == "chatgpt"
            assert row["provenance"] == "imported"
            assert isinstance(row["tags"], list)
            assert 0.0 <= row["confidence"] <= 1.0
            assert row["source_ref"]
            # content stays within the Memanto content cap
            assert len(row["content"]) <= 10000
            assert len(row["title"]) <= 100

    def test_conversation_title_and_role_become_tags(self, tmp_path):
        export = self._export(tmp_path)
        rows = map_chatgpt(export)
        chengdu = [r for r in rows if "Trip to Chengdu" in r["tags"]]
        assert chengdu, "conversation title should become a tag"
        assert any("role=user" in r["tags"] for r in chengdu)

    def test_supporting_data_footer_present(self, tmp_path):
        export = self._export(tmp_path)
        rows = map_chatgpt(export)
        assert any("[Supporting data]" in r["content"] for r in rows)


class TestChatgptMigrationDryRun:
    def test_run_migration_dry_run(self, tmp_path):
        src = _write_export(tmp_path, _sample_conversations())
        _, export = run_chatgpt_export(src, tmp_path / "out")
        summary, rows = run_migration(
            provider="chatgpt",
            export=export,
            client=None,
            agent_id="",
            dry_run=True,
        )
        assert summary.mapped_count == export["memory_count"]
        assert summary.source_count == export["memory_count"]
        assert summary.imported == 0
        assert summary.failed == 0
        assert len(rows) == export["memory_count"]
