"""Tests for the Zouroboros memory provider.

Run: python -m pytest plugins/memory/zouroboros/test_zouroboros.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.memory.zouroboros.config import ZouroborosConfig, load_config
from plugins.memory.zouroboros.bridge import ZoMemoryBridge
from plugins.memory.zouroboros.provider import (
    ZouroborosMemoryProvider,
    _extract_entities,
    _summarize_for_storage,
)


# ---- Config tests -----------------------------------------------------------

class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = ZouroborosConfig()
        self.assertEqual(cfg.persona, "hermes")
        self.assertEqual(cfg.bridge_timeout_ms, 500)
        self.assertEqual(cfg.prefetch_limit, 6)
        self.assertTrue(cfg.async_writes)

    def test_env_override(self):
        with patch.dict(os.environ, {"ZO_MEMORY_PERSONA": "alaric", "ZO_MEMORY_BRIDGE_TIMEOUT_MS": "1000"}):
            cfg = load_config()
        self.assertEqual(cfg.persona, "alaric")
        self.assertEqual(cfg.bridge_timeout_ms, 1000)

    def test_env_invalid_int_ignored(self):
        with patch.dict(os.environ, {"ZO_MEMORY_BRIDGE_TIMEOUT_MS": "not_a_number"}):
            cfg = load_config()
        self.assertEqual(cfg.bridge_timeout_ms, 500)


# ---- Bridge tests -----------------------------------------------------------

class TestBridge(unittest.TestCase):
    def setUp(self):
        self.cfg = ZouroborosConfig(bridge_timeout_ms=500)
        self.bridge = ZoMemoryBridge(self.cfg)

    @patch("plugins.memory.zouroboros.bridge.subprocess.run")
    def test_hybrid_search_success(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps([{"entity": "test", "key": "k", "value": "v", "score": 0.9}]),
            stderr="",
        )
        results = self.bridge.hybrid_search("test query")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["entity"], "test")

    @patch("plugins.memory.zouroboros.bridge.subprocess.run")
    def test_hybrid_search_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="bun", timeout=0.5)
        results = self.bridge.hybrid_search("test query")
        self.assertEqual(results, [])

    @patch("plugins.memory.zouroboros.bridge.subprocess.run")
    def test_hybrid_search_nonzero_exit(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="Error: DB not found",
        )
        results = self.bridge.hybrid_search("test query")
        self.assertEqual(results, [])

    @patch("plugins.memory.zouroboros.bridge.subprocess.run")
    def test_hybrid_search_invalid_json(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not json", stderr="",
        )
        results = self.bridge.hybrid_search("test query")
        self.assertEqual(results, [])

    @patch("plugins.memory.zouroboros.bridge.subprocess.run")
    def test_store_success(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Stored: abc-123\nEmbedding: generated", stderr="",
        )
        result = self.bridge.store(entity="test", key="k", value="v")
        self.assertEqual(result, "abc-123")

    @patch("plugins.memory.zouroboros.bridge.subprocess.run")
    def test_store_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="bun", timeout=2.0)
        result = self.bridge.store(entity="test", key="k", value="v")
        self.assertIsNone(result)

    @patch("plugins.memory.zouroboros.bridge.subprocess.run")
    def test_store_failure(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="Error: missing entity",
        )
        result = self.bridge.store(entity="test", key="k", value="v")
        self.assertIsNone(result)

    @patch("plugins.memory.zouroboros.bridge.subprocess.run")
    def test_async_search_returns_future(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="[]", stderr="",
        )
        future = self.bridge.async_hybrid_search("query")
        self.assertIsInstance(future, Future)
        result = future.result(timeout=5)
        self.assertEqual(result, [])


# ---- Provider tests ---------------------------------------------------------

class TestProvider(unittest.TestCase):
    def setUp(self):
        self.provider = ZouroborosMemoryProvider()
        self.mock_bridge = MagicMock(spec=ZoMemoryBridge)
        self.provider._bridge = self.mock_bridge
        self.provider._config = ZouroborosConfig()
        self.provider._session_id = "test-session-123"

    def test_name(self):
        self.assertEqual(self.provider.name, "zouroboros")

    def test_is_available_no_db(self):
        provider = ZouroborosMemoryProvider()
        with patch("plugins.memory.zouroboros.provider.load_config") as mock_cfg:
            cfg = ZouroborosConfig(shared_db="/nonexistent/path.db", cli_path="/nonexistent/cli.ts")
            mock_cfg.return_value = cfg
            self.assertFalse(provider.is_available())

    def test_system_prompt_block(self):
        block = self.provider.system_prompt_block()
        self.assertIn("Zouroboros", block)
        self.assertIn("cross-persona", block)

    def test_prefetch_formats_results(self):
        self.mock_bridge.hybrid_search.return_value = [
            {"entity": "user", "key": "pref", "value": "dark mode", "decayClass": "stable", "score": 0.85},
        ]
        result = self.provider.prefetch("dark mode")
        self.assertIn("user.pref", result)
        self.assertIn("dark mode", result)
        self.assertIn("0.85", result)

    def test_prefetch_empty_results(self):
        self.mock_bridge.hybrid_search.return_value = []
        result = self.provider.prefetch("random query")
        self.assertEqual(result, "")

    def test_sync_turn_stores_decision(self):
        self.provider.sync_turn(
            "I decided to use PostgreSQL",
            "Understood, I've confirmed PostgreSQL as the database choice.",
        )
        self.mock_bridge.async_store.assert_called_once()
        call_kwargs = self.mock_bridge.async_store.call_args[1]
        self.assertEqual(call_kwargs["decay"], "active")
        self.assertEqual(call_kwargs["source"], "hermes")

    def test_sync_turn_skips_conversational(self):
        self.provider.sync_turn("hello", "Hi there!")
        self.mock_bridge.async_store.assert_not_called()

    def test_on_pre_compress_stores_insights(self):
        messages = [
            {"role": "assistant", "content": "A" * 150 + ". Short sentence."},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "B" * 200 + ". Another point here."},
        ]
        result = self.provider.on_pre_compress(messages)
        self.assertIn("insights extracted", result)
        self.mock_bridge.async_store.assert_called_once()
        call_kwargs = self.mock_bridge.async_store.call_args[1]
        self.assertEqual(call_kwargs["persona"], "mimir")

    def test_on_pre_compress_empty_messages(self):
        result = self.provider.on_pre_compress([])
        self.assertEqual(result, "")
        self.mock_bridge.async_store.assert_not_called()

    def test_tool_schemas_exposed(self):
        schemas = self.provider.get_tool_schemas()
        names = {s["name"] for s in schemas}
        self.assertIn("zo_memory_search", names)
        self.assertIn("zo_memory_store", names)

    def test_handle_search_tool(self):
        self.mock_bridge.hybrid_search.return_value = [{"entity": "test", "value": "v"}]
        result = json.loads(self.provider.handle_tool_call("zo_memory_search", {"query": "test"}))
        self.assertEqual(result["count"], 1)

    def test_handle_store_tool(self):
        self.mock_bridge.store.return_value = "abc-123"
        result = json.loads(self.provider.handle_tool_call(
            "zo_memory_store",
            {"entity": "user", "key": "pref", "value": "dark mode"},
        ))
        self.assertTrue(result["stored"])

    def test_handle_unknown_tool(self):
        result = json.loads(self.provider.handle_tool_call("unknown_tool", {}))
        self.assertIn("error", result)

    def test_on_session_end_stores_summary(self):
        messages = [
            {"role": "user", "content": "Help with trading"},
            {"role": "assistant", "content": "Sure"},
            {"role": "user", "content": "Optimize the backtest"},
        ]
        self.provider._turn_count = 2
        self.provider.on_session_end(messages)
        self.mock_bridge.async_store.assert_called_once()
        call_kwargs = self.mock_bridge.async_store.call_args[1]
        self.assertIn("2 turns", call_kwargs["value"])
        self.assertEqual(call_kwargs["decay"], "stable")

    def test_on_memory_write_mirrors(self):
        self.provider.on_memory_write("add", "memory", "user prefers dark mode")
        self.mock_bridge.async_store.assert_called_once()

    def test_on_memory_write_skips_remove(self):
        self.provider.on_memory_write("remove", "memory", "deleted entry")
        self.mock_bridge.async_store.assert_not_called()

    def test_shutdown_flushes_pending(self):
        mock_future = MagicMock(spec=Future)
        mock_future.done.return_value = False
        self.provider._pending_writes = [mock_future]
        self.provider.shutdown()
        mock_future.result.assert_called_once()
        self.mock_bridge.shutdown.assert_called_once()


# ---- Utility tests ----------------------------------------------------------

class TestUtils(unittest.TestCase):
    def test_extract_entities(self):
        text = "See [[project.jhf]] and [[user.preference|user pref]] for details"
        entities = _extract_entities(text)
        self.assertEqual(entities, ["project.jhf", "user.preference"])

    def test_summarize_decision(self):
        result = _summarize_for_storage(
            "I decided to use dark mode",
            "Confirmed. Dark mode has been set as your default preference.",
        )
        self.assertIsNotNone(result)
        self.assertIn("Dark mode", result)

    def test_summarize_skips_chat(self):
        result = _summarize_for_storage("hello", "Hi there!")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
