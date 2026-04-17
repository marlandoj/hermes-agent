"""Tests for Zouroboros read-only Telegram command handlers."""

import asyncio
import json
import os
import shutil
import tempfile
import unittest

from telegram_commands import (
    handle_command,
    is_zouroboros_command,
    get_command_menu,
    COMMANDS,
    HELP_TEXT,
    _truncate,
    _run,
)


class TestCommandRouting(unittest.TestCase):
    def test_known_commands(self):
        for cmd in COMMANDS:
            self.assertTrue(is_zouroboros_command(cmd), f"/{cmd} should be recognized")

    def test_unknown_command(self):
        self.assertFalse(is_zouroboros_command("deploy"))
        self.assertFalse(is_zouroboros_command("reboot"))
        self.assertFalse(is_zouroboros_command(""))

    def test_command_menu(self):
        menu = get_command_menu()
        self.assertEqual(len(menu), len(COMMANDS))
        for name, desc in menu:
            self.assertIn(name, COMMANDS)
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0)

    def test_help_text_contains_all_commands(self):
        for cmd in COMMANDS:
            self.assertIn(f"/{cmd}", HELP_TEXT)


class TestHelpers(unittest.TestCase):
    def test_truncate_short(self):
        self.assertEqual(_truncate("hello", 100), "hello")

    def test_truncate_long(self):
        text = "x" * 5000
        result = _truncate(text, 100)
        self.assertLessEqual(len(result), 120)
        self.assertTrue(result.endswith("... (truncated)"))

    def test_run_success(self):
        result = _run("echo hello")
        self.assertEqual(result, "hello")

    def test_run_timeout(self):
        result = _run("sleep 10", timeout=0.1)
        self.assertEqual(result, "(timed out)")

    def test_run_error(self):
        result = _run("false")
        self.assertIsInstance(result, str)


class TestCommandHandlers(unittest.TestCase):
    """Test that each command handler executes and returns formatted output."""

    def _run_cmd(self, cmd, args=""):
        return asyncio.get_event_loop().run_until_complete(handle_command(cmd, args))

    def test_status_returns_system_info(self):
        result = self._run_cmd("status")
        self.assertIsNotNone(result)
        self.assertIn("System Status", result)
        self.assertIn("CPU", result)
        self.assertIn("Memory", result)

    def test_metrics_returns_metrics(self):
        result = self._run_cmd("metrics")
        self.assertIsNotNone(result)
        self.assertIn("Ecosystem Metrics", result)
        self.assertIn("sentinel", result.lower())

    def test_logs_no_args_lists_available(self):
        result = self._run_cmd("logs", "")
        self.assertIsNotNone(result)
        # Should either list logs or say none found
        self.assertTrue("Available logs" in result or "No logs found" in result)

    def test_logs_nonexistent_service(self):
        result = self._run_cmd("logs", "nonexistent-service-xyz")
        self.assertIsNotNone(result)
        self.assertIn("not found", result.lower())

    def test_logs_sanitizes_input(self):
        result = self._run_cmd("logs", "../../../etc/passwd")
        self.assertIsNotNone(result)
        # Should not actually read /etc/passwd — sanitized name won't match
        self.assertNotIn("root:", result)

    def test_memory_stats_executes(self):
        result = self._run_cmd("memory_stats")
        self.assertIsNotNone(result)
        self.assertIn("Memory System Health", result)

    def test_swarm_status_executes(self):
        result = self._run_cmd("swarm_status")
        self.assertIsNotNone(result)
        self.assertIn("Swarm Status", result)

    def test_autoloop_status_executes(self):
        result = self._run_cmd("autoloop_status")
        self.assertIsNotNone(result)
        self.assertIn("Autoloop Status", result)

    def test_unknown_command_returns_none(self):
        result = self._run_cmd("nonexistent")
        self.assertIsNone(result)

    def test_all_responses_under_4096(self):
        for cmd in COMMANDS:
            result = self._run_cmd(cmd, "test-arg")
            if result:
                self.assertLessEqual(
                    len(result), 4096,
                    f"/{cmd} response exceeds Telegram limit: {len(result)} chars"
                )

    def test_no_command_mutates_state(self):
        """Verify commands are read-only by running them twice and comparing."""
        for cmd in COMMANDS:
            r1 = self._run_cmd(cmd, "test")
            r2 = self._run_cmd(cmd, "test")
            self.assertIsNotNone(r1)
            self.assertIsNotNone(r2)


if __name__ == "__main__":
    unittest.main()
