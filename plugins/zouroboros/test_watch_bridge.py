"""Tests for the Zouroboros Watch Pattern → Sentinel Bridge."""

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from watch_bridge import WatchBridge, init_bridge, get_bridge


class TestWatchBridge(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="swarm-events-test-")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _make_notification(self, **overrides):
        base = {
            "session_id": "test-session-1",
            "command": "npm run build",
            "type": "watch_match",
            "pattern": "BUILD SUCCESS",
            "output": "✓ Build completed in 4.2s\nTotal: 42 modules",
            "suppressed": 0,
        }
        base.update(overrides)
        return base

    def test_init_creates_sentinel_dir(self):
        nested = os.path.join(self.test_dir, "nested", "deep")
        bridge = WatchBridge(sentinel_dir=nested)
        self.assertTrue(os.path.isdir(nested))

    def test_on_notification_writes_sentinel(self):
        bridge = WatchBridge(sentinel_dir=self.test_dir)
        bridge.on_notification(self._make_notification())

        files = [f for f in os.listdir(self.test_dir) if f.endswith(".json")]
        self.assertEqual(len(files), 1)

        with open(os.path.join(self.test_dir, files[0])) as f:
            event = json.load(f)

        self.assertEqual(event["task_id"], "test-session-1")
        self.assertEqual(event["event_type"], "pattern_match")
        self.assertEqual(event["pattern"], "BUILD SUCCESS")
        self.assertEqual(event["source"], "hermes:watch_patterns")

    def test_ignores_non_watch_match(self):
        bridge = WatchBridge(sentinel_dir=self.test_dir)
        bridge.on_notification({"type": "process_exit", "session_id": "x"})

        files = [f for f in os.listdir(self.test_dir) if f.endswith(".json")]
        self.assertEqual(len(files), 0)

    def test_atomic_write_no_tmp_files(self):
        bridge = WatchBridge(sentinel_dir=self.test_dir)
        bridge.on_notification(self._make_notification())

        files = os.listdir(self.test_dir)
        tmp_files = [f for f in files if f.startswith(".tmp_")]
        self.assertEqual(len(tmp_files), 0)

    def test_extracts_first_line(self):
        bridge = WatchBridge(sentinel_dir=self.test_dir)
        bridge.on_notification(self._make_notification(
            output="\n\n  ✓ Build completed in 4.2s\nExtra line"
        ))

        files = [f for f in os.listdir(self.test_dir) if f.endswith(".json")]
        with open(os.path.join(self.test_dir, files[0])) as f:
            event = json.load(f)
        self.assertEqual(event["matched_line"], "✓ Build completed in 4.2s")

    def test_metadata_includes_command_and_suppressed(self):
        bridge = WatchBridge(sentinel_dir=self.test_dir)
        bridge.on_notification(self._make_notification(suppressed=5))

        files = [f for f in os.listdir(self.test_dir) if f.endswith(".json")]
        with open(os.path.join(self.test_dir, files[0])) as f:
            event = json.load(f)
        self.assertEqual(event["metadata"]["command"], "npm run build")
        self.assertEqual(event["metadata"]["suppressed"], 5)

    def test_stats_tracking(self):
        bridge = WatchBridge(sentinel_dir=self.test_dir)
        bridge.on_notification(self._make_notification())
        bridge.on_notification(self._make_notification(session_id="s2"))

        self.assertEqual(bridge.stats["sentinels_written"], 2)
        self.assertEqual(bridge.stats["errors"], 0)

    def test_singleton_init_and_get(self):
        bridge = init_bridge(sentinel_dir=self.test_dir)
        self.assertIs(get_bridge(), bridge)

    def test_rapid_writes(self):
        bridge = WatchBridge(sentinel_dir=self.test_dir)
        for i in range(50):
            bridge.on_notification(self._make_notification(session_id=f"rapid-{i}"))

        files = [f for f in os.listdir(self.test_dir) if f.endswith(".json")]
        self.assertEqual(len(files), 50)
        self.assertEqual(bridge.stats["sentinels_written"], 50)

    def test_sentinel_schema_compatibility(self):
        """Verify the sentinel file matches the TypeScript SwarmEvent schema."""
        bridge = WatchBridge(sentinel_dir=self.test_dir)
        bridge.on_notification(self._make_notification())

        files = [f for f in os.listdir(self.test_dir) if f.endswith(".json")]
        with open(os.path.join(self.test_dir, files[0])) as f:
            event = json.load(f)

        required_fields = ["task_id", "event_type", "timestamp", "source"]
        for field in required_fields:
            self.assertIn(field, event, f"Missing required field: {field}")

        valid_types = ["pattern_match", "task_complete", "task_failed", "health_check"]
        self.assertIn(event["event_type"], valid_types)


if __name__ == "__main__":
    unittest.main()
