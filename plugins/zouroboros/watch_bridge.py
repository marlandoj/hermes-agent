"""Zouroboros Watch Pattern → Sentinel Bridge for Hermes Agent.

When Hermes' ProcessRegistry fires a watch_pattern notification, this bridge
writes a file sentinel to /tmp/swarm-events/ (or configured directory) so the
Zouroboros orchestrator can consume it and advance DAG transitions.

Optionally also POSTs to the HTTP callback endpoint (localhost:7821) for
real-time reactive orchestration.

Usage:
    Enable via --swarm-events flag or SWARM_EVENTS_ENABLED=1 env var.
    The bridge hooks into the ProcessRegistry's completion_queue consumer.

Architecture:
    ProcessRegistry._check_watch_patterns()
        → completion_queue.put({type: "watch_match", ...})
        → WatchBridge.on_notification() [registered consumer]
            → write file sentinel (atomic: tmp → rename)
            → POST to HTTP callback (async, fire-and-forget)
"""

import json
import os
import time
import tempfile
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

logger = logging.getLogger("zouroboros.watch_bridge")

DEFAULT_SENTINEL_DIR = "/tmp/swarm-events"
DEFAULT_HTTP_ENDPOINT = "http://localhost:7821/swarm/event"
HTTP_TIMEOUT_SECONDS = 2.0


class WatchBridge:
    """Bridges Hermes watch_pattern notifications to Zouroboros sentinel files."""

    def __init__(
        self,
        sentinel_dir: Optional[str] = None,
        http_endpoint: Optional[str] = None,
        http_enabled: bool = False,
    ):
        self.sentinel_dir = sentinel_dir or os.environ.get(
            "SWARM_SENTINEL_DIR", DEFAULT_SENTINEL_DIR
        )
        self.http_endpoint = http_endpoint or os.environ.get(
            "SWARM_EVENT_HTTP_ENDPOINT", DEFAULT_HTTP_ENDPOINT
        )
        self.http_enabled = http_enabled or os.environ.get(
            "SWARM_EVENT_HTTP_ENABLED", ""
        ).lower() in ("1", "true", "yes")

        self._stats = {
            "sentinels_written": 0,
            "http_posted": 0,
            "http_failures": 0,
            "errors": 0,
        }

        Path(self.sentinel_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "WatchBridge initialized: sentinel_dir=%s, http=%s",
            self.sentinel_dir,
            self.http_enabled,
        )

    def on_notification(self, notification: dict) -> None:
        """Handle a watch_match notification from ProcessRegistry.

        Expected keys: session_id, command, type, pattern, output, suppressed
        """
        if notification.get("type") != "watch_match":
            return

        event = {
            "task_id": notification.get("session_id", "unknown"),
            "event_type": "pattern_match",
            "pattern": notification.get("pattern", ""),
            "matched_line": self._extract_first_line(notification.get("output", "")),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "hermes:watch_patterns",
            "metadata": {
                "command": notification.get("command", ""),
                "suppressed": notification.get("suppressed", 0),
            },
        }

        self._write_sentinel(event)

        if self.http_enabled:
            threading.Thread(
                target=self._post_http, args=(event,), daemon=True
            ).start()

    def _write_sentinel(self, event: dict) -> Optional[str]:
        """Atomically write a sentinel file (write-tmp → rename)."""
        try:
            task_id = event["task_id"]
            uid = uuid4().hex[:8]
            filename = f"{task_id}_{int(time.time() * 1000)}_{uid}.json"
            target = os.path.join(self.sentinel_dir, filename)

            fd, tmp_path = tempfile.mkstemp(
                prefix=".tmp_", suffix=".json", dir=self.sentinel_dir
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(event, f, indent=2)
                os.rename(tmp_path, target)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            self._stats["sentinels_written"] += 1
            logger.debug("Sentinel written: %s", target)
            return target

        except Exception as e:
            self._stats["errors"] += 1
            logger.error("Failed to write sentinel: %s", e)
            return None

    def _post_http(self, event: dict) -> None:
        """Fire-and-forget POST to the HTTP callback endpoint."""
        if not _HAS_HTTPX:
            try:
                import urllib.request
                req = urllib.request.Request(
                    self.http_endpoint,
                    data=json.dumps(event).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS)
                self._stats["http_posted"] += 1
            except Exception as e:
                self._stats["http_failures"] += 1
                logger.debug("HTTP callback failed (urllib): %s", e)
            return

        try:
            with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
                resp = client.post(self.http_endpoint, json=event)
                if resp.status_code in (200, 201, 202):
                    self._stats["http_posted"] += 1
                else:
                    self._stats["http_failures"] += 1
                    logger.debug("HTTP callback returned %d", resp.status_code)
        except Exception as e:
            self._stats["http_failures"] += 1
            logger.debug("HTTP callback failed: %s", e)

    @staticmethod
    def _extract_first_line(text: str) -> str:
        """Extract first non-empty line from output."""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return text[:200] if text else ""

    @property
    def stats(self) -> dict:
        return dict(self._stats)


_bridge_instance: Optional[WatchBridge] = None


def get_bridge() -> Optional[WatchBridge]:
    """Get the singleton WatchBridge instance (if enabled)."""
    return _bridge_instance


def init_bridge(**kwargs) -> WatchBridge:
    """Initialize the singleton WatchBridge."""
    global _bridge_instance
    _bridge_instance = WatchBridge(**kwargs)
    return _bridge_instance


def hook_into_process_registry(registry) -> bool:
    """Hook the WatchBridge into a Hermes ProcessRegistry's completion_queue consumer.

    Call after init_bridge(). Returns True if hooked successfully.
    """
    bridge = get_bridge()
    if not bridge:
        logger.warning("WatchBridge not initialized, cannot hook into ProcessRegistry")
        return False

    original_check = getattr(registry, "_check_watch_patterns", None)
    if not original_check:
        logger.warning("ProcessRegistry has no _check_watch_patterns method")
        return False

    def patched_check(session, new_text):
        original_check(session, new_text)
        # After the original check runs, broadcast any watch_match notifications
        # to the WatchBridge so it can write file sentinels.
        notifications = getattr(registry, "_pending_notifications", None)
        if notifications is not None:
            while True:
                try:
                    notification = notifications.get_nowait()
                    bridge.on_notification(notification)
                except Exception:
                    break
        else:
            # Fallback: inspect the completion_queue for any watch_match notifications
            completion_queue = getattr(registry, "completion_queue", None)
            if completion_queue is not None:
                pending = []
                while True:
                    try:
                        item = completion_queue.get_nowait()
                        if item.get("type") == "watch_match":
                            bridge.on_notification(item)
                        else:
                            pending.append(item)
                    except Exception:
                        break
                for item in pending:
                    try:
                        completion_queue.put_nowait(item)
                    except Exception:
                        pass

    registry._check_watch_patterns = patched_check
    logger.info("WatchBridge hooked into ProcessRegistry._check_watch_patterns")
    return True
