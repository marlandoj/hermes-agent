"""Async subprocess bridge to zo-memory CLI.

All calls are non-blocking: they run in a background thread pool so the
Hermes event loop is never blocked. A hard timeout ensures runaway CLI
processes don't hang the agent.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Dict, List, Optional

from .config import ZouroborosConfig

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="zo-bridge")


class ZoMemoryBridge:
    """Subprocess wrapper for the zo-memory CLI."""

    def __init__(self, config: ZouroborosConfig):
        self._cfg = config
        self._cli = config.cli_path
        self._timeout_s = config.bridge_timeout_ms / 1000.0

    def _run_cli(self, args: List[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess:
        timeout = timeout or self._timeout_s
        cmd = ["bun", self._cli] + args
        logger.debug("zo-bridge: %s", " ".join(cmd))
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/home/workspace",
        )

    def hybrid_search(self, query: str, *, persona: Optional[str] = None,
                      limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Synchronous hybrid search — call from thread pool."""
        args = ["hybrid", query, "--json", "true"]
        if persona:
            args.extend(["--persona", persona])
        if limit:
            args.extend(["--limit", str(limit)])
        try:
            result = self._run_cli(args)
            if result.returncode != 0:
                logger.warning("zo-bridge hybrid failed: %s", result.stderr.strip())
                return []
            return json.loads(result.stdout) if result.stdout.strip() else []
        except subprocess.TimeoutExpired:
            logger.warning("zo-bridge hybrid timed out after %.1fs", self._timeout_s)
            return []
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("zo-bridge hybrid error: %s", e)
            return []

    def store(self, *, entity: str, key: str, value: str,
              persona: str = "shared", decay: str = "active",
              category: str = "fact", source: str = "hermes") -> Optional[str]:
        """Synchronous store — call from thread pool."""
        args = [
            "store",
            "--entity", entity,
            "--key", key,
            "--value", value,
            "--persona", persona,
            "--decay", decay,
            "--category", category,
            "--source", source,
        ]
        try:
            result = self._run_cli(args, timeout=2.0)
            if result.returncode != 0:
                logger.warning("zo-bridge store failed: %s", result.stderr.strip())
                return None
            for line in result.stdout.strip().splitlines():
                if line.startswith("Stored:"):
                    return line.split(":", 1)[1].strip()
            return "ok"
        except subprocess.TimeoutExpired:
            logger.warning("zo-bridge store timed out")
            return None
        except Exception as e:
            logger.warning("zo-bridge store error: %s", e)
            return None

    def stats(self) -> str:
        """Get memory stats (blocking, for health checks)."""
        try:
            result = self._run_cli(["stats"], timeout=5.0)
            return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr.strip()}"
        except Exception as e:
            return f"Error: {e}"

    # --- Async wrappers (fire-and-forget or awaitable) ---

    def async_hybrid_search(self, query: str, *, persona: Optional[str] = None,
                            limit: Optional[int] = None) -> Future:
        """Non-blocking hybrid search returning a Future."""
        return _executor.submit(self.hybrid_search, query, persona=persona, limit=limit)

    def async_store(self, **kwargs) -> Future:
        """Non-blocking store returning a Future (fire-and-forget)."""
        return _executor.submit(self.store, **kwargs)

    def shutdown(self):
        """Shutdown the thread pool."""
        _executor.shutdown(wait=False)
