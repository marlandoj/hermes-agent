"""Zouroboros Memory Provider — dual-write bridge.

Implements the MemoryProvider ABC to connect Hermes Agent to the
Zouroboros memory ecosystem:
  - prefetch(): Hybrid search over shared-facts.db (BM25 + vector)
  - sync_turn(): Async fact extraction and storage
  - on_pre_compress(): Extract insights from expiring messages into mimir.db
  - on_session_end(): Final session summary stored as stable fact

Architecture: Hindsight handles fast local recall. This provider adds
cross-persona visibility by bridging to zo-memory asynchronously.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import Future
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from .bridge import ZoMemoryBridge
from .config import ZouroborosConfig, load_config

logger = logging.getLogger(__name__)

_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


def _extract_entities(text: str) -> List[str]:
    """Extract wikilink entity references from text."""
    return _WIKILINK_RE.findall(text)


def _summarize_for_storage(user: str, assistant: str, max_len: int = 300) -> Optional[str]:
    """Extract a storable fact from a turn pair.

    Returns a short summary if the turn contains a decision, preference,
    or notable fact. Returns None if the turn is purely conversational.
    """
    signals = [
        "decided", "agreed", "confirmed", "chose", "approved",
        "preference", "always", "never", "remember",
        "important", "note that", "key point", "conclusion",
    ]
    combined = f"{user} {assistant}".lower()
    if not any(s in combined for s in signals):
        return None

    snippet = assistant[:max_len].strip()
    if len(assistant) > max_len:
        last_period = snippet.rfind(".")
        if last_period > max_len // 2:
            snippet = snippet[: last_period + 1]
    return snippet if snippet else None


class ZouroborosMemoryProvider(MemoryProvider):
    """Bridges Hermes to the Zouroboros zo-memory system."""

    def __init__(self):
        self._config: Optional[ZouroborosConfig] = None
        self._bridge: Optional[ZoMemoryBridge] = None
        self._session_id: str = ""
        self._platform: str = "cli"
        self._turn_count: int = 0
        self._prefetch_cache: str = ""
        self._prefetch_future: Optional[Future] = None
        self._pending_writes: List[Future] = []
        self._available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "zouroboros"

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from pathlib import Path
            cfg = load_config()
            self._available = Path(cfg.shared_db).exists() and Path(cfg.cli_path).exists()
        except Exception:
            self._available = False
        return self._available

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home", "")
        self._config = load_config(hermes_home)
        self._bridge = ZoMemoryBridge(self._config)
        self._session_id = session_id
        self._platform = kwargs.get("platform", "cli")
        self._turn_count = 0
        logger.info(
            "Zouroboros memory bridge initialized (shared=%s, mimir=%s, persona=%s)",
            self._config.shared_db, self._config.mimir_db, self._config.persona,
        )

    def system_prompt_block(self) -> str:
        return (
            "[Zouroboros Memory Bridge active — cross-persona facts from "
            "shared-facts.db are available. Decisions and preferences from "
            "this session will be stored for other Zo personas to access.]"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._bridge:
            return ""
        if self._prefetch_future is not None:
            try:
                results = self._prefetch_future.result(timeout=0.3)
                self._prefetch_future = None
                if results:
                    self._prefetch_cache = self._format_results(results)
            except Exception:
                self._prefetch_future = None

        if not self._prefetch_cache:
            try:
                results = self._bridge.hybrid_search(
                    query,
                    persona=self._config.persona if self._config else None,
                    limit=self._config.prefetch_limit if self._config else 6,
                )
                if results:
                    self._prefetch_cache = self._format_results(results)
            except Exception as e:
                logger.debug("Prefetch fallback failed: %s", e)

        ctx = self._prefetch_cache
        self._prefetch_cache = ""
        return ctx

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._bridge:
            return
        self._prefetch_future = self._bridge.async_hybrid_search(
            query,
            persona=self._config.persona if self._config else None,
            limit=self._config.prefetch_limit if self._config else 6,
        )

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._bridge or not self._config:
            return
        self._turn_count += 1

        summary = _summarize_for_storage(user_content, assistant_content)
        if not summary:
            return

        entity = f"hermes.session.{self._session_id[:8]}"
        key = f"turn_{self._turn_count}"

        if self._config.async_writes:
            future = self._bridge.async_store(
                entity=entity,
                key=key,
                value=summary,
                persona="shared",
                decay="active",
                category="fact",
                source="hermes",
            )
            self._pending_writes.append(future)
            self._pending_writes = [f for f in self._pending_writes if not f.done()]
        else:
            self._bridge.store(
                entity=entity,
                key=key,
                value=summary,
                persona="shared",
                decay="active",
                category="fact",
                source="hermes",
            )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "zo_memory_search",
                "description": (
                    "Search the Zouroboros shared memory for facts, decisions, "
                    "and preferences stored by any Zo persona. Use when you need "
                    "cross-persona context or historical decisions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for hybrid (BM25 + vector) retrieval",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "zo_memory_store",
                "description": (
                    "Store a fact, decision, or preference into Zouroboros shared "
                    "memory so other Zo personas can access it. Use for important "
                    "decisions, user preferences, and cross-session context."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "description": "Entity identifier (e.g., 'user.preference', 'project.jhf')",
                        },
                        "key": {
                            "type": "string",
                            "description": "Attribute name (e.g., 'trading_strategy', 'color_theme')",
                        },
                        "value": {
                            "type": "string",
                            "description": "The fact or decision to store",
                        },
                        "decay": {
                            "type": "string",
                            "enum": ["permanent", "stable", "active", "session"],
                            "description": "How long this fact should persist (default: stable)",
                        },
                        "category": {
                            "type": "string",
                            "enum": ["fact", "preference", "decision", "convention", "reference", "project"],
                            "description": "Category of the memory entry (default: fact)",
                        },
                    },
                    "required": ["entity", "key", "value"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._bridge:
            return json.dumps({"error": "Bridge not initialized"})

        if tool_name == "zo_memory_search":
            query = args.get("query", "")
            results = self._bridge.hybrid_search(
                query,
                persona=self._config.persona if self._config else None,
            )
            return json.dumps({"results": results, "count": len(results)})

        elif tool_name == "zo_memory_store":
            fact_id = self._bridge.store(
                entity=args.get("entity", "hermes"),
                key=args.get("key", ""),
                value=args.get("value", ""),
                persona="shared",
                decay=args.get("decay", "stable"),
                category=args.get("category", "fact"),
                source="hermes",
            )
            if fact_id:
                return json.dumps({"stored": True, "id": fact_id})
            return json.dumps({"stored": False, "error": "Bridge write failed"})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn_count = turn_number

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._bridge or not self._config or not self._config.enable_mimir_synthesis:
            return ""

        insights = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            if role == "assistant" and len(content) > 100:
                snippet = content[:200].strip()
                last_period = snippet.rfind(".")
                if last_period > 50:
                    snippet = snippet[: last_period + 1]
                insights.append(snippet)

        if not insights:
            return ""

        combined = " | ".join(insights[:5])
        self._bridge.async_store(
            entity=f"hermes.synthesis.{self._session_id[:8]}",
            key="pre_compress",
            value=combined,
            persona="mimir",
            decay="stable",
            category="fact",
            source="hermes:compress",
        )

        return f"[Zouroboros: {len(insights)} insights extracted and stored to mimir.db before compression]"

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._bridge or not messages:
            return
        user_msgs = [
            m.get("content", "")[:100]
            for m in messages
            if m.get("role") == "user" and m.get("content")
        ]
        if not user_msgs:
            return

        topics = ", ".join(user_msgs[:5])
        self._bridge.async_store(
            entity=f"hermes.session.{self._session_id[:8]}",
            key="session_summary",
            value=f"Hermes session ({self._platform}): {self._turn_count} turns. Topics: {topics}",
            persona="shared",
            decay="stable",
            category="fact",
            source="hermes:session_end",
        )

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not self._bridge or action == "remove":
            return
        self._bridge.async_store(
            entity="hermes.builtin_mirror",
            key=f"{target}_{action}",
            value=content[:500],
            persona="shared",
            decay="active",
            category="fact",
            source="hermes:builtin_mirror",
        )

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        if not self._bridge:
            return
        summary = result[:300] if result else "no result"
        self._bridge.async_store(
            entity="hermes.delegation",
            key=f"child_{child_session_id[:8]}" if child_session_id else "child",
            value=f"Task: {task[:100]} | Result: {summary}",
            persona="shared",
            decay="active",
            category="fact",
            source="hermes:delegation",
        )

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "shared_db",
                "description": "Path to Zouroboros shared-facts.db",
                "default": "/home/workspace/.zo/memory/shared-facts.db",
                "required": True,
                "env_var": "ZO_MEMORY_SHARED_DB",
            },
            {
                "key": "mimir_db",
                "description": "Path to Mimir synthesis database (mimir.db)",
                "default": "/home/workspace/.zo/memory/mimir.db",
                "env_var": "ZO_MEMORY_MIMIR_DB",
            },
            {
                "key": "cli_path",
                "description": "Path to zo-memory CLI (memory.ts)",
                "default": "/home/workspace/Skills/zo-memory-system/scripts/memory.ts",
                "env_var": "ZO_MEMORY_CLI",
            },
            {
                "key": "persona",
                "description": "Persona name for memory routing",
                "default": "hermes",
                "env_var": "ZO_MEMORY_PERSONA",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        from pathlib import Path
        cfg_dir = Path(hermes_home) / "zouroboros"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / "config.yaml"
        try:
            import yaml
            with open(cfg_path, "w") as f:
                yaml.safe_dump(values, f, default_flow_style=False)
            logger.info("Saved Zouroboros config to %s", cfg_path)
        except ImportError:
            with open(cfg_path, "w") as f:
                for k, v in values.items():
                    f.write(f"{k}: {v}\n")

    def shutdown(self) -> None:
        for f in self._pending_writes:
            if not f.done():
                try:
                    f.result(timeout=1.0)
                except Exception:
                    pass
        if self._bridge:
            self._bridge.shutdown()
        logger.info("Zouroboros memory bridge shut down")

    @staticmethod
    def _format_results(results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""
        lines = ["[Zouroboros Memory — cross-persona context]"]
        for r in results[:6]:
            decay = r.get("decayClass", "?")
            entity = r.get("entity", "?")
            key = r.get("key", "")
            value = r.get("value", "")[:150]
            score = r.get("score", 0)
            lines.append(f"  [{decay}] {entity}.{key} = {value} (score: {score:.2f})")
        return "\n".join(lines)
