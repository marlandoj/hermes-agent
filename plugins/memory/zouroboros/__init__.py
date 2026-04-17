"""Zouroboros memory provider plugin for Hermes Agent.

Bridges Hermes to the Zouroboros zo-memory ecosystem with dual-write:
  - Fast local recall via Hindsight (handled by Hindsight provider)
  - Cross-persona visibility via async zo-memory CLI bridge

Config via environment variables:
  ZO_MEMORY_SHARED_DB  — path to shared-facts.db
  ZO_MEMORY_MIMIR_DB   — path to mimir.db
  ZO_MEMORY_CLI        — path to zo-memory CLI (memory.ts)
  ZO_MEMORY_PERSONA    — persona name for routing (default: hermes)

Or via $HERMES_HOME/zouroboros/config.yaml (profile-scoped).
"""

from .provider import ZouroborosMemoryProvider


def register(ctx) -> None:
    """Register Zouroboros as a memory provider plugin."""
    ctx.register_memory_provider(ZouroborosMemoryProvider())
