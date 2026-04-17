"""Configuration for the Zouroboros memory bridge.

Resolution order:
  1. Environment variables (highest priority)
  2. $HERMES_HOME/zouroboros/config.yaml (profile-scoped)
  3. Hard-coded defaults (lowest priority)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_SHARED_DB = "/home/workspace/.zo/memory/shared-facts.db"
_DEFAULT_MIMIR_DB = "/home/workspace/.zo/memory/mimir.db"
_DEFAULT_CLI = "/home/workspace/Skills/zo-memory-system/scripts/memory.ts"
_DEFAULT_PERSONA = "hermes"
_DEFAULT_BRIDGE_TIMEOUT_MS = 500
_DEFAULT_PREFETCH_LIMIT = 6


@dataclass
class ZouroborosConfig:
    shared_db: str = _DEFAULT_SHARED_DB
    mimir_db: str = _DEFAULT_MIMIR_DB
    cli_path: str = _DEFAULT_CLI
    persona: str = _DEFAULT_PERSONA
    bridge_timeout_ms: int = _DEFAULT_BRIDGE_TIMEOUT_MS
    prefetch_limit: int = _DEFAULT_PREFETCH_LIMIT
    enable_mimir_synthesis: bool = True
    enable_wikilinks: bool = True
    async_writes: bool = True


def load_config(hermes_home: str = "") -> ZouroborosConfig:
    """Load config with env-var overrides."""
    cfg = ZouroborosConfig()

    if hermes_home:
        cfg_path = Path(hermes_home) / "zouroboros" / "config.yaml"
        if cfg_path.exists():
            try:
                import yaml
                with open(cfg_path) as f:
                    data = yaml.safe_load(f) or {}
                for k, v in data.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
            except Exception as e:
                logger.warning("Failed to load %s: %s", cfg_path, e)

    env_map = {
        "ZO_MEMORY_SHARED_DB": "shared_db",
        "ZO_MEMORY_MIMIR_DB": "mimir_db",
        "ZO_MEMORY_CLI": "cli_path",
        "ZO_MEMORY_PERSONA": "persona",
        "ZO_MEMORY_BRIDGE_TIMEOUT_MS": "bridge_timeout_ms",
        "ZO_MEMORY_PREFETCH_LIMIT": "prefetch_limit",
    }
    for env_key, attr in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            current = getattr(cfg, attr)
            if isinstance(current, int):
                try:
                    val = int(val)
                except ValueError:
                    continue
            elif isinstance(current, bool):
                val = val.lower() in ("1", "true", "yes")
            setattr(cfg, attr, val)

    return cfg
