#!/usr/bin/env python3
"""Zo-to-Hermes bridge — accepts a query via CLI arg, returns Hermes's response.

Usage:
    python scripts/zo-bridge.py "your message here"

Designed to be invoked by a Zo Persona via run_bash_command.
Reads model configuration from ~/.hermes/config.yaml and ~/.hermes/.env.
"""

import os
import sys

# Ensure hermes-agent root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from dotenv import load_dotenv
import yaml

# Load env vars from ~/.hermes/.env
load_dotenv(dotenv_path=Path.home() / ".hermes" / ".env", override=True)

# Read config to determine provider
_config_path = Path.home() / ".hermes" / "config.yaml"
_model_name = None
_base_url = None
_api_key = None

if _config_path.exists():
    with open(_config_path) as f:
        _cfg = yaml.safe_load(f) or {}
    # Model can be a string or dict
    _model_cfg = _cfg.get("model", {})
    if isinstance(_model_cfg, str):
        _model_name = _model_cfg
    elif isinstance(_model_cfg, dict):
        _model_name = _model_cfg.get("default")
        _base_url = _model_cfg.get("base_url")

# Resolve base_url and api_key from env if not in config
if not _base_url:
    _base_url = os.getenv("OPENAI_BASE_URL")

if _base_url and "openrouter" not in _base_url.lower():
    # Custom provider — use OPENAI_API_KEY
    _api_key = os.getenv("OPENAI_API_KEY", "")
else:
    # OpenRouter — use OPENROUTER_API_KEY, prevent OPENAI_API_KEY conflict
    os.environ.pop("OPENAI_API_KEY", None)
    _api_key = os.getenv("OPENROUTER_API_KEY", "")
    _base_url = None  # Let AIAgent default to OpenRouter

from run_agent import AIAgent


def main():
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("Usage: python scripts/zo-bridge.py \"your message\"", file=sys.stderr)
        sys.exit(1)

    query = sys.argv[1]

    kwargs = dict(
        quiet_mode=True,
        skip_context_files=True,
        max_iterations=30,
        max_tokens=1056,
    )
    if _model_name:
        kwargs["model"] = _model_name
    if _base_url:
        kwargs["base_url"] = _base_url
    if _api_key:
        kwargs["api_key"] = _api_key

    agent = AIAgent(**kwargs)

    response = agent.chat(query)
    print(response)


if __name__ == "__main__":
    main()
