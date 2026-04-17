#!/usr/bin/env bash
# Wrapper to launch Hermes with the correct API key precedence.
# Unsets OPENAI_API_KEY so OPENROUTER_API_KEY is used instead.
DIR="$(cd "$(dirname "$0")" && pwd)"
exec env OPENAI_API_KEY= "$DIR/.venv/bin/hermes" "$@"
