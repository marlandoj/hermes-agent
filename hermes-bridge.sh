#!/usr/bin/env bash
# Hermes bridge script — invokes Hermes CLI in single-query mode
# Routes through OmniRoute for access to swarm combos, Anthropic Claude models, and shared memory
#
# Usage:
#   ./hermes-bridge.sh "Your prompt here"
#
# Environment:
#   HERMES_MODEL — override model combo (default: swarm-mid)
#     Available: swarm-light, swarm-mid, swarm-heavy, swarm-failover
#     Or use individual models: claude-opus-4-6, gemini-2.5-flash, gpt-5.3-codex, etc.
#   HERMES_TIMEOUT — timeout in seconds (default: 300)

cd /home/workspace/hermes-agent
unset OPENAI_API_KEY
export HERMES_MODEL="${HERMES_MODEL:-swarm-mid}"
export HERMES_TIMEOUT="${HERMES_TIMEOUT:-300}"
python3 << 'EOF'
import sys
t = sys.stdin.read()
q = t.find("Query:\n")
r = t.find("Resume:\n")
if q >= 0 and r >= 0:
    print(t[q+7:r].strip())
else:
    print(t.strip())
EOF
