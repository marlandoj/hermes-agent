"""Zouroboros Read-Only Telegram Command Handlers.

Provides system monitoring commands for the Zouroboros ecosystem via Telegram.
All commands are pure reads — no mutations, no state changes.

Commands:
    /status         — System health (CPU, memory, disk, services)
    /metrics        — Key ecosystem metrics (memory facts, decay stats, services)
    /logs <name>    — Last 20 lines of a service log
    /memory_stats   — Full zo-memory health report
    /swarm_status   — Active swarm info, sentinel events, recent results
    /autoloop_status — Active autoloop sessions and trajectory data

Usage:
    Register with Hermes gateway by hooking into the message handler pipeline.
    Commands are intercepted before reaching the main agent loop.
"""

import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

WORKSPACE = os.environ.get("HOME", "/home/workspace")
if not os.path.isdir(os.path.join(WORKSPACE, ".zo")):
    WORKSPACE = "/home/workspace"

LOG_DIR = "/dev/shm"
SWARM_DIR = os.path.join(os.environ.get("HOME", "/root"), ".swarm")
SENTINEL_DIR = os.environ.get("SWARM_SENTINEL_DIR", "/tmp/swarm-events")
MEMORY_CLI = os.path.join(WORKSPACE, "Skills", "zo-memory-system", "scripts", "memory.ts")


COMMANDS = {
    "status": "System health — CPU, memory, disk, services",
    "metrics": "Key ecosystem metrics",
    "logs": "Recent logs: /logs <service>",
    "memory_stats": "Memory system health report",
    "swarm_status": "Active swarm and recent results",
    "autoloop_status": "Active autoloop sessions",
}

HELP_TEXT = "*Zouroboros Monitor*\n\n" + "\n".join(
    f"/{cmd} — {desc}" for cmd, desc in COMMANDS.items()
)


def _run(cmd: str, timeout: float = 5.0) -> str:
    """Run a shell command and return stdout (or error text)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip() or result.stderr.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "(timed out)"
    except Exception as e:
        return f"(error: {e})"


def _truncate(text: str, max_len: int = 3800) -> str:
    """Truncate to fit Telegram's 4096 char limit with room for headers."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... (truncated)"


async def cmd_status() -> str:
    """System health: CPU, memory, disk, services."""
    loop = asyncio.get_event_loop()

    cpu = await loop.run_in_executor(None, _run, "uptime | sed 's/.*load average: /Load: /'")
    mem = await loop.run_in_executor(
        None, _run,
        "free -h | awk '/Mem:/{printf \"Used: %s / %s (%.0f%%)\", $3, $2, $3/$2*100}'"
    )
    disk = await loop.run_in_executor(
        None, _run,
        "df -h / | awk 'NR==2{printf \"Used: %s / %s (%s)\", $3, $2, $5}'"
    )

    # Service health
    svc_lines = []
    log_files = sorted(Path(LOG_DIR).glob("*.log"))
    active_services = set()
    for lf in log_files:
        name = lf.stem
        if name.startswith("zosite-") or name.endswith("_err"):
            continue
        active_services.add(name)
        size = lf.stat().st_size
        mtime = datetime.fromtimestamp(lf.stat().st_mtime, tz=timezone.utc)
        age_min = (datetime.now(timezone.utc) - mtime).total_seconds() / 60
        status = "🟢" if age_min < 10 else "🟡" if age_min < 60 else "🔴"
        svc_lines.append(f"  {status} {name} ({size // 1024}KB, {age_min:.0f}m ago)")

    services = "\n".join(svc_lines[:15]) if svc_lines else "  (no service logs found)"

    return (
        f"*System Status*\n\n"
        f"⚡ CPU: {cpu}\n"
        f"💾 Memory: {mem}\n"
        f"📀 Disk: {disk}\n"
        f"\n*Services* ({len(active_services)}):\n{services}"
    )


async def cmd_metrics() -> str:
    """Key ecosystem metrics: memory facts, decay stats, active services."""
    loop = asyncio.get_event_loop()

    # Memory stats
    mem_stats = await loop.run_in_executor(
        None, _run, f"bun {MEMORY_CLI} stats 2>/dev/null || echo '(memory CLI unavailable)'"
    )

    # Service count
    log_count = len(list(Path(LOG_DIR).glob("*.log")))

    # Sentinel events
    sentinel_count = 0
    if os.path.isdir(SENTINEL_DIR):
        sentinel_count = len([f for f in os.listdir(SENTINEL_DIR) if f.endswith(".json")])

    return (
        f"*Ecosystem Metrics*\n\n"
        f"📡 Pending sentinels: {sentinel_count}\n"
        f"📋 Service logs: {log_count}\n\n"
        f"*Memory System:*\n```\n{_truncate(mem_stats, 2500)}\n```"
    )


async def cmd_logs(service_name: str) -> str:
    """Last 20 lines of a service log."""
    if not service_name:
        # List available logs
        logs = sorted(Path(LOG_DIR).glob("*.log"))
        names = [lf.stem for lf in logs if not lf.stem.endswith("_err")][:20]
        return "*Available logs:*\n" + "\n".join(f"  • {n}" for n in names) if names else "No logs found."

    # Sanitize input
    safe_name = "".join(c for c in service_name if c.isalnum() or c in "-_.")
    log_path = os.path.join(LOG_DIR, f"{safe_name}.log")

    if not os.path.isfile(log_path):
        # Try stderr log
        err_path = os.path.join(LOG_DIR, f"{safe_name}_err.log")
        if os.path.isfile(err_path):
            log_path = err_path
        else:
            return f"Log not found: `{safe_name}`\nUse /logs to list available."

    loop = asyncio.get_event_loop()
    lines = await loop.run_in_executor(None, _run, f"tail -20 '{log_path}'")

    return f"*Logs: {safe_name}*\n```\n{_truncate(lines, 3500)}\n```"


async def cmd_memory_stats() -> str:
    """Full memory system health report."""
    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(
        None, _run, f"bun {MEMORY_CLI} stats 2>/dev/null", 10.0
    )
    return f"*Memory System Health*\n```\n{_truncate(stats, 3500)}\n```"


async def cmd_swarm_status() -> str:
    """Active swarm info, recent results, sentinel events."""
    parts = ["*Swarm Status*\n"]

    # Check for active swarm lock
    lock_files = list(Path("/dev/shm").glob("*.lock"))
    if lock_files:
        for lf in lock_files[:5]:
            try:
                lock = json.loads(lf.read_text())
                name = lf.stem
                age_s = (time.time() - lock.get("ts", 0)) / 60
                parts.append(f"🔄 Active: `{name}` (PID {lock.get('pid', '?')}, {age_s:.0f}m)")
            except Exception:
                pass
    else:
        parts.append("No active swarms.")

    # Progress files
    progress_files = list(Path("/dev/shm").glob("*-progress.json"))
    for pf in progress_files[:3]:
        try:
            prog = json.loads(pf.read_text())
            c = prog.get("completed", 0)
            f = prog.get("failed", 0)
            t = prog.get("total", 0)
            parts.append(f"\n📊 Progress: {c}/{t} done, {f} failed")
        except Exception:
            pass

    # Pending sentinels
    sentinel_count = 0
    if os.path.isdir(SENTINEL_DIR):
        sentinel_count = len([f for f in os.listdir(SENTINEL_DIR) if f.endswith(".json")])
    parts.append(f"\n📡 Pending sentinel events: {sentinel_count}")

    # Recent results
    results_dir = os.path.join(SWARM_DIR, "results")
    if os.path.isdir(results_dir):
        result_files = sorted(Path(results_dir).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:3]
        if result_files:
            parts.append("\n*Recent Results:*")
            for rf in result_files:
                try:
                    data = json.loads(rf.read_text())
                    status = data.get("status", "?")
                    completed = data.get("completed", 0)
                    failed = data.get("failed", 0)
                    elapsed = data.get("elapsedMs", 0)
                    emoji = "✅" if status == "complete" else "❌"
                    parts.append(f"  {emoji} {rf.stem}: {completed} ok, {failed} fail ({elapsed // 1000}s)")
                except Exception:
                    pass

    return "\n".join(parts)


async def cmd_autoloop_status() -> str:
    """Active autoloop sessions and trajectory data."""
    parts = ["*Autoloop Status*\n"]

    # Check for running autoloop processes
    loop = asyncio.get_event_loop()
    ps_output = await loop.run_in_executor(
        None, _run, "ps aux | grep autoloop | grep -v grep | head -5"
    )
    if ps_output and "autoloop" in ps_output:
        parts.append("🔄 *Active Processes:*")
        for line in ps_output.splitlines()[:3]:
            parts.append(f"  `{line.split()[-1] if line.split() else line[:60]}`")
    else:
        parts.append("No active autoloop sessions.")

    # Trajectory DBs
    traj_dir = os.path.join(WORKSPACE, "Skills", "autoloop", "trajectories")
    if os.path.isdir(traj_dir):
        dbs = list(Path(traj_dir).glob("*.db"))
        if dbs:
            parts.append(f"\n📈 *Trajectory DBs:* {len(dbs)}")
            for db in sorted(dbs, key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
                size_kb = db.stat().st_size // 1024
                mtime = datetime.fromtimestamp(db.stat().st_mtime, tz=timezone.utc)
                parts.append(f"  • {db.stem} ({size_kb}KB, last: {mtime.strftime('%m/%d %H:%M')})")

    return "\n".join(parts)


COMMAND_HANDLERS = {
    "status": lambda args: cmd_status(),
    "metrics": lambda args: cmd_metrics(),
    "logs": lambda args: cmd_logs(args),
    "memory_stats": lambda args: cmd_memory_stats(),
    "swarm_status": lambda args: cmd_swarm_status(),
    "autoloop_status": lambda args: cmd_autoloop_status(),
}


async def handle_command(command: str, args: str = "") -> Optional[str]:
    """Route a command to its handler. Returns response text or None if unknown."""
    handler = COMMAND_HANDLERS.get(command)
    if handler:
        try:
            return await handler(args.strip())
        except Exception as e:
            return f"⚠️ Error executing /{command}: {e}"
    return None


def is_zouroboros_command(command: str) -> bool:
    """Check if a command is a Zouroboros read-only command."""
    return command in COMMAND_HANDLERS


def get_command_menu() -> list:
    """Return (name, description) tuples for Telegram bot menu."""
    return [(cmd, desc) for cmd, desc in COMMANDS.items()]
