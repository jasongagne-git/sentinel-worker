#!/usr/bin/env python3

# Copyright 2026 Jason Gagne
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SENTINEL Worker Monitor — local dashboard for worker operators.

Polls the local worker's /v1/dashboard endpoint and displays a
live-updating terminal view of agent activity, thermal state,
and throughput. No access to the coordinator DB required.

Usage:
    python3 run_worker_monitor.py
    python3 run_worker_monitor.py --port 9400 --refresh 5
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── Resolve sibling repos ───────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PROJECTS = _HERE.parent
_common = _PROJECTS / "sentinel-common"
if _common.is_dir() and str(_common) not in sys.path:
    sys.path.insert(0, str(_common))

# ── Auto-load .env if present ───────────────────────────────────────────────
_env_file = _HERE / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            line = line.removeprefix("export ").strip()
            if "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def fetch_dashboard(host: str, port: int, secret: str) -> dict:
    """Fetch dashboard data from the local worker."""
    from sentinel_common.auth import sign_request
    auth = sign_request(b"", secret)
    url = f"http://{host}:{port}/v1/dashboard"
    req = urllib.request.Request(url, method="GET")
    req.add_header("X-Sentinel-Auth", auth)
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        return json.loads(resp.read().decode())
    except (urllib.error.URLError, ConnectionError, OSError) as exc:
        return {"error": str(exc)}


def format_uptime(seconds: int) -> str:
    """Format seconds into human-readable uptime."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h}h {m}m"


def render(data: dict):
    """Render dashboard data to terminal."""
    # Clear screen
    print("\033[2J\033[H", end="")

    if "error" in data:
        print("=" * 60)
        print(" SENTINEL Worker Monitor")
        print("=" * 60)
        print()
        print(f"  Worker unreachable: {data['error']}")
        print()
        print("  Waiting for worker to start...")
        return

    node_id = data.get("node_id", "?")
    status = data.get("status", "?")
    uptime = format_uptime(data.get("uptime_s", 0))
    model = data.get("loaded_model") or "--"
    total_inf = data.get("total_inferences", 0)
    total_probes = data.get("total_probes", 0)
    active = data.get("active_inference", False)
    active_agent = data.get("active_agent") or ""
    experiment_id = data.get("experiment_id") or "--"

    thermal = data.get("thermal", {})
    temp = thermal.get("current_temp_c")
    temp_str = f"{temp:.1f}C" if temp is not None else "--"
    pause_count = thermal.get("pause_count", 0)
    max_temp = thermal.get("max_temp_c", 0)

    agents = data.get("agents", [])

    print("=" * 60)
    print(" SENTINEL Worker Monitor")
    print("=" * 60)
    print()
    max_turns = data.get("max_turns")
    current_turn = max((a.get("last_turn", 0) for a in agents), default=0) if agents else 0
    if max_turns:
        turn_str = f"{current_turn}/{max_turns}"
        pct = 100 * current_turn // max_turns if max_turns > 0 else 0
        bar_w = 30
        filled = bar_w * current_turn // max_turns if max_turns > 0 else 0
        progress_bar = f"[{'#' * filled}{'-' * (bar_w - filled)}] {pct}%"
    else:
        turn_str = str(current_turn)
        progress_bar = ""

    print(f"  Node:       {node_id}")
    print(f"  Status:     {status.upper()}")
    print(f"  Uptime:     {uptime}")
    print(f"  Model:      {model}")
    print(f"  Experiment: {experiment_id[:8] if experiment_id != '--' else '--'}")
    print(f"  Turn:       {turn_str}")
    if progress_bar:
        print(f"  Progress:   {progress_bar}")
    print()
    print(f"  Temperature:  {temp_str}  (max: {max_temp:.1f}C, pauses: {pause_count})")
    print(f"  Inferences:   {total_inf}  (probes: {total_probes})")
    if active:
        print(f"  Active:       {active_agent} (inferring...)")
    else:
        print(f"  Active:       idle")
    print()

    if agents:
        print("-" * 60)
        print(f" {'Agent':20s} {'Inferences':>11s} {'Avg ms':>8s} {'Last Turn':>10s}")
        print("-" * 60)
        for a in sorted(agents, key=lambda x: x.get("name", "")):
            name = a.get("name", "?")
            inf = a.get("inferences", 0)
            avg = a.get("avg_ms", 0)
            turn = a.get("last_turn", 0)
            print(f" {name:20s} {inf:11d} {avg:8d} {turn:10d}")
        print("-" * 60)
        print(f" Total: {len(agents)} agent(s)")
    else:
        print("  No agents active yet")

    print()
    print("  Press Ctrl+C to exit")


def main():
    parser = argparse.ArgumentParser(description="SENTINEL Worker Monitor")
    parser.add_argument("--host", default="localhost", help="Worker host (default: localhost)")
    parser.add_argument("--port", type=int, default=9400, help="Worker port (default: 9400)")
    parser.add_argument("--refresh", type=float, default=3.0, help="Refresh interval in seconds")
    args = parser.parse_args()

    secret = os.environ.get("SENTINEL_CLUSTER_SECRET", "")
    if not secret:
        # Try loading from .env
        env_path = _HERE / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "SENTINEL_CLUSTER_SECRET" in line:
                    secret = line.split("=", 1)[1].strip()
                    break
    if not secret:
        print("Error: SENTINEL_CLUSTER_SECRET not set. Source .env or export it.")
        sys.exit(1)

    print(f"Connecting to worker at {args.host}:{args.port}...")

    try:
        while True:
            data = fetch_dashboard(args.host, args.port, secret)
            render(data)
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")


if __name__ == "__main__":
    main()
