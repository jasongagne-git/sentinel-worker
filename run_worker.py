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

"""Start a SENTINEL worker service on this node.

Expects sibling directory:
  ../sentinel-common/   — shared distributed modules (public)

Usage:
    python3 run_worker.py --config ../sentinel-common/config/cluster.json --node-id mayhem1
    python3 run_worker.py --config cluster.json --node-id mayhem1 --agents my_agents.json
"""

import argparse
import logging
import signal
import sys
from pathlib import Path

# ── Resolve sibling repos ───────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PROJECTS = _HERE.parent
_common = _PROJECTS / "sentinel-common"
if _common.is_dir() and str(_common) not in sys.path:
    sys.path.insert(0, str(_common))

log = logging.getLogger("sentinel.worker")


def main():
    parser = argparse.ArgumentParser(
        description="SENTINEL Worker Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to cluster config JSON file")
    parser.add_argument("--node-id", required=True, help="This node's ID in the cluster config")
    parser.add_argument("--port", type=int, default=None, help="Override port from config")
    parser.add_argument("--bind", default=None, help="Override bind interface from config")
    parser.add_argument("--agents", default=None, help="(Open mode) Path to agent definitions JSON")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Load cluster config ─────────────────────────────────────────────
    from sentinel_common.cluster import load_cluster_config, validate_cluster_config

    try:
        cluster = load_cluster_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        log.error("Failed to load cluster config: %s", exc)
        sys.exit(1)

    errors = validate_cluster_config(cluster)
    if errors:
        for err in errors:
            log.error("Config error: %s", err)
        sys.exit(1)

    # ── Validate archetypes ─────────────────────────────────────────────
    from sentinel_common.validate import validate_archetypes

    arch_errors = validate_archetypes()
    if arch_errors:
        for err in arch_errors:
            log.error("Archetype error: %s", err)
        sys.exit(1)

    # ── Find this node's config ─────────────────────────────────────────
    node = cluster.find_node(args.node_id)
    if node is None:
        log.error(
            "Node '%s' not found in cluster config. Available: %s",
            args.node_id, ", ".join(n.node_id for n in cluster.all_nodes()),
        )
        sys.exit(1)

    if args.port is not None:
        node.port = args.port
    if args.bind is not None:
        node.bind_interface = args.bind

    # ── TLS context ─────────────────────────────────────────────────────
    tls_context = None
    if cluster.tls_enabled:
        from sentinel_common.auth import create_tls_context
        tls_context = create_tls_context(
            cert_path=node.tls_cert, key_path=node.tls_key,
            ca_cert_path=cluster.tls_ca_cert, server_side=True,
        )
        if tls_context is None:
            log.error("TLS enabled but no cert for node %s", node.node_id)
            sys.exit(1)

    # ── Pre-flight: Ollama ──────────────────────────────────────────────
    from sentinel_common.ollama import OllamaClient

    client = OllamaClient(node.ollama_url)
    if not client.is_available():
        log.error("Ollama not available at %s", node.ollama_url)
        sys.exit(1)

    models = [m["name"] for m in client.list_models()]
    log.info("Ollama OK — %d models: %s", len(models), ", ".join(models))

    # ── Handle --agents for open mode ──────────────────────────────────
    if args.agents:
        if cluster.mode != "open":
            log.warning("--agents ignored (cluster mode is '%s', not 'open')", cluster.mode)
        else:
            import json as _json
            from sentinel_common.auth import sign_request
            import urllib.request
            import urllib.error

            agents_path = Path(args.agents)
            if not agents_path.exists():
                log.error("Agents file not found: %s", agents_path)
                sys.exit(1)

            with agents_path.open() as f:
                agent_defs = _json.load(f).get("agents", [])

            if agent_defs:
                coord = cluster.coordinator
                body = _json.dumps({"node_id": node.node_id, "agents": agent_defs}).encode()
                req = urllib.request.Request(f"{coord.base_url}/v1/register_agents", method="POST")
                req.add_header("Content-Type", "application/json")
                req.add_header("X-Sentinel-Auth", sign_request(body, cluster.shared_secret))
                req.data = body
                try:
                    resp = urllib.request.urlopen(req, timeout=30)
                    result = _json.loads(resp.read().decode())
                    log.info("Registered %d agents (%d rejected)", result.get("accepted", 0), result.get("rejected", 0))
                except urllib.error.URLError as exc:
                    log.warning("Could not register agents with coordinator: %s", exc)

    # ── Start worker service ────────────────────────────────────────────
    from sentinel_worker.worker import WorkerService

    service = WorkerService(node_config=node, shared_secret=cluster.shared_secret, tls_context=tls_context)

    signal.signal(signal.SIGTERM, lambda *_: (log.info("SIGTERM received"), service.stop()))

    log.info("Starting worker '%s' on %s:%d (cluster: %s)", node.node_id, node.bind_interface, node.port, cluster.cluster_id)
    service.start()


if __name__ == "__main__":
    main()
