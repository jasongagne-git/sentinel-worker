#!/usr/bin/env bash

# SENTINEL Worker — Launch Script
#
# Starts the worker service. If first-time setup hasn't been completed,
# walks through the install steps first.
#
# Usage:
#   ./launch.sh                     # auto-detect node ID from cluster.json
#   ./launch.sh --node-id mayhem1   # explicit node ID
#   ./launch.sh --log-level DEBUG   # pass-through args to run_worker.py

set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$(dirname "$SCRIPT_DIR")"
COMMON_DIR="$INSTALL_DIR/sentinel-common"

# ── Parse args (extract --node-id, pass rest through) ───────────────────────
NODE_ID=""
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --node-id) NODE_ID="$2"; shift 2;;
        *)         PASS_ARGS+=("$1"); shift;;
    esac
done

# ── Pre-flight checks ──────────────────────────────────────────────────────
echo -e "\n${BOLD}SENTINEL Worker Launch${NC}\n"

NEEDS_INSTALL=false

# Check sentinel-common exists
if [ ! -d "$COMMON_DIR/sentinel_common" ]; then
    warn "sentinel-common not found at $COMMON_DIR"
    NEEDS_INSTALL=true
fi

# Check .env
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    warn "No .env file — shared secret not configured"
    NEEDS_INSTALL=true
fi

# Check cluster.json
if [ ! -f "$SCRIPT_DIR/cluster.json" ]; then
    warn "No cluster.json — cluster not configured"
    NEEDS_INSTALL=true
fi

# Check Ollama
if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    warn "Ollama not running"
    if command -v ollama &>/dev/null; then
        info "Starting Ollama..."
        sudo systemctl start ollama 2>/dev/null || true
        sleep 2
        if curl -sf http://localhost:11434/api/tags &>/dev/null; then
            ok "Ollama started"
        else
            warn "Could not start Ollama"
            NEEDS_INSTALL=true
        fi
    else
        warn "Ollama not installed"
        NEEDS_INSTALL=true
    fi
fi

# If anything is missing, run install
if $NEEDS_INSTALL; then
    echo ""
    warn "First-time setup required. Running install..."
    echo ""
    if [ -f "$SCRIPT_DIR/install.sh" ]; then
        bash "$SCRIPT_DIR/install.sh"
        echo ""
        info "Install complete. Continuing to launch..."
        echo ""
    else
        fail "install.sh not found in $SCRIPT_DIR"
    fi
fi

# ── Load environment ───────────────────────────────────────────────────────
source "$SCRIPT_DIR/.env"

# ── Auto-detect node ID from cluster.json if not specified ─────────────────
if [ -z "$NODE_ID" ]; then
    NODE_ID=$(python3 -c "
import json
with open('$SCRIPT_DIR/cluster.json') as f:
    config = json.load(f)
# Use first worker entry, or coordinator if role=both
workers = config.get('workers', [])
if workers:
    print(workers[0]['node_id'])
elif config.get('coordinator', {}).get('role') == 'both':
    print(config['coordinator']['node_id'])
" 2>/dev/null)

    if [ -z "$NODE_ID" ]; then
        fail "Could not auto-detect node ID from cluster.json. Use --node-id."
    fi
    info "Auto-detected node ID: $NODE_ID"
fi

# ── Launch ─────────────────────────────────────────────────────────────────
ok "Starting worker '$NODE_ID'..."
echo ""
exec python3 "$SCRIPT_DIR/run_worker.py" \
    --config "$SCRIPT_DIR/cluster.json" \
    --node-id "$NODE_ID" \
    "${PASS_ARGS[@]}"
