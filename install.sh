#!/usr/bin/env bash

# SENTINEL Worker — Installation Script
#
# Sets up a worker node for distributed SENTINEL experiments.
# Checks all dependencies, installs Ollama if missing (with Jetson CUDA support),
# clones required repos, and generates configuration.
#
# Usage:
#   curl -sL <url>/install.sh | bash
#   # or
#   git clone ... && cd sentinel-worker && ./install.sh
#
# Environment variables (optional):
#   SENTINEL_INSTALL_DIR  — parent directory for repos (default: ~/Projects)
#   SENTINEL_SECRET       — shared cluster secret (will prompt if not set)
#   SENTINEL_INVITE       — invite token from coordinator (alternative to SECRET)
#   SENTINEL_COORDINATOR  — coordinator address (will prompt if not set)

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
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }
step()  { echo -e "\n${BOLD}=== $* ===${NC}"; }
ask()   { echo -en "${BOLD}$*${NC} "; }

INSTALL_DIR="${SENTINEL_INSTALL_DIR:-$HOME/Projects}"
ERRORS=0

# ── Step 1: System checks ──────────────────────────────────────────────────
step "System Checks"

# OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    info "OS: $PRETTY_NAME"
else
    warn "Cannot detect OS"
fi

# Architecture
ARCH=$(uname -m)
info "Architecture: $ARCH"

# Jetson detection
IS_JETSON=false
if [ -f /proc/device-tree/model ]; then
    MODEL=$(tr -d '\0' < /proc/device-tree/model)
    info "Hardware: $MODEL"
    if echo "$MODEL" | grep -qi "jetson"; then
        IS_JETSON=true
        ok "NVIDIA Jetson detected"
    fi
fi

# Disk space
AVAIL_GB=$(df -BG "$HOME" | awk 'NR==2 {print $4}' | tr -d 'G')
if [ "$AVAIL_GB" -lt 5 ]; then
    fail "Only ${AVAIL_GB}GB free disk space (need at least 5GB)"
    ERRORS=$((ERRORS + 1))
else
    ok "Disk space: ${AVAIL_GB}GB available"
fi

# RAM
TOTAL_RAM_MB=$(grep MemTotal /proc/meminfo | awk '{print int($2/1024)}')
info "Total RAM: ${TOTAL_RAM_MB}MB"
if [ "$TOTAL_RAM_MB" -lt 4000 ]; then
    warn "Less than 4GB RAM — small models only"
fi

# ── Step 2: Python ──────────────────────────────────────────────────────────
step "Python"

if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
        ok "Python $PY_VERSION"
    else
        fail "Python $PY_VERSION found but 3.10+ required"
        info "Install with: sudo apt install python3.10 (or newer)"
        ERRORS=$((ERRORS + 1))
    fi
else
    fail "Python 3 not found"
    info "Install with: sudo apt install python3"
    ERRORS=$((ERRORS + 1))
fi

# ── Step 3: Git ─────────────────────────────────────────────────────────────
step "Git"

if command -v git &>/dev/null; then
    ok "Git $(git --version | awk '{print $3}')"
else
    fail "Git not found"
    info "Install with: sudo apt install git"
    ERRORS=$((ERRORS + 1))
fi

# ── Step 4: CUDA / GPU ─────────────────────────────────────────────────────
step "GPU / CUDA"

HAS_CUDA=false
if $IS_JETSON; then
    # Jetson uses unified memory — check JetPack
    if [ -f /etc/nv_tegra_release ]; then
        info "Tegra release: $(head -1 /etc/nv_tegra_release)"
    fi
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ',')
        ok "CUDA $CUDA_VER (JetPack)"
        HAS_CUDA=true
    elif [ -d /usr/local/cuda ]; then
        ok "CUDA found at /usr/local/cuda"
        HAS_CUDA=true
    else
        warn "CUDA not found — Ollama will run on CPU"
    fi
elif command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU: $GPU_NAME"
    HAS_CUDA=true
else
    warn "No NVIDIA GPU detected — Ollama will run on CPU"
fi

# ── Step 5: Ollama ──────────────────────────────────────────────────────────
step "Ollama"

OLLAMA_INSTALLED=false
OLLAMA_RUNNING=false

if command -v ollama &>/dev/null; then
    OLLAMA_VER=$(ollama --version 2>/dev/null | head -1 || echo "unknown")
    ok "Ollama installed: $OLLAMA_VER"
    OLLAMA_INSTALLED=true

    # Check if running
    if curl -sf http://localhost:11434/api/tags &>/dev/null; then
        ok "Ollama is running"
        OLLAMA_RUNNING=true

        # List models
        MODELS=$(curl -sf http://localhost:11434/api/tags | python3 -c "
import json, sys
data = json.load(sys.stdin)
for m in data.get('models', []):
    size_gb = m.get('size', 0) / 1e9
    print(f\"  {m['name']:30s} ({size_gb:.1f}GB)\")
" 2>/dev/null || echo "  (could not list)")
        info "Available models:"
        echo "$MODELS"
    else
        warn "Ollama is installed but not running"
        ask "Start Ollama now? [Y/n]"
        read -r REPLY
        if [[ -z "$REPLY" || "$REPLY" =~ ^[Yy] ]]; then
            sudo systemctl start ollama
            sleep 2
            if curl -sf http://localhost:11434/api/tags &>/dev/null; then
                ok "Ollama started"
                OLLAMA_RUNNING=true
            else
                fail "Could not start Ollama"
            fi
        fi
    fi
else
    warn "Ollama is not installed"
    ask "Install Ollama now? [Y/n]"
    read -r REPLY
    if [[ -z "$REPLY" || "$REPLY" =~ ^[Yy] ]]; then
        echo ""
        info "Downloading Ollama installer..."
        curl -fsSL https://ollama.com/install.sh | sh

        if command -v ollama &>/dev/null; then
            ok "Ollama installed"
            OLLAMA_INSTALLED=true

            # ── Jetson-specific CUDA configuration ──
            if $IS_JETSON; then
                step "Jetson Ollama Configuration"
                info "Applying JetPack-specific CUDA settings..."

                # Create systemd override
                sudo mkdir -p /etc/systemd/system/ollama.service.d

                # Detect CUDA lib paths
                CUDA_LIB="/usr/local/cuda/lib64"
                TEGRA_LIB="/usr/lib/aarch64-linux-gnu/tegra"

                cat <<OVERRIDE | sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null
[Service]
# SENTINEL: Jetson CUDA configuration
Environment="LD_LIBRARY_PATH=${CUDA_LIB}:${TEGRA_LIB}"
Environment="GGML_CUDA_NO_VMM=1"
OVERRIDE

                # Optional: custom model storage location
                ask "Store Ollama models in custom location? (default: /usr/share/ollama/.ollama) [path/N]"
                read -r MODEL_PATH
                if [[ -n "$MODEL_PATH" && "$MODEL_PATH" != "N" && "$MODEL_PATH" != "n" ]]; then
                    sudo mkdir -p "$MODEL_PATH"
                    echo "Environment=\"OLLAMA_MODELS=$MODEL_PATH\"" | sudo tee -a /etc/systemd/system/ollama.service.d/override.conf > /dev/null
                    ok "Model storage: $MODEL_PATH"
                fi

                sudo systemctl daemon-reload
                ok "Jetson CUDA overrides applied"
            fi

            # Start Ollama
            sudo systemctl enable ollama
            sudo systemctl start ollama
            sleep 3

            if curl -sf http://localhost:11434/api/tags &>/dev/null; then
                ok "Ollama is running"
                OLLAMA_RUNNING=true
            else
                fail "Ollama installed but failed to start. Check: journalctl -u ollama"
                ERRORS=$((ERRORS + 1))
            fi
        else
            fail "Ollama installation failed"
            ERRORS=$((ERRORS + 1))
        fi
    else
        fail "Ollama is required for worker operation"
        ERRORS=$((ERRORS + 1))
    fi
fi

# ── Step 5b: Ensure at least one model is available ─────────────────────────
if $OLLAMA_RUNNING; then
    MODEL_COUNT=$(curl -sf http://localhost:11434/api/tags | python3 -c "
import json, sys
print(len(json.load(sys.stdin).get('models', [])))
" 2>/dev/null || echo "0")

    if [ "$MODEL_COUNT" -eq 0 ]; then
        warn "No models installed"
        info "Recommended models for Jetson (8GB):"
        info "  gemma2:2b    — 1.6GB, fast, good quality"
        info "  llama3.2:3b  — 2.0GB, strong general purpose"
        info "  phi3:mini    — 2.2GB, efficient"
        ask "Pull a model now? Enter name (e.g., gemma2:2b) or N to skip:"
        read -r PULL_MODEL
        if [[ -n "$PULL_MODEL" && "$PULL_MODEL" != "N" && "$PULL_MODEL" != "n" ]]; then
            info "Pulling $PULL_MODEL (this may take a few minutes)..."
            ollama pull "$PULL_MODEL"
            if [ $? -eq 0 ]; then
                ok "Model $PULL_MODEL ready"
            else
                warn "Failed to pull $PULL_MODEL"
            fi
        fi
    fi

    # Verify GPU inference
    if $HAS_CUDA; then
        info "Testing GPU inference..."
        RESULT=$(curl -sf http://localhost:11434/api/tags | python3 -c "
import json, sys
models = json.load(sys.stdin).get('models', [])
if models:
    print(models[0]['name'])
else:
    print('')
" 2>/dev/null)
        if [ -n "$RESULT" ]; then
            TEST=$(curl -sf -X POST http://localhost:11434/api/generate \
                -d "{\"model\": \"$RESULT\", \"prompt\": \"Say OK\", \"stream\": false}" \
                2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('GPU' if data.get('response') else 'FAIL')
" 2>/dev/null || echo "FAIL")
            if [ "$TEST" = "GPU" ]; then
                ok "GPU inference verified with $RESULT"
            else
                warn "GPU inference test inconclusive"
            fi
        fi
    fi
fi

# ── Step 6: Clone repositories ──────────────────────────────────────────────
step "Repositories"

mkdir -p "$INSTALL_DIR"

clone_repo() {
    local REPO=$1
    local DIR="$INSTALL_DIR/$REPO"
    if [ -d "$DIR/.git" ]; then
        info "$REPO already cloned — pulling latest..."
        cd "$DIR" && git pull --ff-only && cd - > /dev/null
        ok "$REPO updated"
    else
        info "Cloning $REPO..."
        git clone "https://github.com/jasongagne-git/$REPO.git" "$DIR"
        ok "$REPO cloned"
    fi
}

clone_repo "sentinel"
clone_repo "sentinel-common"

# sentinel-worker might already be here if running from inside the repo
WORKER_DIR="$INSTALL_DIR/sentinel-worker"
if [ -d "$WORKER_DIR/.git" ]; then
    info "sentinel-worker already present"
    ok "sentinel-worker ready"
else
    clone_repo "sentinel-worker"
fi

# ── Step 7: Network configuration ──────────────────────────────────────────
step "Network Configuration"

# Auto-detect this machine's identity
HOSTNAME=$(hostname)
info "Hostname: $HOSTNAME"

# Detect IP addresses
IPS=$(ip -4 addr show | grep -oP 'inet \K[0-9.]+' | grep -v 127.0.0.1)
info "IP addresses:"
echo "$IPS" | while read -r ip; do echo "  $ip"; done

# Check mDNS
MDNS_AVAILABLE=false
if command -v avahi-resolve &>/dev/null; then
    if avahi-resolve -n "${HOSTNAME}.local" &>/dev/null; then
        MDNS_AVAILABLE=true
        ok "mDNS available: ${HOSTNAME}.local"
    fi
fi

# Get this node's address
if $MDNS_AVAILABLE; then
    DEFAULT_ADDR="${HOSTNAME}.local"
else
    DEFAULT_ADDR=$(echo "$IPS" | head -1)
fi
ask "This worker's address [$DEFAULT_ADDR]:"
read -r WORKER_ADDR
WORKER_ADDR="${WORKER_ADDR:-$DEFAULT_ADDR}"

# Get node ID
DEFAULT_NODE_ID="$HOSTNAME"
ask "What should we call this machine? [$DEFAULT_NODE_ID]:"
read -r NODE_ID
NODE_ID="${NODE_ID:-$DEFAULT_NODE_ID}"

# Get coordinator address
COORDINATOR_ADDR="${SENTINEL_COORDINATOR:-}"
if [ -z "$COORDINATOR_ADDR" ]; then
    ask "Coordinator address (IP or hostname.local, e.g., mayhem0.local):"
    read -r COORDINATOR_ADDR
fi

if [ -z "$COORDINATOR_ADDR" ]; then
    fail "Coordinator address is required"
    ERRORS=$((ERRORS + 1))
fi

# ── Step 8: Shared secret ──────────────────────────────────────────────────
step "Security"

CLUSTER_SECRET="${SENTINEL_SECRET:-}"
INVITE_TOKEN="${SENTINEL_INVITE:-}"

if [ -z "$CLUSTER_SECRET" ] && [ -n "$INVITE_TOKEN" ] && [ -n "$COORDINATOR_ADDR" ]; then
    # ── Invite token enrollment ──
    info "Enrolling with coordinator via invite token..."
    ENROLL_RESULT=$(python3 -c "
import json, urllib.request, sys

body = json.dumps({
    'invite_token': '$INVITE_TOKEN',
    'node_id': '$NODE_ID',
}).encode()

req = urllib.request.Request(
    'http://$COORDINATOR_ADDR:9400/v1/enroll',
    data=body, method='POST',
    headers={'Content-Type': 'application/json'},
)
try:
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read().decode())
    print(data.get('shared_secret', ''))
except urllib.error.HTTPError as e:
    body = e.read().decode() if e.fp else ''
    try:
        msg = json.loads(body).get('error', str(e))
    except Exception:
        msg = str(e)
    print(f'ERROR:{msg}', file=sys.stderr)
except Exception as e:
    print(f'ERROR:{e}', file=sys.stderr)
" 2>/tmp/sentinel_enroll_err)

    if [ -n "$ENROLL_RESULT" ]; then
        CLUSTER_SECRET="$ENROLL_RESULT"
        ok "Enrolled via invite token — secret received"
    else
        ENROLL_ERR=$(cat /tmp/sentinel_enroll_err 2>/dev/null | head -1)
        fail "Enrollment failed: ${ENROLL_ERR:-unknown error}"
        info "Falling back to manual secret entry..."
    fi
    rm -f /tmp/sentinel_enroll_err
fi

if [ -z "$CLUSTER_SECRET" ]; then
    echo ""
    echo "  How would you like to provide the shared cluster secret?"
    echo ""
    echo "    1) Paste it here (from coordinator admin)"
    echo "    2) Use an invite token (from coordinator admin)"
    echo "    3) Read from file: SENTINEL_SECRET=\$(cat /path/to/secret) ./install.sh"
    echo ""
    ask "Choice [1/2/3]:"
    read -r SECRET_METHOD

    case "${SECRET_METHOD:-1}" in
        2)
            ask "Invite token:"
            read -r INVITE_TOKEN
            ask "Coordinator address (IP or hostname.local):"
            read -r ENROLL_ADDR
            ENROLL_ADDR="${ENROLL_ADDR:-$COORDINATOR_ADDR}"

            if [ -n "$INVITE_TOKEN" ] && [ -n "$ENROLL_ADDR" ]; then
                info "Enrolling with coordinator..."
                ENROLL_RESULT=$(python3 -c "
import json, urllib.request, sys

body = json.dumps({
    'invite_token': '$INVITE_TOKEN',
    'node_id': '${NODE_ID:-$(hostname)}',
}).encode()

req = urllib.request.Request(
    'http://$ENROLL_ADDR:9400/v1/enroll',
    data=body, method='POST',
    headers={'Content-Type': 'application/json'},
)
try:
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read().decode())
    print(data.get('shared_secret', ''))
except urllib.error.HTTPError as e:
    body = e.read().decode() if e.fp else ''
    try:
        msg = json.loads(body).get('error', str(e))
    except Exception:
        msg = str(e)
    print(f'ERROR:{msg}', file=sys.stderr)
except Exception as e:
    print(f'ERROR:{e}', file=sys.stderr)
" 2>/tmp/sentinel_enroll_err)

                if [ -n "$ENROLL_RESULT" ]; then
                    CLUSTER_SECRET="$ENROLL_RESULT"
                    ok "Enrolled via invite token — secret received"
                else
                    ENROLL_ERR=$(cat /tmp/sentinel_enroll_err 2>/dev/null | head -1)
                    fail "Enrollment failed: ${ENROLL_ERR:-unknown error}"
                fi
                rm -f /tmp/sentinel_enroll_err
            fi
            ;;
        *)
            ask "Shared cluster secret:"
            read -r CLUSTER_SECRET

            # Detect corrupted input: repeated patterns from read -rs paste bugs
            if [ -n "$CLUSTER_SECRET" ] && [ ${#CLUSTER_SECRET} -ge 32 ]; then
                HALF_LEN=$(( ${#CLUSTER_SECRET} / 2 ))
                FIRST_HALF="${CLUSTER_SECRET:0:$HALF_LEN}"
                SECOND_HALF="${CLUSTER_SECRET:$HALF_LEN:$HALF_LEN}"
                if [ "$FIRST_HALF" = "$SECOND_HALF" ]; then
                    warn "Secret looks duplicated (paste bug detected) — using first half"
                    CLUSTER_SECRET="$FIRST_HALF"
                fi
            fi

            # Strip whitespace/control characters
            CLUSTER_SECRET=$(echo "$CLUSTER_SECRET" | tr -d '[:cntrl:]' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            ;;
    esac
fi

if [ -z "$CLUSTER_SECRET" ]; then
    fail "Shared secret is required"
    ERRORS=$((ERRORS + 1))
elif [ ${#CLUSTER_SECRET} -lt 16 ]; then
    fail "Shared secret must be at least 16 characters (got ${#CLUSTER_SECRET})"
    ERRORS=$((ERRORS + 1))
elif [ ${#CLUSTER_SECRET} -gt 128 ]; then
    fail "Shared secret suspiciously long (${#CLUSTER_SECRET} chars) — possible paste corruption"
    ERRORS=$((ERRORS + 1))
else
    ok "Shared secret set (${#CLUSTER_SECRET} chars)"
fi

# ── Step 9: Generate cluster.json ───────────────────────────────────────────
step "Configuration"

CONFIG_DIR="$INSTALL_DIR/sentinel-worker"
CONFIG_FILE="$CONFIG_DIR/cluster.json"

if [ $ERRORS -eq 0 ]; then
    # Write secret to env file
    ENV_FILE="$CONFIG_DIR/.env"
    echo "export SENTINEL_CLUSTER_SECRET=$CLUSTER_SECRET" > "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    ok "Secret written to $ENV_FILE (mode 600)"

    # Generate cluster.json
    python3 -c "
import json
config = {
    'cluster_id': 'sentinel-cluster',
    'mode': 'managed',
    'shared_secret_env': 'SENTINEL_CLUSTER_SECRET',
    'heartbeat_interval_s': 10.0,
    'heartbeat_timeout_s': 30.0,
    'default_failure_policy': 'recover',
    'tls_enabled': False,
    'coordinator': {
        'node_id': 'coordinator',
        'host': '$COORDINATOR_ADDR',
        'port': 9400,
        'role': 'both',
    },
    'workers': [
        {
            'node_id': '$NODE_ID',
            'host': '$WORKER_ADDR',
            'port': 9400,
            'role': 'worker',
        }
    ],
}
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)
print('OK')
"
    ok "Config written to $CONFIG_FILE"
    info "NOTE: The coordinator must also have this worker in its cluster.json"
else
    warn "Skipping config generation due to errors above"
fi

# ── Step 10: Test connectivity ──────────────────────────────────────────────
if [ $ERRORS -eq 0 ] && [ -n "$COORDINATOR_ADDR" ]; then
    step "Connectivity Test"

    if ping -c 1 -W 2 "$COORDINATOR_ADDR" &>/dev/null; then
        ok "Can reach coordinator at $COORDINATOR_ADDR"
    else
        warn "Cannot ping coordinator at $COORDINATOR_ADDR (may be firewalled — not necessarily a problem)"
    fi

    # Try the coordinator's worker service port
    if curl -sf --max-time 3 "http://${COORDINATOR_ADDR}:9400/v1/status" &>/dev/null; then
        ok "Coordinator service is responding on port 9400"
    else
        info "Coordinator not yet running on port 9400 (start it first)"
    fi
fi

# ── Summary ─────────────────────────────────────────────────────────────────
step "Summary"

echo ""
if [ $ERRORS -gt 0 ]; then
    fail "$ERRORS error(s) found — fix the issues above and re-run"
    exit 1
fi

ok "Installation complete!"
echo ""
echo -e "${BOLD}Here's what was set up:${NC}"
echo ""
echo "  Machine name:  $NODE_ID"
echo "  Address:       $WORKER_ADDR"
echo "  Coordinator:   $COORDINATOR_ADDR"
echo "  Config:        $CONFIG_FILE"
echo "  Secret:        $ENV_FILE (mode 600)"
echo ""
echo -e "${BOLD}What to do next:${NC}"
echo ""
echo "  Start the worker:"
echo "    cd $INSTALL_DIR/sentinel-worker"
echo "    source .env"
echo "    python3 run_worker.py --config cluster.json --node-id $NODE_ID"
echo ""
echo "  To start at boot (systemd):"
echo "    sudo cp sentinel-worker.service /etc/systemd/system/"
echo "    sudo systemctl enable sentinel-worker"
echo "    sudo systemctl start sentinel-worker"
echo ""
echo "  Make sure the coordinator knows about this worker:"
echo "    On the coordinator machine: ./add_worker.sh --id $NODE_ID --host $WORKER_ADDR"
echo ""
echo "  To uninstall:"
echo "    cd $INSTALL_DIR/sentinel-worker && ./uninstall.sh"
echo ""
