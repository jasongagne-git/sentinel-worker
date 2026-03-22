#!/usr/bin/env bash

# SENTINEL Worker — Uninstall Script
#
# Stops the worker process, removes configuration, systemd service,
# and optionally cloned repositories.
#
# Does NOT uninstall Ollama or remove Ollama models (shared system dependencies).
#
# Usage:
#   ./uninstall.sh              # interactive, confirms each step
#   ./uninstall.sh --all        # remove everything without prompts
#   ./uninstall.sh --dry-run    # show what would be removed, change nothing

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
WORKER_DIR="$INSTALL_DIR/sentinel-worker"

MODE=""
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)     MODE="all"; shift;;
        --dry-run) DRY_RUN=true; shift;;
        --help|-h)
            echo "Usage: $0 [--all] [--dry-run]"
            echo "  --all        Remove everything without prompts"
            echo "  --dry-run    Show what would be removed, change nothing"
            echo "  (default)    Interactive — confirm each step"
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

confirm() {
    if [ "$MODE" = "all" ]; then
        return 0
    fi
    ask "$1 [y/N]"
    read -r REPLY
    [[ "$REPLY" =~ ^[Yy] ]]
}

do_rm() {
    if $DRY_RUN; then
        info "(dry run) would remove: $1"
    else
        rm -f "$1"
        ok "Removed $(basename "$1")"
    fi
}

echo -e "\n${BOLD}SENTINEL Worker Uninstall${NC}"
if $DRY_RUN; then
    echo -e "${YELLOW}(dry run — no changes will be made)${NC}"
fi
echo ""

if [ ! -d "$WORKER_DIR" ]; then
    fail "Worker directory not found: $WORKER_DIR"
    exit 1
fi

# ── Step 1: Stop the worker if running ────────────────────────────────────
step "Worker Process"

# Check systemd first
SYSTEMD_INSTALLED=false
if systemctl list-unit-files sentinel-worker.service &>/dev/null 2>&1; then
    SYSTEMD_INSTALLED=true
    if systemctl is-active sentinel-worker.service &>/dev/null 2>&1; then
        info "sentinel-worker.service is running"
        if $DRY_RUN; then
            info "(dry run) would stop and disable sentinel-worker.service"
        elif confirm "Stop and disable sentinel-worker.service?"; then
            sudo systemctl stop sentinel-worker.service
            sudo systemctl disable sentinel-worker.service
            ok "Stopped and disabled"
        fi
    fi
fi

# Check for manually-started process
WORKER_PID=$(pgrep -f "run_worker.py" 2>/dev/null || true)
if [ -n "$WORKER_PID" ]; then
    info "Worker process running (PID: $WORKER_PID)"
    if $DRY_RUN; then
        info "(dry run) would stop worker process (PID $WORKER_PID)"
    elif confirm "Stop worker process?"; then
        kill "$WORKER_PID" 2>/dev/null || true
        sleep 1
        if kill -0 "$WORKER_PID" 2>/dev/null; then
            warn "Process didn't stop cleanly, sending SIGKILL"
            kill -9 "$WORKER_PID" 2>/dev/null || true
        fi
        ok "Worker process stopped"
    fi
else
    if ! $SYSTEMD_INSTALLED; then
        info "No worker process running"
    fi
fi

# ── Step 2: Systemd unit file ────────────────────────────────────────────
step "Systemd Unit"

if [ -f /etc/systemd/system/sentinel-worker.service ]; then
    if $DRY_RUN; then
        info "(dry run) would remove /etc/systemd/system/sentinel-worker.service"
    elif confirm "Remove /etc/systemd/system/sentinel-worker.service?"; then
        sudo rm /etc/systemd/system/sentinel-worker.service
        sudo systemctl daemon-reload
        ok "Removed systemd unit"
    fi
else
    info "No systemd unit file installed"
fi

# ── Step 3: Generated config files ────────────────────────────────────────
step "Configuration Files"

for f in .env cluster.json; do
    if [ -f "$WORKER_DIR/$f" ]; then
        do_rm "$WORKER_DIR/$f"
    fi
done

# ── Step 4: Ollama overrides ─────────────────────────────────────────────
step "Ollama Configuration"

OLLAMA_OVERRIDE="/etc/systemd/system/ollama.service.d/override.conf"
if [ -f "$OLLAMA_OVERRIDE" ]; then
    info "Found Ollama override: $OLLAMA_OVERRIDE"
    warn "Shared system config — not removing"
    info "To remove manually: sudo rm $OLLAMA_OVERRIDE && sudo systemctl daemon-reload"
else
    info "No Ollama override found"
fi

# ── Step 5: Cloned repositories ──────────────────────────────────────────
step "Repositories"

REPOS=("sentinel" "sentinel-common" "sentinel-worker")
REPOS_FOUND=()
for repo in "${REPOS[@]}"; do
    if [ -d "$INSTALL_DIR/$repo/.git" ]; then
        REPOS_FOUND+=("$repo")
    fi
done

if [ ${#REPOS_FOUND[@]} -gt 0 ]; then
    info "Found ${#REPOS_FOUND[@]} SENTINEL repo(s): ${REPOS_FOUND[*]}"

    if $DRY_RUN; then
        for repo in "${REPOS_FOUND[@]}"; do
            DIR="$INSTALL_DIR/$repo"
            DIRTY=$(cd "$DIR" && git status --porcelain 2>/dev/null | head -1)
            if [ -n "$DIRTY" ]; then
                info "(dry run) would remove $repo (has uncommitted changes — would prompt)"
            else
                info "(dry run) would remove $repo"
            fi
        done
    elif [ "$MODE" = "all" ] || confirm "Remove SENTINEL repositories?"; then
        for repo in "${REPOS_FOUND[@]}"; do
            DIR="$INSTALL_DIR/$repo"
            DIRTY=$(cd "$DIR" && git status --porcelain 2>/dev/null | head -1)
            if [ -n "$DIRTY" ]; then
                warn "$repo has uncommitted changes!"
                if ! confirm "Remove $repo anyway (uncommitted changes will be lost)?"; then
                    warn "Skipping $repo"
                    continue
                fi
            fi
            rm -rf "$DIR"
            ok "Removed $repo"
        done
    else
        warn "Keeping repositories"
    fi
else
    info "No SENTINEL repositories found in $INSTALL_DIR"
fi

# ── Summary ──────────────────────────────────────────────────────────────
step "Done"
echo ""
if $DRY_RUN; then
    ok "Dry run complete — no changes were made."
else
    ok "Worker uninstall complete."
fi
echo ""
info "Not removed:"
echo "  - Ollama (systemd service, binary, models)"
echo "  - Python 3, Git, system packages"
echo ""
