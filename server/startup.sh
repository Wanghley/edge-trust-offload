#!/usr/bin/env bash
# =============================================================================
# startup.sh — EdgeTrust Compute Server launcher (Jetson Nano)
# =============================================================================
# Usage:
#   chmod +x server/startup.sh
#   ./server/startup.sh
#
# Or as a systemd service (see docs/rpi_run_guide.md for pattern):
#   [Service]
#   ExecStart=/path/to/edge-trust-offload/server/startup.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Resolve Tailscale IP dynamically (overridable via ENV) ───────────────────
if [[ -z "${TAILSCALE_IP:-}" ]]; then
    # Query the local Tailscale daemon for the Jetson's own Tailscale IP.
    # Falls back to the architecture default if tailscale is not in PATH.
    if command -v tailscale &>/dev/null; then
        TAILSCALE_IP="$(tailscale ip -4 2>/dev/null | head -n1 || echo '100.1.1.4')"
    else
        TAILSCALE_IP="100.1.1.4"
    fi
fi

export TAILSCALE_IP
export SERVER_PORT="${SERVER_PORT:-8000}"

# ── Model paths ──────────────────────────────────────────────────────────────
export MODELS_DIR="${PROJECT_ROOT}/models"
mkdir -p "${MODELS_DIR}"

# ── Logs dir ─────────────────────────────────────────────────────────────────
mkdir -p "${PROJECT_ROOT}/logs"

# ── Zero-Trust device allowlist (override in production via secrets manager) ─
export ALLOWED_DEVICES="${ALLOWED_DEVICES:-rpi-client-01:100.1.1.2,rpi-client-scheduler:100.1.1.2}"

echo "======================================================"
echo " EdgeTrust Compute Server — Jetson Nano"
echo " Tailscale IP : ${TAILSCALE_IP}"
echo " Port         : ${SERVER_PORT}"
echo " Models dir   : ${MODELS_DIR}"
echo " Allowlist    : ${ALLOWED_DEVICES}"
echo "======================================================"

# ── Verify Tailscale is connected before binding ─────────────────────────────
if command -v tailscale &>/dev/null; then
    TS_STATUS="$(tailscale status --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('BackendState','Unknown'))" 2>/dev/null || echo 'Unknown')"
    echo "Tailscale status: ${TS_STATUS}"
    if [[ "${TS_STATUS}" != "Running" ]]; then
        echo "WARNING: Tailscale is not Running. Server will only be reachable locally."
    fi
fi

# ── Environment Setup ──────────────────────────────────────────────────────────
# Ensure CUDA binaries and Tegra driver libraries are in the path (standard on Jetson/JetPack)
export PATH="/usr/local/cuda/bin:${PATH:-}"
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# ── Launch server ─────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"
# Automatically use the .jetson venv if it exists
if [[ -f ".jetson/bin/python" ]]; then
    echo "Using virtual environment: .jetson"
    exec .jetson/bin/python server/compute_server.py
else
    exec python3 server/compute_server.py
fi
