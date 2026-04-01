# EdgeTrust — Jetson Orin Compute Server Deployment Guide

> **Target hardware:** NVIDIA Jetson Orin (Nano / NX / AGX)  
> **JetPack version:** 5.x (CUDA 11.4+, TensorRT 8.x)  
> **Python version:** 3.8 – 3.11 (as shipped with JetPack 5)

This guide walks you through the complete server-side deployment of
`compute_server.py` on a Jetson Orin — from a fresh JetPack flash all the
way to a running, production-hardened FastAPI service accessible only via
the Tailscale Tailnet.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Flash JetPack (if needed)](#2-flash-jetpack-if-needed)
3. [First Boot & System Prep](#3-first-boot--system-prep)
4. [Install Tailscale](#4-install-tailscale)
5. [Clone the Repository](#5-clone-the-repository)
6. [Python Environment Setup](#6-python-environment-setup)
7. [Verify CUDA & TensorRT](#7-verify-cuda--tensorrt)
8. [Build the TensorRT Engine](#8-build-the-tensorrt-engine)
9. [Configure the Server](#9-configure-the-server)
10. [Smoke Test (Manual Run)](#10-smoke-test-manual-run)
11. [Run as a systemd Service (Production)](#11-run-as-a-systemd-service-production)
12. [Verify from the Raspberry Pi](#12-verify-from-the-raspberry-pi)
13. [Monitoring & Telemetry](#13-monitoring--telemetry)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Prerequisites

Before you begin, make sure you have:

| Item | Details |
|---|---|
| Jetson Orin board | Nano / NX / AGX — any variant works |
| microSD card ≥ 64 GB (or NVMe for AGX/NX) | UHS-I or faster |
| Host machine | Any x86 Linux/Mac with NVIDIA SDK Manager, or use the Jetson itself for setup |
| Tailscale account | Free tier is sufficient — [tailscale.com](https://tailscale.com) |
| Repository access | SSH key or GitHub token for `git clone` |
| Raspberry Pi already on your Tailnet | The RPi client (`100.1.1.2`) must already be enrolled |

---

## 2. Flash JetPack (if needed)

> Skip this step if your Jetson already runs **JetPack 5.x**.

### Option A — NVIDIA SDK Manager (recommended, on a host x86 Linux machine)

```bash
# Install SDK Manager from https://developer.nvidia.com/sdk-manager
sdkmanager --action install \
           --product Jetson \
           --target JETSON_ORIN_NANO \
           --version 5.1.3 \
           --flash-only
```

### Option B — Pre-built SD card image (Jetson Orin Nano only)

1. Download the `.img.gz` from [developer.nvidia.com/embedded/downloads](https://developer.nvidia.com/embedded/downloads).
2. Flash with Balena Etcher or:
```bash
# macOS / Linux
gunzip -c jetson-orin-nano-jp513-sd-card-image.img.gz | \
    sudo dd of=/dev/sdX bs=4M status=progress
```

> [!IMPORTANT]
> After first boot, run through the OEM setup wizard completely before
> proceeding. Set your hostname to something recognisable, e.g. `jetson-orin`.

---

## 3. First Boot & System Prep

Open a terminal on the Jetson (or SSH in once you have a local IP):

```bash
# 1 — Update the OS (takes a few minutes)
sudo apt-get update && sudo apt-get upgrade -y

# 2 — Install essential build tools
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    pkg-config \
    libatlas-base-dev \
    gfortran \
    git \
    curl \
    htop \
    nano

# 3 — Confirm CUDA is available
nvcc --version
# Expected output: Cuda compilation tools, release 11.4 (or 12.x)

# 4 — Confirm tegrastats works (the server uses this for GPU telemetry)
sudo tegrastats --interval 1000 --count 3
# Expected: lines with RAM, CPU, GR3D_FREQ, GPU temp
```

> [!NOTE]
> On **Jetson Orin**, `tegrastats` output format differs slightly from the
> classic Nano. The server's parser handles both formats — look for
> `GPU@XX.XC` or `GPU Therm@XX.XC` in the output.

---

## 4. Install Tailscale

The server listens **only** on the Tailscale interface (`100.x.x.x`). Tailscale must be running before the server starts.

```bash
# 1 — Install Tailscale (official script, works on Jetson/Ubuntu)
curl -fsSL https://tailscale.com/install.sh | sh

# 2 — Authenticate and join your Tailnet
sudo tailscale up

# 3 — Note down the Tailscale IP (you will need it for configuration)
tailscale ip -4
# Example output:  100.1.1.4
```

> [!IMPORTANT]
> The **Tailscale IP must match** the `TAILSCALE_IP` environment variable you
> set in Step 9. If the IP is different from `100.1.1.4`, you **must** update
> the config — the server refuses all traffic from non-Tailnet addresses.

### Lock down Tailscale ACLs (optional but recommended)

In your Tailscale admin console → **Access Controls**, restrict the Jetson so
that **only** your Raspberry Pi can reach port 8000:

```json
{
  "acls": [
    {
      "action": "accept",
      "src": ["100.1.1.2"],
      "dst": ["100.1.1.4:8000"]
    }
  ]
}
```

---

## 5. Clone the Repository

```bash
# Choose a working directory (home dir recommended)
cd ~

# Clone over SSH
git clone git@github.com:<your-org>/edge-trust-offload.git

# — OR — clone over HTTPS
git clone https://github.com/<your-org>/edge-trust-offload.git

cd edge-trust-offload

# Verify the server file is present
ls server/
# compute_server.py  requirements.txt  startup.sh
```

---

## 6. Python Environment Setup

JetPack 5 ships with Python 3.8. We create an isolated virtual environment
so project packages don't conflict with system packages.

```bash
cd ~/edge-trust-offload

# 1 — Create the virtual environment
python3 -m venv .jetson
source .jetson/bin/activate

# 2 — Upgrade pip
pip install --upgrade pip wheel setuptools

# 3 — Install server dependencies
pip install -r server/requirements.txt
```

### Expose JetPack system packages to the venv

TensorRT and pycuda are installed **system-wide** by JetPack and are NOT
available on PyPI. You need to tell the virtual environment to see them:

```bash
# Find where TensorRT Python bindings live
python3 -c "import tensorrt; print(tensorrt.__file__)"
# Typical path: /usr/lib/python3/dist-packages/tensorrt/__init__.py

# Symlink the system site-packages into the venv
SYSTEM_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "/usr/lib/python3/dist-packages")
VENV_SITE=$(.jetson/bin/python -c "import site; print(site.getsitepackages()[0])")

# Create .pth file so the venv finds system packages
echo "$SYSTEM_SITE" > "$VENV_SITE/jetpack_system.pth"

# Verify TensorRT is importable inside the venv
.jetson/bin/python -c "import tensorrt; print('TensorRT OK:', tensorrt.__version__)"
.jetson/bin/python -c "import pycuda.driver; print('pyCUDA OK')"
```

> [!TIP]
> If `import tensorrt` fails inside the venv even after the `.pth` trick,
> run this fix: `pip install --ignore-installed tensorrt` — but check your
> JetPack version first. On JetPack 5.1.x, TensorRT 8.5 or 8.6 is bundled.

### Install ONNX Runtime with CUDA/TensorRT provider (fallback)

```bash
# NVIDIA provides a GPU-enabled wheel for Jetson:
pip install onnxruntime-gpu \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# If the above fails (CUDA version mismatch), use the standard CPU wheel:
pip install onnxruntime
```

---

## 7. Verify CUDA & TensorRT

Run this quick sanity check before building the engine:

```bash
source .jetson/bin/activate

python3 - <<'EOF'
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

print(f"TensorRT version : {trt.__version__}")
print(f"CUDA device count: {cuda.Device.count()}")
dev = cuda.Device(0)
print(f"GPU device name  : {dev.name()}")
print(f"GPU total memory : {dev.total_memory() // 1024**2} MB")
EOF
```

Expected output (Jetson Orin Nano, JetPack 5.1):
```
TensorRT version : 8.6.11
CUDA device count: 1
GPU device name  : Orin
GPU total memory : 7644 MB
```

---

## 8. Build the "Heavy" Model

The server loads a pre-built `.trt` engine file from `models/gesture_heavy.trt`. You must generate the "Heavy" 1D-CNN architecture specified in the project requirements before starting the server.

### Step 8a — Generate the ONNX model

On the Jetson, run the provided build script. This creates a schema-accurate but randomly initialized 1D-CNN model (256-unit hidden layers) for benchmarking.

```bash
source .jetson/bin/activate
python3 scripts/build_heavy_model.py
# Output: models/gesture_heavy.onnx
```

### Step 8b — Convert to TensorRT Engine

Convert the ONNX file to a high-speed engine optimized for the Orin's 128 CUDA cores.

```bash
# --fp16   : enables FP16 precision (uses Orin's Tensor Cores)
# --workspace=512 : 512 MB GPU workspace during build
trtexec \
    --onnx=models/gesture_heavy.onnx \
    --saveEngine=models/gesture_heavy.trt \
    --fp16 \
    --workspace=512
```

> [!NOTE]
> Building the engine takes **2–5 minutes**. Subsequent server restarts load from the cached `.trt` file instantly.

> [!TIP]
> To also enable INT8 (fastest, requires calibration data):
> ```bash
> trtexec --onnx=models/gesture_cnn.onnx \
>         --saveEngine=models/gesture_heavy.trt \
>         --int8 --fp16 --workspace=1024
> ```

---

## 9. Configure the Server

All configuration is done via **environment variables**. To prevent binding errors, the `.env` file should use your actual Tailscale IP.

```bash
cd ~/edge-trust-offload

# Detect your Tailscale IP automatically
export MY_TS_IP=$(tailscale ip -4)

cat > server/.env <<EOF
# ── Network ──────────────────────────────────────────────────────
TAILSCALE_IP=$MY_TS_IP
SERVER_PORT=8000

# ── Zero-Trust Device Allowlist ───────────────────────────────────
# format: <device_id>:<tailscale_ip>
ALLOWED_DEVICES=rpi-client-scheduler:100.1.1.2,rpi-client-01:100.1.1.2
EOF
```

> [!IMPORTANT]
> **Check your IP bind**: If the server fails with `Errno 99 (Cannot assign requested address)`, ensure `TAILSCALE_IP` matches the output of `tailscale ip -4`.

Make the startup script executable:

```bash
chmod +x server/startup.sh
```

---

## 10. Smoke Test (Manual Run)

Run the server interactively first to verify everything is wired up correctly.

```bash
cd ~/edge-trust-offload
source .jetson/bin/activate

# Load env vars
export $(grep -v '^#' server/.env | xargs)

# Start the server (Ctrl+C to stop)
python server/compute_server.py
```

Expected startup output:
```
2026-04-01 10:00:00 [INFO] EdgeTrust_Jetson: ============================================================
2026-04-01 10:00:00 [INFO] EdgeTrust_Jetson: EdgeTrust Compute Server — Jetson Nano
2026-04-01 10:00:00 [INFO] EdgeTrust_Jetson: Bind address : 100.1.1.4:8000
2026-04-01 10:00:00 [INFO] EdgeTrust_Jetson: TRT engine   : /home/.../models/gesture_heavy.trt
2026-04-01 10:00:00 [INFO] EdgeTrust_Jetson: Allowlist    : ['rpi-client-01', 'rpi-client-scheduler']
2026-04-01 10:00:00 [INFO] EdgeTrust_Jetson: TensorRT engine loaded from .../gesture_heavy.trt
2026-04-01 10:00:00 [INFO] EdgeTrust_Jetson: TensorRTEngine ready | backend=tensorrt | n_channels=8
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://100.1.1.4:8000
```

### Test the health endpoint from another Tailnet node

```bash
# From the Raspberry Pi or any Tailnet machine:
curl http://100.1.1.4:8000/health
```

Expected JSON:
```json
{
  "status": "ok",
  "server": "jetson-nano",
  "tailscale_ip": "100.1.1.4",
  "inference_backend": "tensorrt",
  "system_stats": {
    "gpu_temp_c": 35.0,
    "ram_used_mb": 248.5,
    "gpu_util_pct": 0.0,
    "cpu_pct": 4.2
  }
}
```

### Test a full inference request

```bash
# From the Raspberry Pi (replace 100.1.1.4 as needed):
curl -s -X POST http://100.1.1.4:8000/api/v1/offload/fft \
  -H "Content-Type: application/json" \
  -H "X-Device-ID: rpi-client-scheduler" \
  -d '{
    "device_id": "rpi-client-scheduler",
    "timestamp_ms": 1711928374123,
    "workload_type": "fft_1024",
    "payload": {
      "sampling_rate_hz": 256,
      "n_channels": 1,
      "samples": '"$(python3 -c "import json,random; print(json.dumps([random.uniform(-1,1) for _ in range(1024)]))")"'
    },
    "metadata": {"client_cpu_load_pct": 50.0, "battery_level_pct": 100}
  }' | python3 -m json.tool
```

---

## 11. Run as a systemd Service (Production)

Running the server as a `systemd` service ensures it starts automatically on
boot and restarts on crashes.

### Step 11a — Create the service file

```bash
sudo nano /etc/systemd/system/edgetrust-server.service
```

Paste the following (adjust paths and username as needed):

```ini
[Unit]
Description=EdgeTrust Jetson Orin Compute Server
Documentation=https://github.com/<your-org>/edge-trust-offload
After=network-online.target tailscaled.service
Wants=network-online.target tailscaled.service

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/edge-trust-offload
EnvironmentFile=/home/YOUR_USERNAME/edge-trust-offload/server/.env
ExecStartPre=/bin/bash -c 'until tailscale status --json | python3 -c "import sys,json; s=json.load(sys.stdin); exit(0 if s[\"BackendState\"]==\"Running\" else 1)"; do sleep 2; done'
ExecStart=/home/YOUR_USERNAME/edge-trust-offload/.jetson/bin/python \
          /home/YOUR_USERNAME/edge-trust-offload/server/compute_server.py
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=edgetrust-server

# Resource limits — prevent the server from starving the OS
CPUQuota=80%
MemoryMax=4G

[Install]
WantedBy=multi-user.target
```

> [!IMPORTANT]
> Replace every occurrence of `YOUR_USERNAME` with the actual Linux username
> on your Jetson (e.g., `nvidia`, `jetson`, or your own username).

### Step 11b — Enable and start the service

```bash
# Reload systemd to pick up the new unit file
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable edgetrust-server.service

# Start it now
sudo systemctl start edgetrust-server.service

# Verify it started correctly
sudo systemctl status edgetrust-server.service
```

### Step 11c — View live logs

```bash
# Follow logs in real time
sudo journalctl -u edgetrust-server.service -f

# View the last 100 lines
sudo journalctl -u edgetrust-server.service -n 100
```

---

## 12. Verify from the Raspberry Pi

On the **Raspberry Pi** side, update your environment to point at the Jetson:

```bash
# On the Raspberry Pi
export JETSON_IP="100.1.1.4"   # Your Jetson's Tailscale IP
export JETSON_PORT="8000"

# Run the scheduler — it will now offload when beneficial
python3 client/main_scheduler.py
```

You should see log lines like:
```
Decision: OFFLOAD (Network 3.21ms is below threshold 18.40ms)
```

And on the Jetson logs:
```
AUTH OK: device_id=rpi-client-scheduler ip=100.1.1.2
Request OK | device=rpi-client-scheduler | gesture=IndexFlexion (87%) | t_infer=2.14 ms | t_total=3.88 ms | GPU=38.0°C
```

---

## 13. Monitoring & Telemetry

### Real-time tegrastats dashboard

```bash
# On the Jetson — full system stats every 500 ms
watch -n 0.5 tegrastats

# Or in one-shot mode for scripting:
tegrastats --interval 500 --count 1
```

### Server telemetry log

Every inference request appends a JSON record to:

```
logs/jetson_telemetry.jsonl
```

Each line contains:

| Field | Description |
|---|---|
| `timestamp_iso` | UTC timestamp of the request |
| `device_id` | Which RPi node triggered the inference |
| `t_inference_ms` | Pure GPU compute time (T_inference) |
| `t_total_ms` | Full request wall-clock time |
| `gesture` | Predicted gesture class |
| `confidence` | Softmax confidence score |
| `backend` | `tensorrt` / `onnxruntime` / `software_numpy` |
| `gpu_temp_c` | GPU temperature from tegrastats |
| `ram_used_mb` | RAM usage from tegrastats |
| `gpu_util_pct` | GR3D utilisation % |
| `client_ip` | Tailscale IP of the requesting RPi |

Quick analysis:

```bash
# Average inference time
cat logs/jetson_telemetry.jsonl | \
    python3 -c "
import sys, json, statistics
rows = [json.loads(l) for l in sys.stdin]
times = [r['t_inference_ms'] for r in rows]
print(f'Requests : {len(times)}')
print(f'Avg T_inf: {statistics.mean(times):.2f} ms')
print(f'Max T_inf: {max(times):.2f} ms')
print(f'Min T_inf: {min(times):.2f} ms')
"
```

### API documentation (Swagger UI)

While the server is running, open from any Tailnet machine:

```
http://100.1.1.4:8000/docs
```

---

## 14. Troubleshooting

### Server won't start — `OSError: [Errno 99] Cannot assign requested address`

The Tailscale interface isn't up yet when the server tries to bind.

```bash
# Fix: ensure tailscaled is running first
sudo systemctl status tailscaled
sudo tailscale up
# Then restart the server
sudo systemctl restart edgetrust-server.service
```

The `ExecStartPre` line in the systemd unit already retries until Tailscale
is `Running`, but manual starts skip this.

---

### `import tensorrt` fails inside the venv

```bash
# Confirm the .pth file points to the right directory
cat .jetson/lib/python3.*/site-packages/jetpack_system.pth

# Recheck the system TensorRT location
find /usr -name "tensorrt" -type d 2>/dev/null | head -5

# Update the .pth file with the correct path found above
echo "/usr/lib/python3/dist-packages" > \
    .jetson/lib/python3.*/site-packages/jetpack_system.pth
```

---

### Server shows `backend=software_numpy` instead of `tensorrt`

The `.trt` engine file is either missing or failed to load.

```bash
# Check if it exists
ls -lh models/gesture_heavy.trt

# Check TensorRT load errors in the log
sudo journalctl -u edgetrust-server.service | grep -i "trt\|tensorrt\|pycuda"

# Rebuild the engine (see Step 8)
trtexec --onnx=models/gesture_cnn.onnx \
        --saveEngine=models/gesture_heavy.trt \
        --fp16 --workspace=512
```

---

### `403 Forbidden` — auth failures from the RPi

#### Case A: Source IP mismatch
```
AUTH FAIL: device_id 'rpi-client-scheduler' arrived from 100.1.1.3 but is registered to 100.1.1.2
```
Update `ALLOWED_DEVICES` in `server/.env` with the correct Tailscale IP, then:
```bash
sudo systemctl restart edgetrust-server.service
```

#### Case B: Missing `X-Device-ID` header
The Raspberry Pi scheduler must add this header to every request. Verify
`execute_path_b_offload()` in `main_scheduler.py` includes:
```python
headers={"X-Device-ID": payload["device_id"]}
```

---

### `tegrastats` not found (GPU temp shows `null`)

```bash
# Confirm tegrastats is in PATH
which tegrastats         # expected: /usr/bin/tegrastats

# If missing, it's a JetPack installation issue — reinstall JetPack components
sudo apt-get install --reinstall nvidia-l4t-tools
```

---

### GPU temperature is unusually high (> 75°C)

```bash
# Check the cooling solution
# Enable maximum cooling fan speed
sudo /usr/bin/jetson_clocks --fan

# Or set a specific fan speed (0–255)
sudo sh -c "echo 200 > /sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1"
```

---

## Quick Reference

```bash
# --- On the Jetson (once deployed) ---

# Check service status
sudo systemctl status edgetrust-server.service

# View live logs
sudo journalctl -u edgetrust-server.service -f

# Restart the server
sudo systemctl restart edgetrust-server.service

# Check Tailscale connectivity
tailscale status
tailscale ping 100.1.1.2   # Ping the Raspberry Pi

# Live GPU / RAM monitor
watch -n 1 tegrastats

# View telemetry log
tail -f logs/jetson_telemetry.jsonl | python3 -m json.tool
```
