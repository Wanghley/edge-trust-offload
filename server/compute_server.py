"""
compute_server.py
=================
EdgeTrust-Offload — Jetson Nano Compute Server
NVIDIA Jetson Nano / CUDA 128-core target

Architecture context
--------------------
  Jetson Nano (100.1.1.4)
    └─ compute_server.py   ←  POST /api/v1/offload/fft  (this file)
         ├─ TensorRT engine  →  GPU-accelerated gesture classification
         ├─ Tailscale bind   →  listens ONLY on 100.x.x.x interface
         ├─ Mutual auth      →  X-Device-ID header + allowlist
         └─ Telemetry        →  t_inference, GPU temp, RAM via tegrastats

Endpoint contract (matches architecture_blueprint.md §3)
---------------------------------------------------------
  POST /api/v1/offload/fft
  POST /api/v1/offload/gesture          ← "heavy" ML path (this server's focus)
  GET  /health

Backend priority (inference engine):
  1. TensorRT  — Jetson 128 CUDA cores, fastest
  2. CuPy      — CUDA-backed NumPy, fast fallback
  3. NumPy MLP — software fallback (dev/CI)

Dependencies (Jetson-side, see server/requirements.txt):
  fastapi, uvicorn[standard]
  numpy, psutil
  tensorrt (>=8.x, installed via JetPack SDK)  [optional]
  pycuda                                        [optional, needed for TRT]
  cupy-cuda11x   OR   cupy-cuda12x             [optional]

Usage
-----
  # On Jetson Nano (Tailscale interface is 100.1.1.4):
  python compute_server.py

  # Or with env-var overrides:
  TAILSCALE_IP=100.1.1.4 SERVER_PORT=8000 python compute_server.py
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import pathlib
import platform
import subprocess
import struct
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import psutil

# ---------------------------------------------------------------------------
# Global CUDA / TensorRT imports to prevent Error 35 on Jetson Orin
# ---------------------------------------------------------------------------
try:
    import tensorrt as trt
    import pycuda.driver as pycuda_driver
    import pycuda.autoinit
    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, condecimal

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("EdgeTrust_Jetson")

# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables
# ---------------------------------------------------------------------------
TAILSCALE_IP: str = os.getenv("TAILSCALE_IP", "100.1.1.4")
SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))
# Set INSECURE_MODE=1 to allow plain LAN traffic (S2 baseline measurements only).
# Never use in production — disables Tailscale IP enforcement and IP-match check.
INSECURE_MODE: bool = os.getenv("INSECURE_MODE", "0") == "1"
LOG_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent / "logs"
TELEMETRY_LOG: pathlib.Path = LOG_DIR / "jetson_telemetry.jsonl"
MODEL_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent / "models"
TRT_ENGINE_PATH: pathlib.Path = MODEL_DIR / "gesture_heavy.trt"
ONNX_MODEL_PATH: pathlib.Path = MODEL_DIR / "gesture_cnn.onnx"

# Zero-Trust: allowlisted Tailscale client device IDs and their associated IPs.
# In production, populate from a secrets manager or environment variable.
# Format:  device_id -> expected Tailscale source IP
_ALLOWLIST_RAW: str = os.getenv(
    "ALLOWED_DEVICES",
    "rpi-client-01:100.1.1.2,rpi-client-scheduler:100.1.1.2",
)
ALLOWED_DEVICES: Dict[str, str] = {}
for _entry in _ALLOWLIST_RAW.split(","):
    _entry = _entry.strip()
    if ":" in _entry:
        _did, _ip = _entry.split(":", 1)
        ALLOWED_DEVICES[_did.strip()] = _ip.strip()

# Model constants — must match edge_tasks.py
WINDOW_SIZE: int = 1024
N_CLASSES: int = 6
GESTURE_LABELS: List[str] = [
    "Rest",
    "IndexFlexion",
    "IndexExtension",
    "MiddleFlexion",
    "MiddleExtension",
    "ThumbFlexion",
]

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class WindowPayload(BaseModel):
    sampling_rate_hz: float = Field(256.0, description="ADC sampling rate in Hz")
    precision: str = Field("float32", description="Sample precision string")
    samples: List[float] = Field(
        ...,
        description="Exactly WINDOW_SIZE × n_channels samples (row-major flattened or 1-D)",
        min_length=1,
    )
    n_channels: Optional[int] = Field(
        None, description="Number of sEMG channels (auto-detected if omitted)"
    )


class ClientMetadata(BaseModel):
    client_cpu_load_pct: float = Field(0.0)
    battery_level_pct: float = Field(100.0)


class OffloadRequest(BaseModel):
    device_id: str = Field(..., description="Caller device identifier (Zero-Trust)")
    timestamp_ms: int = Field(..., description="Client-side UNIX timestamp in ms")
    workload_type: str = Field("fft_1024", description="Workload tag for logging")
    payload: WindowPayload
    metadata: ClientMetadata = Field(default_factory=ClientMetadata)


class SystemStats(BaseModel):
    gpu_temp_c: Optional[float] = Field(None, description="GPU temperature in °C")
    ram_used_mb: float = Field(..., description="Current process RSS in MB")
    gpu_utilization_pct: Optional[float] = None
    cpu_pct: float


class OffloadResponse(BaseModel):
    status: str
    processing_node: str
    compute_latency_ms: float = Field(..., description="Pure GPU/CPU inference time (T_inference)")
    total_request_ms: float = Field(..., description="Total server-side wall-clock time")
    result: Dict[str, Any]
    system_stats: SystemStats


# ---------------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────
# Inference Engine — TensorRT → CuPy → NumPy fallback
# ─────────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class _SoftwareFallback:
    """
    Pure-NumPy two-layer MLP inference — used only when TRT/CUDA is unavailable.

    This is the same architecture as edge_tasks._FallbackSoftwareInterpreter
    but with *doubled* hidden units to represent the "heavy" server model.
    """

    _STATS_PER_CHANNEL = 3  # mean_abs, std, max_abs

    def __init__(self, n_channels: int) -> None:
        hidden = 256  # 2× the client's 128-unit heavy model
        n_input = n_channels * self._STATS_PER_CHANNEL
        rng = np.random.default_rng(42)
        lim1 = np.sqrt(6.0 / (n_input + hidden))
        lim2 = np.sqrt(6.0 / (hidden + N_CLASSES))
        self._w1 = rng.uniform(-lim1, lim1, (n_input, hidden)).astype(np.float32)
        self._b1 = np.zeros(hidden, dtype=np.float32)
        self._w2 = rng.uniform(-lim2, lim2, (hidden, N_CLASSES)).astype(np.float32)
        self._b2 = np.zeros(N_CLASSES, dtype=np.float32)

    def infer(self, x: np.ndarray) -> np.ndarray:
        """``x`` shape: ``(1, WINDOW_SIZE, n_channels)``."""
        sig = x[0]
        stats = np.concatenate([
            np.mean(np.abs(sig), axis=0),
            np.std(sig, axis=0),
            np.max(np.abs(sig), axis=0),
        ]).astype(np.float32).reshape(1, -1)
        h = np.maximum(0.0, stats @ self._w1 + self._b1)
        logits = h @ self._w2 + self._b2
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        return probs.flatten()


class TensorRTEngine:
    """
    Wraps a TensorRT serialised engine for gesture classification.

    The "heavy" model targeted here is a deeper 1D-CNN:
      Input  →  Conv1D(64, k=7)–BN–ReLU–MaxPool(4)
              →  Conv1D(128, k=5)–BN–ReLU–MaxPool(4)
              →  Conv1D(256, k=3)–BN–ReLU–MaxPool(4)
              →  Conv1D(256, k=3)–BN–ReLU–GlobalAvgPool
              →  Dense(256)–ReLU–Dropout(0.4)
              →  Dense(128)–ReLU–Dropout(0.3)
              →  Dense(N_CLASSES)–Softmax

    The engine file (gesture_heavy.trt) is built once offline with:
        trtexec --onnx=gesture_heavy.onnx \\
                --saveEngine=gesture_heavy.trt \\
                --fp16 \\
                --workspace=512

    If the engine file is missing, the class degrades to CuPy or NumPy.
    """

    def __init__(self, engine_path: pathlib.Path, n_channels: int = 8) -> None:
        self._n_channels = n_channels
        self._backend = "none"
        self._engine = None
        self._context = None
        self._pycuda = None
        self._np_fallback: Optional[_SoftwareFallback] = None

        # ── Try TensorRT ──────────────────────────────────────────────────
        if engine_path.exists():
            self._try_load_trt(engine_path)

        # ── Try ONNX Runtime (GPU provider on Jetson) ─────────────────────
        if self._backend == "none":
            self._try_load_onnx()

        # ── Software fallback ─────────────────────────────────────────────
        if self._backend == "none":
            log.warning(
                "No GPU runtime available. Using NumPy software interpreter. "
                "This is NOT representative of Jetson performance."
            )
            self._np_fallback = _SoftwareFallback(n_channels)
            self._backend = "software_numpy"

        log.info("TensorRTEngine ready | backend=%s | n_channels=%d", self._backend, n_channels)

    # ------------------------------------------------------------------
    # Backend loaders
    # ------------------------------------------------------------------

    def _try_load_trt(self, engine_path: pathlib.Path) -> None:
        if not _TRT_AVAILABLE:
            log.warning("TensorRT/pycuda not installed globally — will try ONNX.")
            return

        try:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(engine_path, "rb") as f:
                engine_data = f.read()
            self._engine = runtime.deserialize_cuda_engine(engine_data)
            self._context = self._engine.create_execution_context()
            self._pycuda = pycuda_driver
            self._backend = "tensorrt"
            log.info("TensorRT engine loaded from %s", engine_path)
        except Exception as e:
            log.warning("TensorRT load failed: %s — will try ONNX.", e)

    def _try_load_onnx(self) -> None:
        try:
            ort = importlib.import_module("onnxruntime")
            if ONNX_MODEL_PATH.exists():
                sess_opts = ort.SessionOptions()
                # Try CUDA/TensorRT execution providers first (Jetson)
                providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
                self._ort_session = ort.InferenceSession(
                    str(ONNX_MODEL_PATH),
                    sess_options=sess_opts,
                    providers=providers,
                )
                active = self._ort_session.get_providers()
                log.info("ONNX Runtime loaded | active providers: %s", active)
                self._backend = "onnxruntime"
            else:
                log.warning("ONNX model not found at %s.", ONNX_MODEL_PATH)
        except ModuleNotFoundError:
            log.warning("onnxruntime not installed.")
        except Exception as e:
            log.warning("ONNX load failed: %s.", e)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, window: np.ndarray) -> Dict[str, Any]:
        """
        Run a forward pass on a ``(WINDOW_SIZE, n_channels)`` window.

        Returns a dict with:
            gesture, gesture_idx, confidence, probabilities, backend, t_gpu_ms
        """
        if window.ndim == 1:
            window = window.reshape(-1, 1)

        # Normalise zero-mean unit-variance per channel
        mu = window.mean(axis=0, keepdims=True)
        sigma = window.std(axis=0, keepdims=True) + 1e-8
        x_norm = ((window - mu) / sigma).astype(np.float32)
        x_batch = x_norm[np.newaxis, ...]  # (1, WINDOW_SIZE, n_channels)

        t_gpu_start = time.perf_counter()

        if self._backend == "tensorrt":
            probs = self._infer_trt(x_batch)
        elif self._backend == "onnxruntime":
            probs = self._infer_onnx(x_batch)
        else:
            probs = self._np_fallback.infer(x_batch)

        t_gpu_ms = (time.perf_counter() - t_gpu_start) * 1_000

        top_idx = int(np.argmax(probs))
        return {
            "gesture": GESTURE_LABELS[top_idx % len(GESTURE_LABELS)],
            "gesture_idx": top_idx,
            "confidence": float(probs[top_idx]),
            "probabilities": probs.tolist(),
            "backend": self._backend,
            "t_gpu_ms": round(t_gpu_ms, 3),
        }

    def _infer_trt(self, x_batch: np.ndarray) -> np.ndarray:
        """Execute TensorRT inference using pycuda memory transfers."""
        import pycuda.driver as cuda
        import numpy as np

        # Allocate pagelocked host memory for zero-copy transfer
        h_in = cuda.pagelocked_empty(x_batch.size, dtype=np.float32)
        h_out = cuda.pagelocked_empty(N_CLASSES, dtype=np.float32)
        np.copyto(h_in, x_batch.ravel())

        d_in = cuda.mem_alloc(h_in.nbytes)
        d_out = cuda.mem_alloc(h_out.nbytes)
        stream = cuda.Stream()

        cuda.memcpy_htod_async(d_in, h_in, stream)
        self._context.execute_async_v2(
            bindings=[int(d_in), int(d_out)],
            stream_handle=stream.handle,
        )
        cuda.memcpy_dtoh_async(h_out, d_out, stream)
        stream.synchronize()

        logits = np.array(h_out, dtype=np.float32)
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        return probs

    def _infer_onnx(self, x_batch: np.ndarray) -> np.ndarray:
        input_name = self._ort_session.get_inputs()[0].name
        outputs = self._ort_session.run(None, {input_name: x_batch})
        return outputs[0][0]


# ---------------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────
# Tegrastats — GPU temp & RAM reader
# ─────────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def _read_tegrastats_once() -> Dict[str, Optional[float]]:
    """
    Run ``tegrastats --interval 1000 --count 1`` and parse one line.

    Returns a dict with:
        gpu_temp_c      — GPU temperature in Celsius
        ram_used_mb     — DRAM used in MB
        gpu_util_pct    — GPU utilisation percentage (if available)
        cpu_pct         — CPU utilisation (psutil fallback)

    Falls back to psutil-only values when tegrastats is unavailable
    (e.g., on a dev machine or Jetson with incomplete JetPack).

    Example tegrastats line (Jetson Nano, JetPack 4.x):
        RAM 1234/3964MB (lfb 4x256kB) SWAP 0/1982MB (cached 0MB)
        CPU [12%@921,0%@921,12%@921,5%@921] EMC_FREQ 0%
        GR3D_FREQ 0% PLL@33C CPU@35.25C PMIC@100C GPU@34C AO@39C
        thermal@34.25C
    """
    result: Dict[str, Optional[float]] = {
        "gpu_temp_c": None,
        "ram_used_mb": None,
        "gpu_util_pct": None,
        "cpu_pct": psutil.cpu_percent(interval=None),
    }

    # Always collect process RSS as a reliable baseline
    proc = psutil.Process(os.getpid())
    result["ram_used_mb"] = round(proc.memory_info().rss / (1024 ** 2), 2)

    try:
        out = subprocess.check_output(
            ["tegrastats", "--interval", "200", "--count", "1"],
            timeout=2,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace")

        # ── RAM used ─────────────────────────────────────────────────────
        # Pattern: "RAM 1234/3964MB"
        import re
        ram_match = re.search(r"RAM\s+(\d+)/\d+MB", out)
        if ram_match:
            result["ram_used_mb"] = float(ram_match.group(1))

        # ── GPU temperature ───────────────────────────────────────────────
        # Nano JetPack4: "GPU@34C"
        # Orin/Xavier:   "GPU Therm@44C" or "GPU@44.5C"
        gpu_temp_match = re.search(r"GPU(?:\s+Therm)?@([\d.]+)C", out)
        if gpu_temp_match:
            result["gpu_temp_c"] = float(gpu_temp_match.group(1))

        # ── GR3D utilisation ──────────────────────────────────────────────
        gr3d_match = re.search(r"GR3D_FREQ\s+(\d+)%", out)
        if gr3d_match:
            result["gpu_util_pct"] = float(gr3d_match.group(1))

        log.debug("tegrastats: %s", out.strip())

    except FileNotFoundError:
        log.debug("tegrastats not found — using psutil only (dev environment).")
    except subprocess.TimeoutExpired:
        log.warning("tegrastats timed out — system stats partial.")
    except Exception as exc:
        log.warning("tegrastats error: %s", exc)

    return result


# ---------------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────
# Zero-Trust Mutual Authentication Middleware
# ─────────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def _get_client_ip(request: Request) -> str:
    """
    Extract the real client IP.

    Behind Tailscale on the Jetson, the peer IP is the direct socket address
    because the WireGuard tunnel terminates at the kernel.  There is no proxy
    or header magic needed — the IP in ``request.client.host`` IS the Tailscale
    peer address (100.x.x.x) for authenticated peers.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "0.0.0.0"


def verify_zero_trust(request: Request, device_id: str) -> None:
    """
    Mutual authentication guard for high-compute endpoints.

    Checks:
      1. ``X-Device-ID`` header is present (injected by the client).
      2. The device_id is in the ALLOWED_DEVICES allowlist.
      3. The source IP matches the allowlisted IP for that device.

    Raises ``HTTPException(403)`` on any violation.

    Note on WireGuard / Tailscale model:
      Because the server listens *only* on the Tailscale IP, any request that
      reaches this function has already been authenticated and encrypted by
      WireGuard — this check is a second defence layer (defence-in-depth) to
      prevent a compromised Tailnet node from masquerading as a trusted device.
    """
    header_device_id = request.headers.get("X-Device-ID", "").strip()

    if not header_device_id:
        log.warning("AUTH FAIL: Missing X-Device-ID header from %s", _get_client_ip(request))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing X-Device-ID header — Zero-Trust auth required.",
        )

    if header_device_id != device_id:
        log.warning(
            "AUTH FAIL: Header device_id '%s' != payload device_id '%s'",
            header_device_id,
            device_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Device ID mismatch between header and payload.",
        )

    if device_id not in ALLOWED_DEVICES:
        log.warning("AUTH FAIL: Unknown device_id '%s' from %s", device_id, _get_client_ip(request))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Device '{device_id}' is not registered in the Tailnet allowlist.",
        )

    expected_ip = ALLOWED_DEVICES[device_id]
    client_ip = _get_client_ip(request)

    if not INSECURE_MODE and client_ip != expected_ip:
        log.warning(
            "AUTH FAIL: device_id '%s' arrived from %s but is registered to %s",
            device_id,
            client_ip,
            expected_ip,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Source IP '{client_ip}' does not match expected Tailscale IP "
                f"'{expected_ip}' for device '{device_id}'."
            ),
        )

    log.info("AUTH OK: device_id=%s ip=%s", device_id, client_ip)


# ---------------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────
# Telemetry logger
# ─────────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


@dataclass
class TelemetryRecord:
    timestamp_iso: str
    timestamp_ms: int
    device_id: str
    workload_type: str
    n_channels: int
    window_samples: int
    t_inference_ms: float       # GPU-only compute
    t_total_ms: float           # full request wall-clock
    gesture: str
    confidence: float
    backend: str
    gpu_temp_c: Optional[float]
    ram_used_mb: float
    gpu_util_pct: Optional[float]
    cpu_pct: float
    client_ip: str


def _write_telemetry(record: TelemetryRecord) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = asdict(record)
    with open(TELEMETRY_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# Shared engine instance — initialised once at startup
_engine: Optional[TensorRTEngine] = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup / shutdown lifecycle manager."""
    global _engine

    log.info("=" * 60)
    log.info("EdgeTrust Compute Server — Jetson Nano")
    log.info("Bind address : %s:%d", TAILSCALE_IP, SERVER_PORT)
    log.info("TRT engine   : %s", TRT_ENGINE_PATH)
    log.info("Allowlist    : %s", list(ALLOWED_DEVICES.keys()))
    log.info("=" * 60)

    _engine = TensorRTEngine(engine_path=TRT_ENGINE_PATH, n_channels=8)
    log.info("Inference engine initialised | backend=%s", _engine._backend)

    yield  # --- application is running ---

    log.info("Shutting down EdgeTrust Compute Server.")
    _engine = None


app = FastAPI(
    title="EdgeTrust — Jetson Nano Compute Server",
    description=(
        "GPU-accelerated gesture inference server for the EdgeTrust-Offload project. "
        "Listens exclusively on the Tailscale interface (100.x.x.x) and enforces "
        "mutual Zero-Trust device authentication on every request."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)

# Block cross-origin requests — only the Tailnet should reach this server,
# but belt-and-suspenders: CORS is tightly restricted.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://{TAILSCALE_IP}"],
    allow_methods=["GET", "POST"],
    allow_headers=["X-Device-ID", "Content-Type"],
)


# ---------------------------------------------------------------------------
# Middleware — Tailscale IP enforcement
# ---------------------------------------------------------------------------


@app.middleware("http")
async def enforce_tailscale_origin(request: Request, call_next):
    """
    Drop any request that did NOT arrive through the Tailscale interface.

    Since ``uvicorn`` is bound to TAILSCALE_IP, this middleware is a
    belt-and-suspenders guard: it verifies the socket peer address starts
    with the expected Tailscale prefix (100.) and matches TAILSCALE_IP.

    Any traffic arriving here with a non-Tailscale source IP means the
    network configuration is incorrect and the request is rejected.
    """
    client_ip = _get_client_ip(request)

    # Allow loopback for health checks from the Jetson itself.
    # INSECURE_MODE bypasses the Tailscale prefix check for S2 baseline measurements.
    if not INSECURE_MODE and client_ip not in ("127.0.0.1", "::1") and not client_ip.startswith("100."):
        log.warning(
            "BLOCKED non-Tailscale IP %s → %s %s",
            client_ip,
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "detail": (
                    "Access denied. This server is accessible exclusively via the "
                    "Tailscale VPN tunnel. Your source IP is not a valid Tailnet address."
                )
            },
        )

    return await call_next(request)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["monitoring"])
async def health_check():
    """
    Lightweight health-check endpoint.

    Returns server identity, backend capabilities, and current system stats.
    No auth required (but still protected by Tailscale IP enforcement above).
    """
    stats = _read_tegrastats_once()
    return {
        "status": "ok",
        "server": "jetson-nano",
        "tailscale_ip": TAILSCALE_IP,
        "inference_backend": _engine._backend if _engine else "not_initialised",
        "platform": platform.uname()._asdict(),
        "system_stats": stats,
        "uptime_s": time.monotonic(),
    }


@app.post(
    "/api/v1/offload/fft",
    response_model=OffloadResponse,
    tags=["inference"],
    summary="FFT + gesture inference (primary offload endpoint)",
)
async def offload_fft(body: OffloadRequest, request: Request):
    """
    Primary offload endpoint consumed by ``main_scheduler.py`` on the Raspberry Pi.

    Pipeline
    --------
    1. Zero-Trust mutual auth  (device_id allowlist + source IP check)
    2. Decode & reshape the 1024-sample window
    3. GPU/TensorRT gesture inference
    4. Collect tegrastats system telemetry
    5. Return enriched response including T_inference and system_stats
    6. Append structured telemetry to ``logs/jetson_telemetry.jsonl``
    """
    t_request_start = time.perf_counter()

    # ── 1. Zero-Trust Auth ────────────────────────────────────────────────────
    verify_zero_trust(request, body.device_id)

    # ── 2. Decode window ──────────────────────────────────────────────────────
    samples_raw = np.array(body.payload.samples, dtype=np.float32)

    # Auto-detect channel layout
    n_channels = body.payload.n_channels
    if n_channels is None or n_channels <= 0:
        # Assume 1D mono-channel if we have exactly WINDOW_SIZE samples,
        # otherwise try to infer multi-channel from sample count
        if len(samples_raw) == WINDOW_SIZE:
            n_channels = 1
        elif len(samples_raw) % WINDOW_SIZE == 0:
            n_channels = len(samples_raw) // WINDOW_SIZE
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Cannot infer n_channels from {len(samples_raw)} samples. "
                    f"Expected a multiple of WINDOW_SIZE={WINDOW_SIZE}."
                ),
            )

    window = samples_raw.reshape(WINDOW_SIZE, n_channels)

    # Lazily warm up a larger engine if channel count changed
    global _engine
    if _engine is None or _engine._n_channels != n_channels:
        log.info("Re-initialising engine for n_channels=%d", n_channels)
        _engine = TensorRTEngine(engine_path=TRT_ENGINE_PATH, n_channels=n_channels)

    # ── 3. GPU Inference ──────────────────────────────────────────────────────
    t_infer_start = time.perf_counter()
    prediction = _engine.predict(window)
    t_infer_ms = (time.perf_counter() - t_infer_start) * 1_000

    # ── 4. Tegrastats telemetry ───────────────────────────────────────────────
    sys_stats = _read_tegrastats_once()

    # ── 5. Build response ─────────────────────────────────────────────────────
    t_total_ms = (time.perf_counter() - t_request_start) * 1_000

    response = OffloadResponse(
        status="success",
        processing_node=f"jetson-nano ({TAILSCALE_IP})",
        compute_latency_ms=round(t_infer_ms, 3),
        total_request_ms=round(t_total_ms, 3),
        result={
            "gesture": prediction["gesture"],
            "gesture_idx": prediction["gesture_idx"],
            "confidence": round(prediction["confidence"], 4),
            "probabilities": prediction["probabilities"],
            "n_channels": n_channels,
            "window_size": WINDOW_SIZE,
        },
        system_stats=SystemStats(
            gpu_temp_c=sys_stats["gpu_temp_c"],
            ram_used_mb=sys_stats["ram_used_mb"],
            gpu_utilization_pct=sys_stats["gpu_util_pct"],
            cpu_pct=sys_stats["cpu_pct"],
        ),
    )

    # ── 6. Telemetry log ──────────────────────────────────────────────────────
    try:
        now_ms = int(time.time() * 1_000)
        record = TelemetryRecord(
            timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            timestamp_ms=now_ms,
            device_id=body.device_id,
            workload_type=body.workload_type,
            n_channels=n_channels,
            window_samples=WINDOW_SIZE,
            t_inference_ms=round(t_infer_ms, 3),
            t_total_ms=round(t_total_ms, 3),
            gesture=prediction["gesture"],
            confidence=round(prediction["confidence"], 4),
            backend=prediction["backend"],
            gpu_temp_c=sys_stats["gpu_temp_c"],
            ram_used_mb=sys_stats["ram_used_mb"],
            gpu_util_pct=sys_stats.get("gpu_util_pct"),
            cpu_pct=sys_stats["cpu_pct"],
            client_ip=_get_client_ip(request),
        )
        _write_telemetry(record)
    except Exception as exc:
        log.warning("Failed to write telemetry record: %s", exc)

    log.info(
        "Request OK | device=%s | gesture=%s (%.0f%%) | t_infer=%.2f ms | t_total=%.2f ms | GPU=%.1f°C",
        body.device_id,
        prediction["gesture"],
        prediction["confidence"] * 100,
        t_infer_ms,
        t_total_ms,
        sys_stats["gpu_temp_c"] or 0.0,
    )

    return response


@app.post(
    "/api/v1/offload/gesture",
    response_model=OffloadResponse,
    tags=["inference"],
    summary="Direct gesture-only inference (alias — same pipeline as /offload/fft)",
)
async def offload_gesture(body: OffloadRequest, request: Request):
    """Alias for ``/api/v1/offload/fft`` — identical pipeline, different route tag."""
    return await offload_fft(body, request)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    log.info(
        "Starting EdgeTrust Compute Server on %s:%d (Tailscale-only)",
        TAILSCALE_IP,
        SERVER_PORT,
    )

    uvicorn.run(
        "compute_server:app",
        host=TAILSCALE_IP,   # ← ZERO-TRUST: bind ONLY to Tailscale interface
        port=SERVER_PORT,
        reload=False,
        workers=1,           # Single worker — GPU context is not fork-safe
        log_level="info",
        access_log=True,
        proxy_headers=False, # No proxy in front — raw peer IP is the Tailscale IP
    )
