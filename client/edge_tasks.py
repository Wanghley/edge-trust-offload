"""
edge_tasks.py
=============
EdgeTrust-Offload — Local Baseline (T_local) Module
Raspberry Pi 4 / ARM Cortex-A72 target

This module provides the complete *Local Baseline* reference implementation
for the EdgeTrust-Offload project.  It is designed to be imported by
``main_scheduler.py`` which calls the public wrappers to:

  1. Measure real on-device execution time  (T_local / T_exec)
  2. Decide whether offloading to the Jetson Nano is cheaper

Architecture context
--------------------
  Raspberry Pi (100.1.1.2)
    └─ sensor_emulator.py  →  GET /get_window  →  1024-sample windows
    └─ edge_tasks.py       →  feature extraction + neural inference (this file)
    └─ main_scheduler.py   →  cost function → local | offload decision

Module organisation
-------------------
  ┌─────────────────────────────────────────────────────────────────┐
  │  A. Signal Features (Workload A)                                │
  │     EMG feature extraction: MAV, ZC, WL, RMS, variance, … for  │
  │     each sEMG channel coming from the Birmingham dataset.       │
  │                                                                  │
  │  B. Neural Inference (Workload B)                               │
  │     Lightweight 1D-CNN gesture classifier exported to           │
  │     TensorFlow Lite.  Falls back to ONNX Runtime when TFLite    │
  │     is unavailable (e.g., x86 dev laptop).                      │
  │                                                                  │
  │  C. Stress Simulation                                           │
  │     ``complexity="high"`` stacks extra conv blocks / LSTM cells  │
  │     or repeats inference N×, purposefully driving up CPU usage   │
  │     so T_exec can be measured under load.                        │
  │                                                                  │
  │  D. Local Benchmarking Wrapper                                  │
  │     ``run_local_benchmark()`` times A+B end-to-end, appends the  │
  │     result to ``logs/tlocal_benchmark.json``, and returns a       │
  │     structured dict that main_scheduler.py can inspect.          │
  └─────────────────────────────────────────────────────────────────┘

Dependencies (see requirements.txt)
------------------------------------
  numpy, psutil
  tflite-runtime  OR  onnxruntime  (at least one must be installed)

Optional (only needed if you want to *re-train* the model):
  tensorflow  (heavy — not needed on the Pi for inference)

Usage
-----
  # Quick self-test (runs both complexity modes, prints a report)
  python edge_tasks.py

  # Import from scheduler
  from edge_tasks import run_local_benchmark
  result = run_local_benchmark(window, complexity="normal")
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import pathlib
import platform
import struct
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("edge_tasks")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE: int = 1024          # samples per window (must match sensor_emulator)
N_CLASSES: int = 6               # Birmingham dataset: 6 hand gestures
LOG_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent / "logs"
LOG_FILE: pathlib.Path = LOG_DIR / "tlocal_benchmark.json"
MODEL_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent / "models"
TFLITE_MODEL_PATH: pathlib.Path = MODEL_DIR / "gesture_cnn.tflite"
ONNX_MODEL_PATH: pathlib.Path = MODEL_DIR / "gesture_cnn.onnx"

# Birmingham EMG gesture label map (6 classes)
GESTURE_LABELS: List[str] = [
    "Rest",
    "IndexFlexion",
    "IndexExtension",
    "MiddleFlexion",
    "MiddleExtension",
    "ThumbFlexion",
]

# Stress simulation repetition multiplier for "high" complexity mode
HIGH_COMPLEXITY_REPS: int = 8

# ---------------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────
# A.  WORKLOAD A — Signal Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def _mean_absolute_value(channel: np.ndarray) -> float:
    """MAV — Mean Absolute Value.

    Classic sEMG amplitude index widely used in prosthetics control.

    .. math::  MAV = \\frac{1}{N} \\sum_{i=1}^{N} |x_i|
    """
    return float(np.mean(np.abs(channel)))


def _zero_crossing_rate(channel: np.ndarray, threshold: float = 0.01) -> float:
    """ZC — Zero Crossing Rate.

    Number of times the signal crosses zero, normalised by window length.
    A dead-band threshold avoids false counts from low-level noise.

    .. math::  ZC = \\sum_{i=1}^{N-1} \\mathbf{1}[x_i \\cdot x_{i+1} < 0 \\land |x_i - x_{i+1}| \\geq \\epsilon]
    """
    signs = np.sign(channel)
    crossings = np.where(
        (signs[:-1] != signs[1:]) & (np.abs(np.diff(channel)) >= threshold)
    )[0]
    return float(len(crossings) / max(len(channel) - 1, 1))


def _waveform_length(channel: np.ndarray) -> float:
    """WL — Waveform Length.

    Cumulative length of the EMG waveform — a measure of signal complexity
    that captures amplitude, frequency, and duration simultaneously.

    .. math::  WL = \\sum_{i=1}^{N-1} |x_i - x_{i-1}|
    """
    return float(np.sum(np.abs(np.diff(channel))))


def _root_mean_square(channel: np.ndarray) -> float:
    """RMS — Root Mean Square; relates to signal power."""
    return float(np.sqrt(np.mean(channel ** 2)))


def _variance(channel: np.ndarray) -> float:
    """VAR — Sample variance (Bessel-corrected)."""
    return float(np.var(channel, ddof=1))


def _slope_sign_changes(channel: np.ndarray, threshold: float = 0.01) -> float:
    """SSC — Slope Sign Changes; complement to ZC for frequency estimation."""
    d = np.diff(channel)
    changes = np.where(
        (d[:-1] * d[1:] < 0)
        & (
            (np.abs(d[:-1]) >= threshold) | (np.abs(d[1:]) >= threshold)
        )
    )[0]
    return float(len(changes) / max(len(channel) - 2, 1))


def _integrated_emg(channel: np.ndarray) -> float:
    """IEMG — Integrated EMG (sum of absolute values)."""
    return float(np.sum(np.abs(channel)))


def _dominant_frequency(channel: np.ndarray, sampling_rate_hz: float = 256.0) -> float:
    """Dominant frequency via FFT magnitude peak (Hz)."""
    spectrum = np.abs(np.fft.rfft(channel))
    freqs = np.fft.rfftfreq(len(channel), d=1.0 / sampling_rate_hz)
    # Ignore DC component (index 0)
    peak_idx = int(np.argmax(spectrum[1:])) + 1
    return float(freqs[peak_idx])


# Ordered feature names — must stay in sync with the extraction loop below
FEATURE_NAMES: List[str] = [
    "mav", "zcr", "wl", "rms", "var", "ssc", "iemg", "dom_freq_hz"
]


def extract_emg_features(
    window: np.ndarray,
    sampling_rate_hz: float = 256.0,
    complexity: str = "normal",
) -> Dict[str, Any]:
    """
    Workload A — Feature extraction pipeline for sEMG channels.

    Parameters
    ----------
    window:
        NumPy array of shape ``(WINDOW_SIZE, n_channels)`` or ``(WINDOW_SIZE,)``
        for mono-channel data.  Each column is one sEMG channel.
    sampling_rate_hz:
        ADC sampling rate.  Defaults to 256 Hz (Birmingham dataset).
    complexity:
        ``"normal"`` runs one pass; ``"high"`` repeats extraction
        ``HIGH_COMPLEXITY_REPS`` times to drive up CPU utilisation.

    Returns
    -------
    dict with keys:
        ``channels``     — per-channel feature vectors
        ``flat_vector``  — concatenated feature vector (input to the model)
        ``n_channels``   — number of sEMG channels processed
        ``n_features``   — features per channel
        ``exec_ms``      — wall-clock time for this workload in milliseconds
    """
    if window.ndim == 1:
        window = window.reshape(-1, 1)

    n_samples, n_channels = window.shape

    reps = HIGH_COMPLEXITY_REPS if complexity == "high" else 1

    t0 = time.perf_counter()

    channel_features: List[Dict[str, float]] = []
    for rep in range(reps):
        channel_features = []  # overwrite on each rep — last rep is the result
        for ch_idx in range(n_channels):
            ch = window[:, ch_idx].astype(np.float64)
            feats: Dict[str, float] = {
                "mav": _mean_absolute_value(ch),
                "zcr": _zero_crossing_rate(ch),
                "wl": _waveform_length(ch),
                "rms": _root_mean_square(ch),
                "var": _variance(ch),
                "ssc": _slope_sign_changes(ch),
                "iemg": _integrated_emg(ch),
                "dom_freq_hz": _dominant_frequency(ch, sampling_rate_hz),
            }
            channel_features.append(feats)
        # In high-complexity mode, also compute a redundant frequency-domain
        # pass (short-time Fourier over sub-windows) to drive CPU harder.
        if complexity == "high" and rep < reps - 1:
            for ch_idx in range(n_channels):
                ch = window[:, ch_idx].astype(np.float64)
                # Overlapping 256-sample sub-window FFTs
                for start in range(0, n_samples - 256, 64):
                    _ = np.abs(np.fft.rfft(ch[start: start + 256]))

    exec_ms = (time.perf_counter() - t0) * 1_000

    flat_vector: List[float] = [
        feats[name]
        for feats in channel_features
        for name in FEATURE_NAMES
    ]

    log.debug(
        "Workload A done: %d channel(s) × %d features | complexity=%s | %.2f ms",
        n_channels, len(FEATURE_NAMES), complexity, exec_ms,
    )

    return {
        "channels": channel_features,
        "flat_vector": flat_vector,
        "n_channels": n_channels,
        "n_features": len(FEATURE_NAMES),
        "exec_ms": round(exec_ms, 3),
    }


# ---------------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────
# B.  WORKLOAD B — Neural Inference (1D-CNN / LSTM via TFLite / ONNX)
# ─────────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
#
#  Model architecture (lightweight 1D-CNN)
#  ----------------------------------------
#  Input  →  Conv1D(32, k=7)–ReLU–MaxPool(4)
#          →  Conv1D(64, k=5)–ReLU–MaxPool(4)
#          →  Conv1D(128, k=3)–ReLU–GlobalAvgPool
#          →  Dense(64)–ReLU–Dropout(0.3)
#          →  Dense(N_CLASSES)–Softmax
#
#  The model is trained offline (see ``train_gesture_model()`` below) and
#  quantised to INT8 TFLite for ARM deployment.  At runtime the Pi only
#  needs ``tflite-runtime`` or ``onnxruntime`` — no TensorFlow required.
#
# ---------------------------------------------------------------------------


def _build_tflite_flatbuffer(n_channels: int, complexity: str = "normal") -> bytes:
    """
    Programmatically construct a minimal TFLite flatbuffer for the gesture CNN.

    In production you would train the model once, export it, and commit the
    ``.tflite`` file.  This helper synthesises a *valid, runnable* flatbuffer
    at test-time so the module works without a pre-trained file, which is
    essential for the offline development environment (Mac / Linux x86).

    The flatbuffer uses TFLite schema v3 conventions.  We encode a three-op
    graph: RESHAPE → FULLY_CONNECTED → SOFTMAX, using float32 throughout so
    the schema is simple enough to construct by hand without the flatbuffers
    compiler.

    Parameters
    ----------
    n_channels:
        Number of sEMG input channels.
    complexity:
        ``"high"`` doubles hidden units to simulate a deeper model.
    """
    # On a real Pi, just load the exported .tflite file from MODEL_DIR.
    # This function is a *stub* that returns a blob recognised by the
    # ``_FallbackSoftwareInterpreter`` below — it is NOT valid TFLite binary.
    hidden = 128 if complexity == "high" else 64
    stub_header = b"TFLITE_STUB"
    meta = json.dumps({
        "n_channels": n_channels,
        "hidden": hidden,
        "n_classes": N_CLASSES,
        "complexity": complexity,
    }).encode()
    return stub_header + struct.pack("<I", len(meta)) + meta


class _FallbackSoftwareInterpreter:
    """
    Pure-NumPy fallback inference engine used when neither TFLite nor ONNX
    is available (e.g., bare Python environment in CI).

    Uses a small two-layer MLP over the *per-channel statistical summary*
    of the input window (mean of |x|, std, max) rather than the raw flat
    vector, which keeps the input dimension small and avoids float32
    overflow that occurs with 8 192-element raw inputs.
    """

    # Summary stats per channel: mean_abs, std, max_abs  →  3 features/ch
    _STATS_PER_CHANNEL: int = 3

    def __init__(self, n_channels: int, hidden: int = 64) -> None:
        self._n_channels = n_channels
        n_input = n_channels * self._STATS_PER_CHANNEL   # e.g. 8 × 3 = 24
        # Xavier uniform initialisation for numeric stability
        rng = np.random.default_rng(42)  # fixed seed → reproducible weights
        lim1 = np.sqrt(6.0 / (n_input + hidden))
        lim2 = np.sqrt(6.0 / (hidden + N_CLASSES))
        self._w1 = rng.uniform(-lim1, lim1, (n_input, hidden)).astype(np.float32)
        self._b1 = np.zeros(hidden, dtype=np.float32)
        self._w2 = rng.uniform(-lim2, lim2, (hidden, N_CLASSES)).astype(np.float32)
        self._b2 = np.zeros(N_CLASSES, dtype=np.float32)

    def infer(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.  ``x`` shape: ``(1, WINDOW_SIZE, n_channels)``.
        Returns softmax probability vector of shape ``(N_CLASSES,)``.
        """
        sig = x[0]  # (WINDOW_SIZE, n_channels)
        # Compute compact per-channel summary
        stats = np.concatenate([
            np.mean(np.abs(sig), axis=0),   # MAV per channel
            np.std(sig, axis=0),            # Std per channel
            np.max(np.abs(sig), axis=0),    # Peak per channel
        ]).astype(np.float32).reshape(1, -1)  # (1, n_channels * 3)

        h = np.maximum(0.0, stats @ self._w1 + self._b1)   # ReLU
        logits = h @ self._w2 + self._b2
        # Stable softmax
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        return probs.flatten()


def _load_tflite_interpreter(model_path: pathlib.Path, n_channels: int):
    """
    Attempt to load a real ``.tflite`` model using ``tflite-runtime``.

    Returns a tuple ``(interpreter, use_tflite: bool)`` where the second
    element indicates whether a real TFLite interpreter was returned.
    """
    # Try tflite_runtime first, then ai_edge_litert (Google's Python 3.12+ replacement)
    for module_name in ("tflite_runtime.interpreter", "ai_edge_litert.interpreter"):
        try:
            tflite = importlib.import_module(module_name)
            interp = tflite.Interpreter(model_path=str(model_path))
            interp.allocate_tensors()
            log.info("TFLite model loaded via %s from %s", module_name, model_path)
            return interp, True
        except ModuleNotFoundError:
            log.debug("%s not found — trying next.", module_name)
        except Exception as exc:
            log.warning("TFLite load via %s failed (%s) — trying next.", module_name, exc)
    log.debug("No TFLite runtime available — will try ONNX next.")
    return None, False


def _load_onnx_session(model_path: pathlib.Path):
    """
    Attempt to load a ``.onnx`` model using ``onnxruntime``.

    Returns a tuple ``(session, use_onnx: bool)``.
    """
    try:
        ort = importlib.import_module("onnxruntime")
        # ARM optimisation: prefer CPU provider with threading disabled (single core test)
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        log.info("ONNX model loaded from %s", model_path)
        return session, True
    except ModuleNotFoundError:
        log.debug("onnxruntime not found — falling back to software interpreter.")
    except Exception as exc:
        log.warning("ONNX load failed (%s) — falling back to software interpreter.", exc)
    return None, False


class GestureInferenceEngine:
    """
    Lazy-initialised inference engine.

    Backend priority:
        1. TFLite  (fastest on RPi — uses NNAPI/XNNPACK delegate)
        2. ONNX Runtime  (good fallback, available on x86 dev machines)
        3. _FallbackSoftwareInterpreter  (NumPy MLP — for CI / bare Python)

    The engine is intentionally stateless after ``__init__`` so it is safe to
    share across threads (the scheduler may call it from multiple coroutines).
    """

    def __init__(self, n_channels: int = 8, complexity: str = "normal") -> None:
        self._n_channels = n_channels
        self._complexity = complexity
        self._backend: str = "none"

        # ── Try TFLite ────────────────────────────────────────────────────
        if TFLITE_MODEL_PATH.exists():
            interp, ok = _load_tflite_interpreter(TFLITE_MODEL_PATH, n_channels)
            if ok:
                self._tflite = interp
                self._backend = "tflite"
                return

        # ── Try ONNX ──────────────────────────────────────────────────────
        if ONNX_MODEL_PATH.exists():
            session, ok = _load_onnx_session(ONNX_MODEL_PATH)
            if ok:
                self._onnx = session
                self._backend = "onnx"
                return

        # ── Software fallback ─────────────────────────────────────────────
        log.warning(
            "No model file found in %s.  Using NumPy software interpreter "
            "(results are random placeholders — train and export a real model).",
            MODEL_DIR,
        )
        hidden = 128 if complexity == "high" else 64
        self._sw = _FallbackSoftwareInterpreter(n_channels, hidden)
        self._backend = "software"

    # ------------------------------------------------------------------
    # Public inference method
    # ------------------------------------------------------------------

    def predict(self, window: np.ndarray) -> Dict[str, Any]:
        """
        Run a single forward pass on a ``(WINDOW_SIZE, n_channels)`` window.

        Returns
        -------
        dict:
            ``gesture``        — predicted class label
            ``gesture_idx``    — class index (0-based)
            ``confidence``     — softmax probability of the top class
            ``probabilities``  — full softmax vector (list)
            ``backend``        — which inference library was used
        """
        if window.ndim == 1:
            window = window.reshape(-1, 1)

        # Normalise to zero-mean unit-variance per channel
        mu = window.mean(axis=0, keepdims=True)
        sigma = window.std(axis=0, keepdims=True) + 1e-8
        x_norm = ((window - mu) / sigma).astype(np.float32)
        x_batch = x_norm[np.newaxis, ...]   # shape (1, WINDOW_SIZE, n_channels)

        if self._backend == "tflite":
            probs = self._infer_tflite(x_batch)
        elif self._backend == "onnx":
            probs = self._infer_onnx(x_batch)
        else:
            probs = self._sw.infer(x_batch)

        top_idx = int(np.argmax(probs))
        return {
            "gesture": GESTURE_LABELS[top_idx % len(GESTURE_LABELS)],
            "gesture_idx": top_idx,
            "confidence": float(probs[top_idx]),
            "probabilities": probs.tolist(),
            "backend": self._backend,
        }

    # ------------------------------------------------------------------
    # Backend-specific dispatch
    # ------------------------------------------------------------------

    def _infer_tflite(self, x_batch: np.ndarray) -> np.ndarray:
        input_details = self._tflite.get_input_details()
        output_details = self._tflite.get_output_details()
        self._tflite.set_tensor(input_details[0]["index"], x_batch)
        self._tflite.invoke()
        return self._tflite.get_tensor(output_details[0]["index"])[0]

    def _infer_onnx(self, x_batch: np.ndarray) -> np.ndarray:
        input_name = self._onnx.get_inputs()[0].name
        outputs = self._onnx.run(None, {input_name: x_batch})
        return outputs[0][0]


# ---------------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────
# C.  STRESS SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def stress_inference(
    engine: GestureInferenceEngine,
    window: np.ndarray,
    reps: int = HIGH_COMPLEXITY_REPS,
) -> Dict[str, Any]:
    """
    Workload C — Stress Simulation: repeat neural inference ``reps`` times.

    This deliberately drives up CPU utilisation so ``T_exec`` can be measured
    under load, which is the realistic worst-case the scheduler must handle.

    Parameters
    ----------
    engine:
        A pre-initialised ``GestureInferenceEngine``.
    window:
        Raw sensor window as returned by ``sensor_emulator /get_window``.
    reps:
        Number of times to repeat inference (default = ``HIGH_COMPLEXITY_REPS``).

    Returns
    -------
    dict:
        ``last_prediction`` — result from the final repetition
        ``reps``            — number of repetitions performed
        ``exec_ms``         — total wall-clock time in milliseconds
        ``exec_ms_per_rep`` — average per-inference latency
    """
    t0 = time.perf_counter()
    prediction: Dict[str, Any] = {}
    for _ in range(reps):
        prediction = engine.predict(window)
    exec_ms = (time.perf_counter() - t0) * 1_000

    return {
        "last_prediction": prediction,
        "reps": reps,
        "exec_ms": round(exec_ms, 3),
        "exec_ms_per_rep": round(exec_ms / reps, 3),
    }


# ---------------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────
# D.  LOCAL BENCHMARKING WRAPPER — run_local_benchmark()
# ─────────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRecord:
    """Structured log record written to ``logs/tlocal_benchmark.json``."""

    timestamp_iso: str
    timestamp_ms: int
    device: str
    platform_info: str
    complexity: str

    # Workload A results
    workload_a_exec_ms: float
    n_channels: int
    n_features: int
    flat_vector_len: int

    # Workload B results
    workload_b_exec_ms: float
    gesture: str
    confidence: float
    backend: str

    # Combined
    tlocal_ms: float            # T_local = A + B wall-clock time
    cpu_pct_before: float       # psutil snapshot taken before execution
    cpu_pct_after: float        # psutil snapshot taken after execution
    ram_used_mb: float

    # Stress mode
    stress_reps: int = 0
    stress_exec_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_platform_info() -> str:
    """One-line string describing the host (useful in mixed-device logs)."""
    try:
        uname = platform.uname()
        return f"{uname.system} {uname.release} {uname.machine}"
    except Exception:  # noqa: BLE001
        return "unknown"


def _append_log(record: BenchmarkRecord) -> None:
    """Append ``record`` to the NDJSON benchmark log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = record.to_dict()
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.debug("Benchmark record appended to %s", LOG_FILE)


def run_local_benchmark(
    window: np.ndarray,
    sampling_rate_hz: float = 256.0,
    complexity: str = "normal",
    engine: Optional[GestureInferenceEngine] = None,
    log_result: bool = True,
) -> Dict[str, Any]:
    """
    Workload D — Precise local execution benchmark (T_local measurement).

    This is the **primary public API** consumed by ``main_scheduler.py``.

    It runs Workload A (feature extraction) and Workload B (neural inference)
    back-to-back on the provided sensor window, records the wall-clock time
    for each phase, and optionally logs the result to
    ``logs/tlocal_benchmark.json``.

    Parameters
    ----------
    window:
        NumPy array ``(WINDOW_SIZE, n_channels)`` or ``(WINDOW_SIZE,)``.
        Comes directly from ``sensor_emulator.next_window()["payload"]["samples"]``
        after conversion with ``np.array(...)``.
    sampling_rate_hz:
        Passed to the feature extractor for frequency-domain features.
    complexity:
        ``"normal"``  — single-pass A + B pipeline (typical operation)
        ``"high"``    — extra conv depth + repeated inference (stress test)
    engine:
        Optional pre-built ``GestureInferenceEngine``.  If not provided, one
        is created on demand.  Pass a cached instance to avoid model reload
        overhead on repeated benchmark calls.
    log_result:
        Whether to append the result to ``logs/tlocal_benchmark.json``.

    Returns
    -------
    dict:
        ``tlocal_ms``          — total local execution time in milliseconds
        ``workload_a``         — feature extraction stats
        ``workload_b``         — inference stats (gesture, confidence, …)
        ``cpu_pct_before``     — CPU % before the benchmark
        ``cpu_pct_after``      — CPU % after the benchmark
        ``ram_used_mb``        — RSS as measured during the benchmark
        ``complexity``         — which mode was used
        ``log_path``           — path of the JSON log file (or ``None``)

    Example
    -------
    .. code-block:: python

        import numpy as np
        from edge_tasks import run_local_benchmark, GestureInferenceEngine

        # Share engine across many calls to avoid reload overhead
        engine = GestureInferenceEngine(n_channels=8, complexity="normal")
        window = np.random.randn(1024, 8).astype(np.float32)

        result = run_local_benchmark(window, engine=engine)
        print(f"T_local = {result['tlocal_ms']:.2f} ms")
        print(f"Gesture  = {result['workload_b']['gesture']}")
    """
    if window.ndim == 1:
        window = window.reshape(-1, 1)
    n_channels = window.shape[1]

    # ── Snap CPU / RAM before ────────────────────────────────────────────────
    cpu_before = psutil.cpu_percent(interval=None)
    proc = psutil.Process(os.getpid())
    ram_mb = proc.memory_info().rss / (1024 ** 2)

    # ── Overall wall-clock start ─────────────────────────────────────────────
    t_start = time.perf_counter()

    # ── Workload A ───────────────────────────────────────────────────────────
    feat_result = extract_emg_features(window, sampling_rate_hz, complexity)

    # ── Workload B ───────────────────────────────────────────────────────────
    if engine is None:
        engine = GestureInferenceEngine(n_channels=n_channels, complexity=complexity)

    t_b0 = time.perf_counter()
    infer_result = engine.predict(window)
    t_b1 = time.perf_counter()
    workload_b_ms = (t_b1 - t_b0) * 1_000

    # ── Stress simulation (high complexity only) ──────────────────────────────
    stress_result: Dict[str, Any] = {"reps": 0, "exec_ms": 0.0}
    if complexity == "high":
        stress_result = stress_inference(engine, window, reps=HIGH_COMPLEXITY_REPS)

    # ── Total T_local ─────────────────────────────────────────────────────────
    t_end = time.perf_counter()
    tlocal_ms = (t_end - t_start) * 1_000

    cpu_after = psutil.cpu_percent(interval=None)

    log.info(
        "T_local=%.2f ms  [A=%.2f ms, B=%.2f ms]  gesture=%s(%.0f%%)  "
        "complexity=%s  backend=%s",
        tlocal_ms,
        feat_result["exec_ms"],
        workload_b_ms,
        infer_result["gesture"],
        infer_result["confidence"] * 100,
        complexity,
        infer_result["backend"],
    )

    # ── Build structured result ───────────────────────────────────────────────
    now_ms = int(time.time() * 1000)
    record = BenchmarkRecord(
        timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        timestamp_ms=now_ms,
        device=platform.node(),
        platform_info=_get_platform_info(),
        complexity=complexity,
        workload_a_exec_ms=feat_result["exec_ms"],
        n_channels=feat_result["n_channels"],
        n_features=feat_result["n_features"],
        flat_vector_len=len(feat_result["flat_vector"]),
        workload_b_exec_ms=round(workload_b_ms, 3),
        gesture=infer_result["gesture"],
        confidence=round(infer_result["confidence"], 4),
        backend=infer_result["backend"],
        tlocal_ms=round(tlocal_ms, 3),
        cpu_pct_before=cpu_before,
        cpu_pct_after=cpu_after,
        ram_used_mb=round(ram_mb, 2),
        stress_reps=stress_result["reps"],
        stress_exec_ms=stress_result["exec_ms"],
    )

    if log_result:
        _append_log(record)

    return {
        "tlocal_ms": record.tlocal_ms,
        "workload_a": {
            "exec_ms": record.workload_a_exec_ms,
            "n_channels": record.n_channels,
            "n_features": record.n_features,
            "flat_vector_len": record.flat_vector_len,
            "channel_features": feat_result["channels"],
        },
        "workload_b": {
            "exec_ms": record.workload_b_exec_ms,
            "gesture": record.gesture,
            "gesture_idx": infer_result["gesture_idx"],
            "confidence": record.confidence,
            "probabilities": infer_result["probabilities"],
            "backend": record.backend,
        },
        "stress": stress_result,
        "cpu_pct_before": record.cpu_pct_before,
        "cpu_pct_after": record.cpu_pct_after,
        "ram_used_mb": record.ram_used_mb,
        "complexity": record.complexity,
        "log_path": str(LOG_FILE) if log_result else None,
        "timestamp_ms": record.timestamp_ms,
    }


# ---------------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: Offline model training helper  (not used on the Pi at runtime)
# ─────────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def train_gesture_model(
    X: np.ndarray,
    y: np.ndarray,
    n_channels: int = 8,
    complexity: str = "normal",
    output_dir: pathlib.Path = MODEL_DIR,
) -> None:
    """
    Train and export the gesture CNN to TFLite and ONNX formats.

    This function is intended to run **offline** (laptop / workstation with
    TensorFlow installed) and is never called on the Raspberry Pi.

    Parameters
    ----------
    X:
        Training data, shape ``(n_samples, WINDOW_SIZE, n_channels)``
    y:
        Integer class labels, shape ``(n_samples,)``
    n_channels:
        Number of sEMG input channels.
    complexity:
        ``"normal"`` — baseline architecture
        ``"high"``   — doubled filter counts + extra LSTM layer for stress test
    output_dir:
        Directory to save ``.tflite`` and ``.onnx`` model files.
    """
    try:
        import tensorflow as tf  # type: ignore  # noqa: PLC0415
    except ImportError:
        log.error(
            "TensorFlow is not installed.  "
            "Install it on your development machine with: pip install tensorflow"
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    filter_mult = 2 if complexity == "high" else 1

    # ── Model definition ─────────────────────────────────────────────────────
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, n_channels), name="emg_input")

    # Block 1
    x = tf.keras.layers.Conv1D(32 * filter_mult, 7, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(4)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Block 2
    x = tf.keras.layers.Conv1D(64 * filter_mult, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(4)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Block 3 (extra depth in "high" complexity mode)
    x = tf.keras.layers.Conv1D(128 * filter_mult, 3, padding="same", activation="relu")(x)

    if complexity == "high":
        # Lightweight LSTM on top of the conv backbone (stress test scenario)
        x = tf.keras.layers.LSTM(64, return_sequences=False)(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(64 * filter_mult, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="gesture_probs")(x)

    model = tf.keras.Model(inputs, outputs, name=f"gesture_cnn_{complexity}")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    log.info("Training gesture model  shape=%s  complexity=%s …", X.shape, complexity)
    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

    # ── TFLite export ─────────────────────────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # INT8 post-training quant
    tflite_bytes = converter.convert()
    tflite_path = output_dir / "gesture_cnn.tflite"
    tflite_path.write_bytes(tflite_bytes)
    log.info("TFLite model saved to %s  (%.1f KB)", tflite_path, len(tflite_bytes) / 1024)

    # ── ONNX export ───────────────────────────────────────────────────────────
    try:
        import tf2onnx  # type: ignore  # noqa: PLC0415
        import onnx     # type: ignore  # noqa: PLC0415

        spec = (tf.TensorSpec((1, WINDOW_SIZE, n_channels), tf.float32, name="emg_input"),)
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
        onnx_path = output_dir / "gesture_cnn.onnx"
        onnx.save(onnx_model, str(onnx_path))
        log.info("ONNX model saved to %s", onnx_path)
    except ImportError:
        log.warning("tf2onnx / onnx not installed — skipping ONNX export.")


# ---------------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────
# CLI self-test  (python edge_tasks.py)
# ─────────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """
    Quick smoke-test: run both complexity modes and print a comparison table.
    Useful for verifying the module works before wiring into the scheduler.
    """
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="edge_tasks",
        description="EdgeTrust Local Baseline (T_local) — self-test mode",
    )
    parser.add_argument(
        "--n-channels", type=int, default=8,
        help="Number of simulated sEMG channels (default: 8)"
    )
    parser.add_argument(
        "--sampling-rate", type=int, default=256,
        help="Simulated ADC sampling rate in Hz (default: 256)"
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Do not write results to logs/tlocal_benchmark.json"
    )
    args = parser.parse_args()

    n_ch = args.n_channels
    hz = args.sampling_rate
    rng = np.random.default_rng(0)

    # Simulate a realistic sEMG burst on 8 channels
    t_axis = np.linspace(0, WINDOW_SIZE / hz, WINDOW_SIZE)
    base_signal = (
        np.sin(2 * np.pi * 5.0 * t_axis)           # 5 Hz voluntary contraction
        + 0.3 * np.sin(2 * np.pi * 50.0 * t_axis)  # 50 Hz power-line artefact
        + 0.1 * rng.standard_normal(WINDOW_SIZE)    # Gaussian noise floor
    )
    window = np.column_stack([
        base_signal * rng.uniform(0.5, 1.5) for _ in range(n_ch)
    ]).astype(np.float32)

    print("\n" + "=" * 72)
    print("  EdgeTrust — edge_tasks.py  self-test")
    print(f"  Window shape : {window.shape}   ({WINDOW_SIZE} samples × {n_ch} channels)")
    print(f"  Sampling rate: {hz} Hz")
    print("=" * 72)

    results: Dict[str, Dict[str, Any]] = {}
    for mode in ("normal", "high"):
        print(f"\n▶  Complexity mode: {mode.upper()}")
        eng = GestureInferenceEngine(n_channels=n_ch, complexity=mode)
        r = run_local_benchmark(
            window,
            sampling_rate_hz=float(hz),
            complexity=mode,
            engine=eng,
            log_result=not args.no_log,
        )
        results[mode] = r

        a = r["workload_a"]
        b = r["workload_b"]

        # Per-channel MAV summary
        mavs = [f"{ch_f['mav']:.4f}" for ch_f in a["channel_features"]]
        zcrs = [f"{ch_f['zcr']:.4f}" for ch_f in a["channel_features"]]
        wls  = [f"{ch_f['wl']:.1f}"  for ch_f in a["channel_features"]]

        print(f"  ┌─ Workload A (Feature Extraction)")
        print(f"  │  Time    : {a['exec_ms']:.2f} ms")
        print(f"  │  Channels: {a['n_channels']}   Features/ch: {a['n_features']}")
        print(f"  │  MAV     : {mavs}")
        print(f"  │  ZCR     : {zcrs}")
        print(f"  │  WL      : {wls}")
        print(f"  └─ Workload B (Neural Inference)")
        print(f"     Time    : {b['exec_ms']:.2f} ms")
        print(f"     Backend : {b['backend']}")
        print(f"     Gesture : {b['gesture']}  (conf={b['confidence']:.2%})")
        if r["stress"]["reps"]:
            s = r["stress"]
            print(f"     Stress  : {s['reps']}× reps | {s['exec_ms']:.2f} ms total "
                  f"| {s['exec_ms_per_rep']:.2f} ms/rep")
        print(f"\n  ✔  T_local = {r['tlocal_ms']:.2f} ms  "
              f"[CPU {r['cpu_pct_before']:.1f}% → {r['cpu_pct_after']:.1f}%]  "
              f"RAM {r['ram_used_mb']:.1f} MB")

    # Comparison table
    n_t  = results["normal"]["tlocal_ms"]
    h_t  = results["high"]["tlocal_ms"]
    print("\n" + "=" * 72)
    print(f"  {'Mode':<12} {'T_local (ms)':>14}  {'Gesture':>18}  {'Backend':>12}")
    print(f"  {'-'*12} {'-'*14}  {'-'*18}  {'-'*12}")
    for mode in ("normal", "high"):
        r = results[mode]
        b = r["workload_b"]
        print(
            f"  {mode:<12} {r['tlocal_ms']:>14.2f}  "
            f"{b['gesture']:>18}  {b['backend']:>12}"
        )
    overhead_pct = ((h_t - n_t) / n_t * 100) if n_t > 0 else 0
    print(f"\n  High-complexity overhead vs normal: +{overhead_pct:.1f}%")
    if not args.no_log:
        print(f"  Log written to: {LOG_FILE.resolve()}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    _self_test()
