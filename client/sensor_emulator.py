"""
sensor_emulator.py
==================
EdgeTrust-Offload — Virtual Sensor Service (Raspberry Pi side)

Wraps the BiomechanicsDataset.hdf5 behind a localhost-only FastAPI server so
that any component in the pipeline can call GET /get_window and receive the
next 1024-sample window of real biomechanical data as if it were live ADC /
IMU output.

Architecture context
--------------------
  Raspberry Pi (100.1.1.2)
    └─ sensor_emulator.py   →  listens on 127.0.0.1:9000
    └─ client/main.py       →  calls GET http://127.0.0.1:9000/get_window
                               then decides whether to offload to Jetson Nano

Usage
-----
    python client/sensor_emulator.py [options]

    # Override defaults with environment variables or CLI flags:
    python client/sensor_emulator.py \
        --hdf5-path data/raw/BiomechanicsDataset.hdf5 \
        --participant-id B026 \
        --task-name IndexFlexion_MVC \
        --sampling-rate 256 \
        --window-size 1024 \
        --port 9000

Environment variables (take lower priority than CLI flags)
----------------------------------------------------------
    HDF5_PATH          Path to the HDF5 file
    PARTICIPANT_ID     Participant to stream
    TASK_NAME          Task to stream
    SAMPLING_RATE_HZ   Simulated ADC sampling rate (default: 256)
    WINDOW_SIZE        Samples per window (default: 1024)
    EMULATOR_PORT      TCP port to bind (default: 9000)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sensor_emulator")


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class EmulatorConfig:
    """All runtime knobs for the virtual sensor."""

    hdf5_path: str = field(
        default_factory=lambda: os.getenv(
            "HDF5_PATH", "data/raw/BiomechanicsDataset.hdf5"
        )
    )
    participant_id: Optional[str] = field(
        default_factory=lambda: os.getenv("PARTICIPANT_ID")
    )
    task_name: Optional[str] = field(
        default_factory=lambda: os.getenv("TASK_NAME")
    )
    sampling_rate_hz: int = field(
        default_factory=lambda: int(os.getenv("SAMPLING_RATE_HZ", "256"))
    )
    window_size: int = field(
        default_factory=lambda: int(os.getenv("WINDOW_SIZE", "1024"))
    )
    host: str = "127.0.0.1"  # local-only — never expose to the network
    port: int = field(
        default_factory=lambda: int(os.getenv("EMULATOR_PORT", "9000"))
    )
    loop_dataset: bool = True  # wrap around when EOF is reached


# ---------------------------------------------------------------------------
# Dataset loader & state manager
# ---------------------------------------------------------------------------
class VirtualSensorState:
    """
    Loads one participant/task dataset from the HDF5 file into memory and
    maintains a thread-safe sample pointer so consecutive /get_window calls
    step through the data chronologically.

    HDF5 structure (inferred from scripts/setup_data.py):
        /
        └── <participant_id>   (Group)
                └── <task_name>  (Dataset, shape=(N, C) or (N,))
                         attrs — dtype.names gives channel names when present
    """

    def __init__(self, config: EmulatorConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._pointer: int = 0
        self._data: np.ndarray = np.empty((0,))
        self._columns: List[str] = []
        self._total_samples: int = 0
        self._windows_served: int = 0
        self._start_time: float = time.time()
        self._loaded_participant: str = ""
        self._loaded_task: str = ""

    # ------------------------------------------------------------------
    # Public bootstrap helpers
    # ------------------------------------------------------------------

    def discover(self) -> Dict[str, List[str]]:
        """Return {participant_id: [task, ...]} catalogue from the HDF5 file."""
        catalogue: Dict[str, List[str]] = {}
        with h5py.File(self._config.hdf5_path, "r") as h5:
            for pid in sorted(h5.keys()):
                group = h5[pid]
                if isinstance(group, h5py.Group):
                    catalogue[pid] = sorted(group.keys())
        return catalogue

    def load(
        self,
        participant_id: Optional[str] = None,
        task_name: Optional[str] = None,
    ) -> None:
        """
        Load a specific participant/task dataset into RAM.  If either argument
        is None the first available entry is chosen automatically, which is
        useful for demos on a freshly downloaded dataset.
        """
        with h5py.File(self._config.hdf5_path, "r") as h5:
            # Auto-select participant
            if participant_id is None:
                participant_id = sorted(h5.keys())[0]
                log.info("No participant specified — using first: %s", participant_id)

            if participant_id not in h5:
                available = ", ".join(sorted(h5.keys())[:10])
                raise KeyError(
                    f"Participant '{participant_id}' not found. "
                    f"Available (first 10): {available}"
                )

            group = h5[participant_id]

            # Auto-select task
            if task_name is None:
                task_name = sorted(group.keys())[0]
                log.info("No task specified — using first: %s", task_name)

            if task_name not in group:
                available_tasks = ", ".join(sorted(group.keys())[:10])
                raise KeyError(
                    f"Task '{task_name}' not found for participant "
                    f"'{participant_id}'. Available (first 10): {available_tasks}"
                )

            dataset = group[task_name]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(
                    f"'{participant_id}/{task_name}' is a Group, not a Dataset."
                )

            raw = dataset[:]  # load fully into RAM — typical per-task arrays fit easily

            # Normalise to 2-D (samples × channels)
            if raw.ndim == 1:
                raw = raw.reshape(-1, 1)

            # Derive column names from compound dtype when available
            columns: List[str]
            if dataset.dtype.names:
                columns = list(dataset.dtype.names)
                # Compound dtype arrays come out as structured; unpack to float32
                raw = np.column_stack(
                    [raw[col].astype(np.float32) for col in columns]
                )
            else:
                n_channels = raw.shape[1]
                columns = [f"ch_{i:02d}" for i in range(n_channels)]
                raw = raw.astype(np.float32)

        with self._lock:
            self._data = raw
            self._columns = columns
            self._total_samples = raw.shape[0]
            self._pointer = 0
            self._windows_served = 0
            self._start_time = time.time()
            self._loaded_participant = participant_id
            self._loaded_task = task_name

        log.info(
            "Loaded dataset: participant=%s  task=%s  shape=%s  channels=%s",
            participant_id,
            task_name,
            raw.shape,
            columns,
        )

    # ------------------------------------------------------------------
    # Core streaming method
    # ------------------------------------------------------------------

    def next_window(self) -> Dict[str, Any]:
        """
        Return the next *window_size* samples, advancing the internal pointer.

        Timing gate
        -----------
        To simulate a real ADC at *sampling_rate_hz*, this method sleeps for
        the wall-clock duration that *window_size* samples would take to
        accumulate.  This lets downstream code call /get_window in a tight
        loop without overrunning the dataset faster than real hardware would
        produce it.

          sleep_time = window_size / sampling_rate_hz
          e.g.  1024 samples @ 256 Hz → 4.0 seconds between windows

        The sleep is applied only when the elapsed time since the previous
        window was served is *less* than expected, effectively rate-limiting
        without blocking the OS thread unnecessarily.
        """
        cfg = self._config

        with self._lock:
            if self._total_samples == 0:
                raise RuntimeError(
                    "Dataset not loaded — call /load first or restart "
                    "with valid --participant-id and --task-name."
                )

            expected_interval_s: float = cfg.window_size / cfg.sampling_rate_hz

            # Soft rate-limit: sleep only the remaining slice of the window
            # interval so the API stays responsive to administrative queries.
            now = time.time()
            elapsed = now - self._start_time
            expected_elapsed = self._windows_served * expected_interval_s
            deficit = expected_elapsed - elapsed
            if deficit > 0:
                # Release lock while sleeping so health/info endpoints don't block
                self._lock.release()
                try:
                    time.sleep(deficit)
                finally:
                    self._lock.acquire()

            start = self._pointer
            end = start + cfg.window_size

            if end > self._total_samples:
                if cfg.loop_dataset:
                    log.info(
                        "Reached end of dataset at sample %d — wrapping to start.",
                        self._total_samples,
                    )
                    self._pointer = 0
                    start = 0
                    end = cfg.window_size
                else:
                    raise StopIteration("End of dataset reached.")

            window = self._data[start:end]  # shape (window_size, n_channels)
            self._pointer = end
            self._windows_served += 1
            windows_served_snapshot = self._windows_served
            pointer_snapshot = self._pointer

        timestamp_ms = int(time.time() * 1000)
        return {
            "device_id": f"virtual-sensor-{self._loaded_participant}-{self._loaded_task}",
            "timestamp_ms": timestamp_ms,
            "window_index": windows_served_snapshot - 1,
            "sample_start": start,
            "sample_end": end,
            "total_samples_in_dataset": self._total_samples,
            "pointer": pointer_snapshot,
            "payload": {
                "participant_id": self._loaded_participant,
                "task_name": self._loaded_task,
                "sampling_rate_hz": cfg.sampling_rate_hz,
                "window_size": cfg.window_size,
                "n_channels": len(self._columns),
                "channel_names": self._columns,
                "precision": "float32",
                # Return as list-of-lists (rows × channels) — matches the
                # architecture_blueprint.md "samples" field convention.
                "samples": window.tolist(),
            },
            "metadata": {
                "loop_dataset": cfg.loop_dataset,
                "dataset_progress_pct": round(
                    pointer_snapshot / self._total_samples * 100, 2
                ),
            },
        }

    # ------------------------------------------------------------------
    # Introspection helpers (used by /info and /health endpoints)
    # ------------------------------------------------------------------

    @property
    def info(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "participant_id": self._loaded_participant,
                "task_name": self._loaded_task,
                "total_samples": self._total_samples,
                "n_channels": len(self._columns),
                "channel_names": self._columns,
                "window_size": self._config.window_size,
                "sampling_rate_hz": self._config.sampling_rate_hz,
                "loop_dataset": self._config.loop_dataset,
                "windows_served": self._windows_served,
                "current_pointer": self._pointer,
                "dataset_progress_pct": (
                    round(self._pointer / self._total_samples * 100, 2)
                    if self._total_samples
                    else 0.0
                ),
                "uptime_s": round(time.time() - self._start_time, 1),
            }


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------

def build_app(config: EmulatorConfig, state: VirtualSensorState) -> FastAPI:
    app = FastAPI(
        title="EdgeTrust Virtual Sensor",
        description=(
            "Local-only API that streams windowed biomechanical data "
            "(EMG / Kinematics) from the BiomechanicsDataset.hdf5 to the "
            "EdgeTrust-Offload pipeline running on the Raspberry Pi."
        ),
        version="1.0.0",
        docs_url="/docs",      # Swagger UI at http://127.0.0.1:9000/docs
        redoc_url="/redoc",
    )

    # Allow same-machine JavaScript dev tools to hit the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "http://127.0.0.1"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------
    @app.get("/health", tags=["Monitoring"])
    async def health() -> JSONResponse:
        """Lightweight liveness probe."""
        return JSONResponse({"status": "ok", "service": "sensor_emulator"})

    # ------------------------------------------------------------------
    # GET /info
    # ------------------------------------------------------------------
    @app.get("/info", tags=["Monitoring"])
    async def info() -> JSONResponse:
        """Return current dataset and streaming configuration."""
        return JSONResponse(state.info)

    # ------------------------------------------------------------------
    # GET /catalogue
    # ------------------------------------------------------------------
    @app.get("/catalogue", tags=["Dataset"])
    async def catalogue() -> JSONResponse:
        """
        List all participants and their tasks available in the HDF5 file.
        Useful for selecting a stream without manually inspecting the file.
        """
        try:
            cat = state.discover()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return JSONResponse({"catalogue": cat})

    # ------------------------------------------------------------------
    # POST /load
    # ------------------------------------------------------------------
    @app.post("/load", tags=["Dataset"])
    async def load_dataset(
        participant_id: Optional[str] = Query(None, description="Participant ID (e.g. B026)"),
        task_name: Optional[str] = Query(None, description="Task name (e.g. IndexFlexion_MVC)"),
    ) -> JSONResponse:
        """
        Hot-swap the active dataset without restarting the server.
        Resets the sample pointer to zero.
        """
        try:
            state.load(participant_id, task_name)
        except (KeyError, TypeError, FileNotFoundError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(
            {"status": "loaded", "info": state.info}
        )

    # ------------------------------------------------------------------
    # GET /get_window  ← PRIMARY ENDPOINT used by client/main.py
    # ------------------------------------------------------------------
    @app.get("/get_window", tags=["Streaming"])
    async def get_window() -> JSONResponse:
        """
        Return the next 1024-sample window of biomechanical data.

        This is the primary endpoint consumed by the EdgeTrust pipeline.
        Each call advances the internal sample pointer forward by *window_size*
        samples, simulating chronological ADC / IMU output at *sampling_rate_hz*.

        The response schema matches the architecture_blueprint.md payload
        convention so the existing client/main.py offloading logic can wrap
        the `payload.samples` field directly into the POST /api/v1/offload/fft
        request body.
        """
        try:
            window_data = state.next_window()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except StopIteration:
            raise HTTPException(
                status_code=410,
                detail=(
                    "End of dataset reached and loop_dataset=False. "
                    "POST /load to restart or set loop_dataset=True."
                ),
            )
        return JSONResponse(window_data)

    # ------------------------------------------------------------------
    # POST /reset
    # ------------------------------------------------------------------
    @app.post("/reset", tags=["Dataset"])
    async def reset_pointer() -> JSONResponse:
        """Rewind the sample pointer to the start of the current dataset."""
        with state._lock:
            state._pointer = 0
            state._windows_served = 0
            state._start_time = time.time()
        return JSONResponse({"status": "reset", "pointer": 0})

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sensor_emulator",
        description=(
            "EdgeTrust Virtual Sensor — streams windowed biomechanical data "
            "from an HDF5 file via a local FastAPI server."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hdf5-path",
        default=os.getenv("HDF5_PATH", "data/raw/BiomechanicsDataset.hdf5"),
        help="Path to the BiomechanicsDataset.hdf5 file.",
    )
    parser.add_argument(
        "--participant-id",
        default=os.getenv("PARTICIPANT_ID"),
        help="Participant ID to stream (e.g. B026). Auto-selects first if omitted.",
    )
    parser.add_argument(
        "--task-name",
        default=os.getenv("TASK_NAME"),
        help="Task name to stream (e.g. IndexFlexion_MVC). Auto-selects first if omitted.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=int(os.getenv("SAMPLING_RATE_HZ", "256")),
        metavar="HZ",
        help=(
            "Simulated ADC/IMU sampling rate in Hz. Controls how long the "
            "service waits between successive /get_window calls so the "
            "pipeline is not served data faster than real hardware would "
            "produce it. Common values: 256, 512, 1000, 2000."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=int(os.getenv("WINDOW_SIZE", "1024")),
        metavar="SAMPLES",
        help="Number of samples per window returned by /get_window.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("EMULATOR_PORT", "9000")),
        help="TCP port to bind (always on 127.0.0.1).",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help=(
            "Stop streaming at the end of the dataset instead of wrapping "
            "around to the first sample."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging verbosity.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.getLogger().setLevel(args.log_level.upper())

    config = EmulatorConfig(
        hdf5_path=args.hdf5_path,
        participant_id=args.participant_id,
        task_name=args.task_name,
        sampling_rate_hz=args.sampling_rate,
        window_size=args.window_size,
        port=args.port,
        loop_dataset=not args.no_loop,
    )

    state = VirtualSensorState(config)

    # Initial dataset load — exits early with a helpful message if the HDF5
    # file is missing rather than silently failing inside the first request.
    try:
        state.load(config.participant_id, config.task_name)
    except FileNotFoundError:
        log.error(
            "HDF5 file not found: %s\n"
            "  → Run: python scripts/setup_data.py --download-dataset\n"
            "  → Or set --hdf5-path to the correct location.",
            config.hdf5_path,
        )
        return 1
    except (KeyError, TypeError) as exc:
        log.error("Failed to load dataset: %s", exc)
        return 1

    app = build_app(config, state)

    log.info(
        "Starting EdgeTrust Virtual Sensor on http://%s:%d",
        config.host,
        config.port,
    )
    log.info(
        "Streaming: participant=%s  task=%s  window=%d samples @ %d Hz "
        "(≈%.2fs per window)",
        state._loaded_participant,
        state._loaded_task,
        config.window_size,
        config.sampling_rate_hz,
        config.window_size / config.sampling_rate_hz,
    )
    log.info("Swagger UI: http://%s:%d/docs", config.host, config.port)

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=args.log_level,
        # Single worker — dataset state is process-local.
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
