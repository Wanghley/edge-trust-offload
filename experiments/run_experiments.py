#!/usr/bin/env python3
"""
EdgeTrust-Offload Experiment Suite
====================================
Runs the four-scenario benchmark between Raspberry Pi 4 (client) and
Jetson Orin Nano (server) to collect the data used in the report.

Scenarios
---------
  S1 — Local execution only (RPi, no network)
  S2 — Insecure offload (plain HTTP over LAN, Tailscale bypassed or disabled)
  S3 — Secure offload (HTTP over Tailscale/WireGuard, Zero-Trust enabled)
  S4 — Secure offload under congestion (S3 + tc netem jitter/loss on this host)

Usage (run on Raspberry Pi)
----------------------------
  # Basic run — all four scenarios, 100 trials each, both complexity levels
  python experiments/run_experiments.py \
      --jetson-ip 100.1.1.4 \
      --jetson-port 8000 \
      --insecure-ip 192.168.1.100 \
      --trials 100

  # Congestion injection only (wraps S3 with tc netem on the eth0 interface)
  python experiments/run_experiments.py \
      --jetson-ip 100.1.1.4 \
      --scenarios S3 S4 \
      --congestion-iface eth0 \
      --congestion-delay-ms 20 \
      --congestion-jitter-ms 5 \
      --congestion-loss-pct 1.0

  # Quick smoke-test (10 trials, normal complexity only)
  python experiments/run_experiments.py \
      --jetson-ip 100.1.1.4 \
      --trials 10 \
      --complexity normal

Output
------
  results/experiment_YYYYMMDD_HHMMSS/
    ├── raw/
    │   ├── s1_local.jsonl
    │   ├── s2_insecure.jsonl
    │   ├── s3_secure.jsonl
    │   └── s4_congested.jsonl
    ├── summary.json          ← per-scenario stats (mean, std, p50, p95, p99)
    └── crossover.json        ← L* per complexity level per scenario

Requirements (RPi client side)
-------------------------------
  pip install requests numpy psutil
  sudo apt-get install iproute2   # for tc netem (S4)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("EdgeTrust_Experiment")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    scenario: str
    complexity: str
    trial: int
    timestamp_iso: str
    decision: str                   # LOCAL | OFFLOAD | OFFLOAD_FAILED
    t_exec_local_ms: float          # local feature extraction + inference time
    t_exec_remote_ms: float         # server-reported inference time (0 if local)
    t_net_ms: float                 # measured RTT contribution (0 if local)
    e_crypto_ms: float              # estimated WireGuard encryption overhead
    t_total_ms: float               # wall-clock end-to-end time
    l_star_ms: float                # crossover latency at this trial
    c_local: float
    c_offload: float
    cpu_pct: float
    ram_used_mb: float
    payload_size_bytes: int
    error: Optional[str] = None


@dataclass
class ScenarioSummary:
    scenario: str
    complexity: str
    n_trials: int
    n_success: int
    t_total_mean: float
    t_total_std: float
    t_total_p50: float
    t_total_p95: float
    t_total_p99: float
    t_exec_mean: float
    t_net_mean: float
    e_crypto_mean: float
    l_star_mean: float


# ---------------------------------------------------------------------------
# Workload: synthetic EMG window generator
# ---------------------------------------------------------------------------

class SyntheticEMGSource:
    """Generates synthetic 8-channel EMG windows without needing the HDF5 dataset."""

    def __init__(self, n_channels: int = 8, window_size: int = 1024, sampling_rate: int = 256):
        self.n_channels = n_channels
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self._rng = np.random.default_rng(seed=42)
        self._index = 0

    def get_window(self) -> dict:
        # Simulate band-limited EMG: 20–500 Hz, white noise shaped by Hann window
        t = np.linspace(0, self.window_size / self.sampling_rate, self.window_size)
        samples = []
        for _ in range(self.n_channels):
            sig = (
                self._rng.normal(0, 0.5, self.window_size)
                + 0.3 * np.sin(2 * np.pi * 50 * t)   # 50 Hz fundamental
                + 0.1 * np.sin(2 * np.pi * 120 * t)  # harmonic
            )
            sig *= np.hanning(self.window_size)
            samples.extend(sig.tolist())

        self._index += 1
        return {
            "window_index": self._index,
            "payload": {
                "sampling_rate_hz": self.sampling_rate,
                "n_channels": self.n_channels,
                "samples": samples,
            },
        }


# ---------------------------------------------------------------------------
# Local execution benchmark
# ---------------------------------------------------------------------------

def _extract_emg_features(samples: np.ndarray, n_channels: int) -> np.ndarray:
    """8 statistical features per channel (matches edge_tasks.py)."""
    window_size = len(samples) // n_channels
    mat = samples.reshape(window_size, n_channels)
    features = []
    for ch in range(n_channels):
        sig = mat[:, ch]
        mav  = np.mean(np.abs(sig))
        zcr  = np.sum(np.diff(np.sign(sig)) != 0) / window_size
        wl   = np.sum(np.abs(np.diff(sig)))
        rms  = np.sqrt(np.mean(sig ** 2))
        var  = np.var(sig)
        ssc  = np.sum(np.diff(np.sign(np.diff(sig))) != 0) / window_size
        iemg = np.sum(np.abs(sig))
        fft  = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(window_size, d=1.0 / 256)
        df   = freqs[np.argmax(fft)]
        features.extend([mav, zcr, wl, rms, var, ssc, iemg, df])
    return np.array(features, dtype=np.float32)


def run_local(window_data: dict, complexity: str, stress_reps: int = 8) -> Tuple[float, float]:
    """
    Execute workload locally. Returns (t_exec_ms, t_inference_ms).
    complexity='high' repeats feature extraction stress_reps times.
    """
    samples = np.array(window_data["payload"]["samples"], dtype=np.float32)
    n_channels = window_data["payload"]["n_channels"]

    t0 = time.perf_counter()
    features = _extract_emg_features(samples, n_channels)
    if complexity == "high":
        for _ in range(stress_reps - 1):
            features = _extract_emg_features(samples, n_channels)
    t_feat = (time.perf_counter() - t0) * 1000

    # Lightweight inference: argmax of a random linear projection (no model needed)
    t1 = time.perf_counter()
    _weights = np.random.randn(len(features), 6).astype(np.float32) * 0.01
    logits = features @ _weights
    _gesture_idx = int(np.argmax(logits))
    t_infer = (time.perf_counter() - t1) * 1000

    return t_feat + t_infer, t_infer


# ---------------------------------------------------------------------------
# Crypto overhead estimator (matches main_scheduler.py)
# ---------------------------------------------------------------------------

CHACHA20_THROUGHPUT_BYTES_PER_MS = 15_000  # ~15 MB/s on RPi 4

def estimate_crypto_overhead(payload_size_bytes: int) -> float:
    """Two-way ChaCha20-Poly1305 latency estimate (ms)."""
    t_enc = payload_size_bytes / CHACHA20_THROUGHPUT_BYTES_PER_MS
    t_dec = payload_size_bytes / CHACHA20_THROUGHPUT_BYTES_PER_MS
    return t_enc + t_dec


# ---------------------------------------------------------------------------
# Network RTT measurement
# ---------------------------------------------------------------------------

def measure_rtt_ms(host: str, port: int, timeout: float = 1.0) -> Tuple[float, bool]:
    """TCP connect-time RTT. Returns (rtt_ms, packet_loss_flag)."""
    t0 = time.perf_counter()
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.close()
        return (time.perf_counter() - t0) * 1000, False
    except (socket.timeout, ConnectionRefusedError, OSError):
        return timeout * 1000, True


# ---------------------------------------------------------------------------
# Offload execution
# ---------------------------------------------------------------------------

def run_offload(
    session: requests.Session,
    offload_url: str,
    window_data: dict,
    device_id: str,
    timeout: float = 5.0,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Send window to Jetson Orin Nano. Returns (t_exec_remote_ms, t_net_ms, status).
    status is 'success' or an error string.
    """
    payload = {
        "device_id": device_id,
        "timestamp_ms": int(time.time() * 1000),
        "workload_type": "fft_1024",
        "payload": window_data["payload"],
        "metadata": {
            "client_cpu_load_pct": psutil.cpu_percent(),
            "battery_level_pct": 100,
        },
    }
    t0 = time.perf_counter()
    try:
        resp = session.post(offload_url, json=payload, timeout=timeout)
        t_total = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            t_remote = data.get("compute_latency_ms", 3.0)
            t_net = max(0.0, t_total - t_remote)
            return t_remote, t_net, "success"
        return None, t_total, f"http_{resp.status_code}"
    except requests.exceptions.Timeout:
        return None, (time.perf_counter() - t0) * 1000, "timeout"
    except requests.exceptions.RequestException as exc:
        return None, (time.perf_counter() - t0) * 1000, str(exc)


# ---------------------------------------------------------------------------
# Congestion injection / removal (requires sudo + iproute2)
# ---------------------------------------------------------------------------

def inject_congestion(iface: str, delay_ms: float, jitter_ms: float, loss_pct: float) -> bool:
    """Add tc netem rules to the given interface. Returns True on success."""
    # Remove any existing qdisc first
    subprocess.run(
        ["sudo", "tc", "qdisc", "del", "dev", iface, "root"],
        capture_output=True,
    )
    cmd = [
        "sudo", "tc", "qdisc", "add", "dev", iface, "root", "netem",
        "delay", f"{delay_ms}ms", f"{jitter_ms}ms", "distribution", "normal",
        "loss", f"{loss_pct}%",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"tc netem inject failed: {result.stderr.strip()}")
        return False
    log.info(f"[Congestion ON] iface={iface} delay={delay_ms}±{jitter_ms}ms loss={loss_pct}%")
    return True


def remove_congestion(iface: str) -> None:
    """Remove tc netem rules."""
    subprocess.run(
        ["sudo", "tc", "qdisc", "del", "dev", iface, "root"],
        capture_output=True,
    )
    log.info(f"[Congestion OFF] iface={iface}")


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    def __init__(self, args: argparse.Namespace, out_dir: Path):
        self.args = args
        self.out_dir = out_dir
        self.raw_dir = out_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.source = SyntheticEMGSource(n_channels=8, window_size=1024)
        self.session = requests.Session()
        self.session.headers["X-Device-ID"] = args.device_id

        # Scenario → server URL mapping
        self.urls: Dict[str, str] = {}
        if "S2" in args.scenarios:
            self.urls["S2"] = f"http://{args.insecure_ip}:{args.jetson_port}/api/v1/offload/fft"
        if "S3" in args.scenarios or "S4" in args.scenarios:
            self.urls["S3"] = f"http://{args.jetson_ip}:{args.jetson_port}/api/v1/offload/fft"
            self.urls["S4"] = self.urls["S3"]

    def _trial_local(self, trial: int, complexity: str) -> TrialResult:
        window = self.source.get_window()
        payload_bytes = len(json.dumps(window["payload"]).encode())
        e_crypto = 0.0  # no network traffic

        t_wall0 = time.perf_counter()
        t_exec, _ = run_local(window, complexity)
        t_wall = (time.perf_counter() - t_wall0) * 1000

        # Cost model (alpha=beta=gamma=delta=1, no offload path)
        c_local = t_exec
        c_offload = float("inf")
        l_star = t_exec - 3.0 - 1.0  # rough estimate (T_remote≈3ms, E_crypto≈1ms)

        return TrialResult(
            scenario="S1",
            complexity=complexity,
            trial=trial,
            timestamp_iso=datetime.utcnow().isoformat() + "Z",
            decision="LOCAL",
            t_exec_local_ms=round(t_exec, 3),
            t_exec_remote_ms=0.0,
            t_net_ms=0.0,
            e_crypto_ms=e_crypto,
            t_total_ms=round(t_wall, 3),
            l_star_ms=round(l_star, 3),
            c_local=round(c_local, 3),
            c_offload=round(c_offload, 3),
            cpu_pct=psutil.cpu_percent(),
            ram_used_mb=psutil.virtual_memory().used / 1e6,
            payload_size_bytes=payload_bytes,
        )

    def _trial_offload(
        self,
        scenario: str,
        trial: int,
        complexity: str,
        host: str,
    ) -> TrialResult:
        window = self.source.get_window()
        payload_bytes = len(json.dumps(window["payload"]).encode())
        e_crypto = estimate_crypto_overhead(payload_bytes) if scenario in ("S3", "S4") else 0.0

        rtt, loss = measure_rtt_ms(host, self.args.jetson_port)
        t_exec_local, _ = run_local(window, "normal")  # always measure local baseline too

        url = self.urls[scenario]
        t_remote, t_net, status = run_offload(
            self.session, url, window, self.args.device_id
        )

        if status == "success":
            t_remote = t_remote or 3.0
            t_net = t_net or rtt
            decision = "OFFLOAD"
            t_total = (t_remote + t_net + e_crypto)
        else:
            t_remote = 0.0
            t_net = rtt
            decision = "OFFLOAD_FAILED"
            t_total = t_exec_local  # fell back to local

        c_local = t_exec_local
        c_offload = t_remote + t_net + e_crypto + (500.0 if loss else 0.0)
        l_star = max(0.0, t_exec_local - t_remote - e_crypto)

        return TrialResult(
            scenario=scenario,
            complexity=complexity,
            trial=trial,
            timestamp_iso=datetime.utcnow().isoformat() + "Z",
            decision=decision,
            t_exec_local_ms=round(t_exec_local, 3),
            t_exec_remote_ms=round(t_remote, 3),
            t_net_ms=round(t_net, 3),
            e_crypto_ms=round(e_crypto, 3),
            t_total_ms=round(t_total, 3),
            l_star_ms=round(l_star, 3),
            c_local=round(c_local, 3),
            c_offload=round(c_offload, 3),
            cpu_pct=psutil.cpu_percent(),
            ram_used_mb=psutil.virtual_memory().used / 1e6,
            payload_size_bytes=payload_bytes,
            error=None if status == "success" else status,
        )

    def _write_trial(self, result: TrialResult, fh) -> None:
        fh.write(json.dumps(asdict(result)) + "\n")
        fh.flush()

    def run_scenario(self, scenario: str, complexity: str) -> List[TrialResult]:
        fname_map = {"S1": "s1_local", "S2": "s2_insecure", "S3": "s3_secure", "S4": "s4_congested"}
        out_file = self.raw_dir / f"{fname_map[scenario]}_{complexity}.jsonl"
        results: List[TrialResult] = []

        log.info(f"=== Scenario {scenario} | complexity={complexity} | trials={self.args.trials} ===")

        with open(out_file, "w") as fh:
            for trial in range(1, self.args.trials + 1):
                if trial % 10 == 0:
                    log.info(f"  Trial {trial}/{self.args.trials}")

                try:
                    if scenario == "S1":
                        r = self._trial_local(trial, complexity)
                    else:
                        host = self.args.jetson_ip
                        r = self._trial_offload(scenario, trial, complexity, host)
                except Exception as exc:
                    log.error(f"  Trial {trial} error: {exc}")
                    continue

                results.append(r)
                self._write_trial(r, fh)

                # Rate-limit to avoid saturating the server
                time.sleep(0.05)

        log.info(f"  Done. {len(results)} trials saved to {out_file}")
        return results

    def compute_summary(self, results: List[TrialResult], scenario: str, complexity: str) -> ScenarioSummary:
        totals = np.array([r.t_total_ms for r in results])
        execs  = np.array([r.t_exec_local_ms for r in results])
        nets   = np.array([r.t_net_ms for r in results])
        crypts = np.array([r.e_crypto_ms for r in results])
        lstars = np.array([r.l_star_ms for r in results])
        n_ok   = sum(1 for r in results if r.error is None)

        return ScenarioSummary(
            scenario=scenario,
            complexity=complexity,
            n_trials=len(results),
            n_success=n_ok,
            t_total_mean=float(np.mean(totals)),
            t_total_std=float(np.std(totals)),
            t_total_p50=float(np.percentile(totals, 50)),
            t_total_p95=float(np.percentile(totals, 95)),
            t_total_p99=float(np.percentile(totals, 99)),
            t_exec_mean=float(np.mean(execs)),
            t_net_mean=float(np.mean(nets)),
            e_crypto_mean=float(np.mean(crypts)),
            l_star_mean=float(np.mean(lstars)),
        )

    def run_all(self) -> None:
        all_results: Dict[str, Dict[str, List[TrialResult]]] = {}
        summaries: List[ScenarioSummary] = []
        crossover: Dict[str, Dict[str, float]] = {}

        for scenario in self.args.scenarios:
            all_results[scenario] = {}
            crossover[scenario] = {}

            for complexity in self.args.complexity:
                congestion_active = False
                try:
                    if scenario == "S4":
                        ok = inject_congestion(
                            self.args.congestion_iface,
                            self.args.congestion_delay_ms,
                            self.args.congestion_jitter_ms,
                            self.args.congestion_loss_pct,
                        )
                        if not ok:
                            log.warning("S4: congestion injection failed, running without congestion")
                        else:
                            congestion_active = True
                        time.sleep(0.5)  # let qdisc settle

                    results = self.run_scenario(scenario, complexity)
                    all_results[scenario][complexity] = results

                    summary = self.compute_summary(results, scenario, complexity)
                    summaries.append(summary)
                    crossover[scenario][complexity] = summary.l_star_mean

                    log.info(
                        f"  Summary | mean={summary.t_total_mean:.1f}ms "
                        f"p95={summary.t_total_p95:.1f}ms L*={summary.l_star_mean:.1f}ms"
                    )
                finally:
                    if congestion_active:
                        remove_congestion(self.args.congestion_iface)

        # Write summary.json
        summary_path = self.out_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump([asdict(s) for s in summaries], f, indent=2)
        log.info(f"Summary saved: {summary_path}")

        # Write crossover.json
        crossover_path = self.out_dir / "crossover.json"
        with open(crossover_path, "w") as f:
            json.dump(crossover, f, indent=2)
        log.info(f"Crossover L* saved: {crossover_path}")

        # Print results table
        self._print_table(summaries)

    def _print_table(self, summaries: List[ScenarioSummary]) -> None:
        print("\n" + "=" * 80)
        print(f"{'Scenario':<6} {'Complexity':<10} {'N':<6} {'Mean(ms)':<10} {'Std':<8} {'p95':<8} {'L*(ms)':<10}")
        print("-" * 80)
        for s in summaries:
            print(
                f"{s.scenario:<6} {s.complexity:<10} {s.n_trials:<6} "
                f"{s.t_total_mean:<10.1f} {s.t_total_std:<8.1f} "
                f"{s.t_total_p95:<8.1f} {s.l_star_mean:<10.1f}"
            )
        print("=" * 80)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_jetson_health(ip: str, port: int, device_id: str) -> bool:
    url = f"http://{ip}:{port}/health"
    try:
        resp = requests.get(url, headers={"X-Device-ID": device_id}, timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            log.info(
                f"[Jetson OK] backend={data.get('inference_backend')} "
                f"gpu={data.get('system_stats', {}).get('gpu_temp_c')}°C"
            )
            return True
        log.warning(f"[Jetson] Health check returned {resp.status_code}")
        return False
    except Exception as exc:
        log.error(f"[Jetson] Unreachable: {exc}")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EdgeTrust-Offload 4-Scenario Benchmark Suite (RPi → Jetson Orin Nano)"
    )

    # Network
    p.add_argument("--jetson-ip",    default=os.getenv("JETSON_IP",    "100.1.1.4"),  help="Jetson Tailscale IP (secure path)")
    p.add_argument("--jetson-port",  default=int(os.getenv("JETSON_PORT", "8000")), type=int, help="Jetson server port")
    p.add_argument("--insecure-ip",  default=os.getenv("INSECURE_IP",  "192.168.1.100"), help="Jetson LAN IP (insecure path, S2)")
    p.add_argument("--device-id",    default="rpi-client-benchmark",   help="X-Device-ID for Zero-Trust auth")

    # Experiment control
    p.add_argument(
        "--scenarios", nargs="+",
        choices=["S1", "S2", "S3", "S4"], default=["S1", "S2", "S3", "S4"],
        help="Which scenarios to run",
    )
    p.add_argument(
        "--complexity", nargs="+",
        choices=["normal", "high"], default=["normal", "high"],
        help="Workload complexity levels",
    )
    p.add_argument("--trials", type=int, default=100, help="Number of trials per (scenario, complexity)")

    # Congestion (S4)
    p.add_argument("--congestion-iface",    default="eth0",  help="Network interface for tc netem (S4)")
    p.add_argument("--congestion-delay-ms", type=float, default=20.0, help="Base delay to add (ms)")
    p.add_argument("--congestion-jitter-ms",type=float, default=5.0,  help="Jitter (ms, Gaussian std)")
    p.add_argument("--congestion-loss-pct", type=float, default=1.0,  help="Packet loss percentage")

    # Output
    p.add_argument("--out-dir", default="results", help="Output directory root")
    p.add_argument("--skip-health-check", action="store_true", help="Skip Jetson /health check on startup")
    p.add_argument("-v", "--verbose", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        log.setLevel(logging.DEBUG)

    # Create timestamped output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"experiment_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {out_dir}")

    # Health check — skip for S1-only or when flag set
    needs_jetson = any(s in args.scenarios for s in ("S2", "S3", "S4"))
    if needs_jetson and not args.skip_health_check:
        ok = check_jetson_health(args.jetson_ip, args.jetson_port, args.device_id)
        if not ok:
            log.error("Jetson Orin Nano unreachable. Use --skip-health-check to override.")
            sys.exit(1)

    runner = ExperimentRunner(args, out_dir)
    runner.run_all()

    log.info(f"All experiments complete. Results in: {out_dir}")


if __name__ == "__main__":
    main()
