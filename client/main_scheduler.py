import time
import json
import logging
import os
import csv
import socket
from pathlib import Path

import requests
import numpy as np
import sys

try:
    from .edge_tasks import run_local_benchmark, GestureInferenceEngine
    HAS_EDGE_TASKS = True
except (ImportError, ValueError):
    try:
        from edge_tasks import run_local_benchmark, GestureInferenceEngine
        HAS_EDGE_TASKS = True
    except ImportError:
        HAS_EDGE_TASKS = False

# Fallback for tflite-runtime if it's not installed on the dev machine
try:
    import tflite_runtime.interpreter as tflite
    HAS_TFLITE = True
except ImportError:
    HAS_TFLITE = False

# ---------------------------------------------------------------------------
# Logging & Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("EdgeTrust_Scheduler")

# Network Settings
EMULATOR_URL = os.getenv("EMULATOR_URL", "http://127.0.0.1:9000/get_window")
JETSON_IP = os.getenv("JETSON_IP", "100.1.1.4")
JETSON_PORT = os.getenv("JETSON_PORT", "8000")
OFFLOAD_URL = f"http://{JETSON_IP}:{JETSON_PORT}/api/v1/offload/fft"

# Cache for GestureInferenceEngine based on n_channels
engine_cache = {}

# Cost Function Weights
ALPHA = float(os.getenv("WEIGHT_ALPHA", "1.0"))   # Weight for Execution Time
BETA = float(os.getenv("WEIGHT_BETA", "1.0"))     # Weight for Network Latency
GAMMA = float(os.getenv("WEIGHT_GAMMA", "1.0"))   # Weight for Cryptographic Overhead
DELTA = float(os.getenv("WEIGHT_DELTA", "1.0"))   # Weight for Packet Loss/Jitter penalty

# Log file for Zero-Trust Metrics
TELEMETRY_LOG = Path(__file__).parent.parent / "offload_telemetry.csv"

# ---------------------------------------------------------------------------
# Telemetry Initialization
# ---------------------------------------------------------------------------
def init_telemetry():
    if not TELEMETRY_LOG.exists():
        with open(TELEMETRY_LOG, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "window_index", "decision", 
                "t_exec_local_ms", "t_exec_remote_ms", "t_net_ms", 
                "e_crypto_ms", "c_local", "c_offload", "l_star_ms"
            ])

def log_telemetry(row):
    with open(TELEMETRY_LOG, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

# ---------------------------------------------------------------------------
# Cryptographic Simulation (Zero-Trust Overhead)
# ---------------------------------------------------------------------------
def estimate_crypto_overhead(payload_size_bytes: int) -> float:
    """
    Simulates the ChaCha20-Poly1305 encryption/decryption overhead that 
    Tailscale (WireGuard) adds. In a real system, this is handled by the OS kernel.
    We synthesize this metric based on known RPi 2/3 cryptographic throughput 
    (approx 10-20 MB/s for ChaCha20).
    """
    throughput_bytes_per_ms = 15000  # Synthesizing ~15 MB/s
    t_enc_ms = payload_size_bytes / throughput_bytes_per_ms
    t_dec_ms = payload_size_bytes / throughput_bytes_per_ms  # Similar on the return trip
    return t_enc_ms + t_dec_ms

# ---------------------------------------------------------------------------
# Path Execution Stubs
# ---------------------------------------------------------------------------
def execute_path_a_local(window_data: dict) -> float:
    """
    Executes a local inference task via edge_tasks.
    Returns the execution time in ms.
    """
    if HAS_EDGE_TASKS:
        samples = np.array(window_data["payload"]["samples"])
        sampling_rate = window_data["payload"].get("sampling_rate_hz", 256.0)
        n_channels = window_data["payload"].get("n_channels", len(samples[0]) if len(samples) > 0 else 1)
        
        if n_channels not in engine_cache:
            log.info(f"Initializing Inference Engine for {n_channels} channels...")
            engine_cache[n_channels] = GestureInferenceEngine(n_channels=n_channels, complexity="normal")

        result = run_local_benchmark(
            window=samples,
            sampling_rate_hz=sampling_rate,
            complexity="normal", 
            engine=engine_cache[n_channels],
            log_result=False
        )
        return result["tlocal_ms"]
    else:
        # Fallback processing if edge_tasks is missing
        start_time = time.perf_counter()
        samples = np.array(window_data["payload"]["samples"])
        _ = np.fft.rfft(samples, axis=0)
        mock_delay = max(0, 0.030 - (time.perf_counter() - start_time))
        time.sleep(mock_delay)

        return (time.perf_counter() - start_time) * 1000

def measure_network_rtt() -> (float, bool):
    """
    Estimates current network latency (T_net) and packet loss state.
    Sends a lightweight ping via Tailscale IP or socket check.
    """
    start = time.perf_counter()
    try:
        # Fast socket connection to mimic TCP handshake RTT 
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)
        sock.connect((JETSON_IP, int(JETSON_PORT)))
        sock.close()
        rtt_ms = (time.perf_counter() - start) * 1000
        return rtt_ms, False # False means no packet loss
    except (socket.timeout, ConnectionRefusedError, OSError):
        # Simulated timeout scenarios
        return 150.0, True # Penalty RTT and Packet Loss Flag

def execute_path_b_offload(window_data: dict) -> dict:
    """
    Sends the data window over Tailscale to the Jetson Nano.
    Returns tracking metrics.
    """
    payload = {
        "device_id": window_data.get("device_id", "rpi-client-scheduler"),
        "timestamp_ms": int(time.time() * 1000),
        "workload_type": "fft_1024",
        "payload": window_data["payload"],
        "metadata": {
            "client_cpu_load_pct": 50.0, # Placeholder
            "battery_level_pct": 100     # Placeholder
        }
    }
    
    t_net_start = time.perf_counter()
    
    processing_time_remote = 0.0
    network_latency = 0.0
    status = "failed"
    packet_loss = False
    
    try:
        req = requests.post(OFFLOAD_URL, json=payload, timeout=2.0)
        t_net_end = time.perf_counter()
        if req.status_code == 200:
            res_json = req.json()
            processing_time_remote = res_json.get("compute_latency_ms", 5.0)
            network_latency = (t_net_end - t_net_start) * 1000 - processing_time_remote
            status = "success"
    except requests.RequestException:
        # Jetson Nano potentially unreachable
        network_latency = (time.perf_counter() - t_net_start) * 1000
        packet_loss = True
        # For simulation purposes, guess Jetson computes extremely fast (~4ms)
        processing_time_remote = 4.0

    return {
        "status": status,
        "t_exec_remote_ms": processing_time_remote,
        "t_net_ms": network_latency,
        "packet_loss": packet_loss
    }

# ---------------------------------------------------------------------------
# Scheduler Main Loop
# ---------------------------------------------------------------------------
def main():
    log.info("Starting EdgeTrust Main Scheduler")
    init_telemetry()
    
    # Establish a baseline for local execution to inform subsequent decisions
    log.info("Profiling local computation baseline...")
    baseline_exec_local_ms = 35.0  # Safe initial default
    
    while True:
        try:
            # 1. Acquire Data
            response = requests.get(EMULATOR_URL, timeout=10.0)
            if response.status_code != 200:
                log.warning(f"Failed to fetch window: {response.status_code}")
                time.sleep(1)
                continue
                
            window_data = response.json()
            window_index = window_data["window_index"]
            payload_size = len(json.dumps(window_data["payload"]))
            
            # 2. Decision Engine Estimates
            e_crypto = estimate_crypto_overhead(payload_size)
            rtt_estimate, packet_loss_detected = measure_network_rtt()
            t_exec_remote_est = 5.0 # Jetson GPU is fast, assumed fixed
            
            # Predict costs
            c_local = ALPHA * baseline_exec_local_ms
            c_offload = (ALPHA * t_exec_remote_est) + (BETA * rtt_estimate) + (GAMMA * e_crypto) + (DELTA * (500.0 if packet_loss_detected else 0.0))
            
            # Calculate the L* Crossover Point
            # L* = Network Latency where C_local == C_offload
            # C_local = ALPHA * T_rem + BETA * L* + GAMMA * E_crypto + DELTA * P_loss
            # L* = (C_local - ALPHA*T_rem - GAMMA*E_crypto - DELTA*P_loss) / BETA
            l_star = (c_local - (ALPHA * t_exec_remote_est) - (GAMMA * e_crypto) - (DELTA * (500.0 if packet_loss_detected else 0.0))) / BETA
            
            log.info(f"--- Window {window_index} ---")
            log.info(f"Cost Local: {c_local:.2f} | Cost Offload: {c_offload:.2f} | L*: {l_star:.2f}ms")
            
            # 3. Execution Path Selection
            t_exec_local = baseline_exec_local_ms # default assumption to override
            t_exec_remote = t_exec_remote_est
            t_net = rtt_estimate
            
            if c_local <= c_offload:
                decision = "LOCAL"
                log.info(f"Decision: LOCAL (Network latency {rtt_estimate:.2f}ms exceeds {l_star:.2f}ms threshold or Jetson unreachable)")
                t_exec_local = execute_path_a_local(window_data)
                
                # Update rolling baseline for actual local compute speed
                baseline_exec_local_ms = (baseline_exec_local_ms * 0.8) + (t_exec_local * 0.2)
                
            else:
                decision = "OFFLOAD"
                log.info(f"Decision: OFFLOAD (Network {rtt_estimate:.2f}ms is below threshold {l_star:.2f}ms)")
                results = execute_path_b_offload(window_data)
                
                t_exec_remote = results["t_exec_remote_ms"]
                t_net = results["t_net_ms"]
                
                # If offload failed, we fall back to local to process the critical data
                if results["status"] != "success":
                    log.warning("Offload failed, enforcing local fallback.")
                    t_exec_local = execute_path_a_local(window_data)
                    decision = "FALLBACK_LOCAL"
            
            # 4. Zero-Trust Logging
            log_telemetry([
                time.time(), 
                window_index, 
                decision,
                round(t_exec_local, 2), 
                round(t_exec_remote, 2), 
                round(t_net, 2), 
                round(e_crypto, 2), 
                round(c_local, 2), 
                round(c_offload, 2), 
                round(l_star, 2)
            ])
            
        except requests.exceptions.Timeout:
            log.info("Waiting for next data window (rate limited).")
        except requests.exceptions.ConnectionError:
            log.error("Cannot connect to Sensor Emulator. Is it running?")
            time.sleep(2)
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
