import time
import json
import logging
import os
import csv
import socket
import argparse
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

# tflite_runtime (Python ≤3.11) or ai_edge_litert (Python 3.12+, Google's replacement)
HAS_TFLITE = False
for _tflite_mod in ("tflite_runtime.interpreter", "ai_edge_litert.interpreter"):
    try:
        __import__(_tflite_mod)
        HAS_TFLITE = True
        break
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Logging & Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("EdgeTrust_Scheduler")

TELEMETRY_LOG = Path(__file__).parent.parent / "offload_telemetry.csv"


class EdgeTrustScheduler:
    def __init__(self, args):
        self.emulator_url = args.emulator_url
        self.jetson_ip = args.jetson_ip
        self.jetson_port = args.jetson_port
        self.device_id = args.device_id
        
        self.alpha_weight = args.alpha
        self.beta_weight = args.beta
        self.gamma_weight = args.gamma
        self.delta_weight = args.delta
        
        self.offload_url = f"http://{self.jetson_ip}:{self.jetson_port}/api/v1/offload/fft"
        self.health_url = f"http://{self.jetson_ip}:{self.jetson_port}/health"

        self.engine_cache = {}
        self.baseline_exec_local_ms = 35.0  # Safe initial default
        
        # Use a single requests.Session for Keep-Alive and TCP pooling
        self.session = requests.Session()
        
        # Optionally attach custom headers
        self.session.headers.update({"X-Device-ID": self.device_id})

    def init_telemetry(self):
        if not TELEMETRY_LOG.exists():
            with open(TELEMETRY_LOG, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "window_index", "decision", 
                    "t_exec_local_ms", "t_exec_remote_ms", "t_net_ms", 
                    "e_crypto_ms", "c_local", "c_offload", "l_star_ms"
                ])

    def log_telemetry(self, row):
        with open(TELEMETRY_LOG, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def test_jetson_connection(self) -> bool:
        log.info(f"Testing Jetson telemtry connection at {self.health_url}...")
        try:
            resp = self.session.get(self.health_url, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                stats = data.get("system_stats", {})
                log.info(f"[Server OK] Backend: {data.get('inference_backend')} | "
                         f"GPU: {stats.get('gpu_temp_c')}°C | "
                         f"RAM: {stats.get('ram_used_mb')} MB")
                return True
            else:
                log.warning(f"Connection test failed. Status code: {resp.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            log.error(f"Cannot reach Jetson: {e}")
            return False

    def estimate_crypto_overhead(self, payload_size_bytes: int) -> float:
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

    def execute_path_a_local(self, window_data: dict) -> float:
        """
        Executes a local inference task via edge_tasks.
        Returns the execution time in ms.
        """
        if HAS_EDGE_TASKS:
            samples = np.array(window_data["payload"]["samples"])
            sampling_rate = window_data["payload"].get("sampling_rate_hz", 256.0)
            n_channels = window_data["payload"].get("n_channels", len(samples[0]) if len(samples) > 0 else 1)
            
            if n_channels not in self.engine_cache:
                log.debug(f"Initializing Inference Engine for {n_channels} channels...")
                self.engine_cache[n_channels] = GestureInferenceEngine(n_channels=n_channels, complexity="normal")

            result = run_local_benchmark(
                window=samples,
                sampling_rate_hz=sampling_rate,
                complexity="normal", 
                engine=self.engine_cache[n_channels],
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

    def measure_network_rtt(self) -> (float, bool):
        start = time.perf_counter()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)
            sock.connect((self.jetson_ip, int(self.jetson_port)))
            sock.close()
            rtt_ms = (time.perf_counter() - start) * 1000
            return rtt_ms, False
        except (socket.timeout, ConnectionRefusedError, OSError):
            return 150.0, True 

    def execute_path_b_offload(self, window_data: dict) -> dict:
        payload = {
            "device_id": self.device_id,
            "timestamp_ms": int(time.time() * 1000),
            "workload_type": "fft_1024",
            "payload": window_data["payload"],
            "metadata": {
                "client_cpu_load_pct": 50.0,
                "battery_level_pct": 100
            }
        }
        
        t_net_start = time.perf_counter()
        
        processing_time_remote = 0.0
        network_latency = 0.0
        status = "failed"
        packet_loss = False
        
        try:
            # Using Session object prevents reconnections (reduces TCP overhead)
            req = self.session.post(self.offload_url, json=payload, timeout=2.0)
            t_net_end = time.perf_counter()
            
            if req.status_code == 200:
                res_json = req.json()
                processing_time_remote = res_json.get("compute_latency_ms", 5.0)
                network_latency = (t_net_end - t_net_start) * 1000 - processing_time_remote
                status = "success"
                
                # Verbose backend output
                log.debug(f"[Offload OK] Gesture: {res_json['result'].get('gesture')} | " 
                          f"Comp. Latency: {processing_time_remote:.2f}ms")
            else:
                log.warning(f"Offload failed with status {req.status_code}: {req.text}")
                network_latency = (time.perf_counter() - t_net_start) * 1000
                packet_loss = True
                
        except requests.RequestException as e:
            log.error(f"Network error during offload: {e}")
            network_latency = (time.perf_counter() - t_net_start) * 1000
            packet_loss = True
            processing_time_remote = 4.0

        return {
            "status": status,
            "t_exec_remote_ms": processing_time_remote,
            "t_net_ms": network_latency,
            "packet_loss": packet_loss
        }

    def run(self):
        log.info(f"Starting EdgeTrust Main Scheduler for {self.device_id}")
        self.init_telemetry()
        
        # Test Jetson on startup
        self.test_jetson_connection()
        
        log.info("Profiling local computation baseline...")
        
        while True:
            try:
                # 1. Acquire Data using persistent connection
                response = self.session.get(self.emulator_url, timeout=10.0)
                if response.status_code != 200:
                    log.warning(f"Failed to fetch window: {response.status_code}")
                    time.sleep(1)
                    continue
                    
                window_data = response.json()
                window_index = window_data["window_index"]
                payload_size = len(json.dumps(window_data["payload"]))
                
                # 2. Decision Engine Estimates
                e_crypto = self.estimate_crypto_overhead(payload_size)
                rtt_estimate, packet_loss_detected = self.measure_network_rtt()
                t_exec_remote_est = 5.0 
                
                c_local = self.alpha_weight * self.baseline_exec_local_ms
                c_offload = (self.alpha_weight * t_exec_remote_est) + (self.beta_weight * rtt_estimate) + \
                            (self.gamma_weight * e_crypto) + (self.delta_weight * (500.0 if packet_loss_detected else 0.0))
                
                l_star = (c_local - (self.alpha_weight * t_exec_remote_est) - \
                         (self.gamma_weight * e_crypto) - (self.delta_weight * (500.0 if packet_loss_detected else 0.0))) / self.beta_weight
                
                log.debug(f"--- Window {window_index} --- | Cost L: {c_local:.2f} | Cost O: {c_offload:.2f} | L*: {l_star:.2f}ms")
                
                # 3. Execution Path Selection
                t_exec_local = self.baseline_exec_local_ms 
                t_exec_remote = t_exec_remote_est
                t_net = rtt_estimate
                
                if c_local <= c_offload:
                    decision = "LOCAL"
                    log.info(f"Decision: LOCAL (Net: {rtt_estimate:.2f}ms > Threshold {l_star:.2f}ms)")
                    t_exec_local = self.execute_path_a_local(window_data)
                    self.baseline_exec_local_ms = (self.baseline_exec_local_ms * 0.8) + (t_exec_local * 0.2)
                else:
                    decision = "OFFLOAD"
                    log.info(f"Decision: OFFLOAD (Net: {rtt_estimate:.2f}ms <= Threshold {l_star:.2f}ms)")
                    results = self.execute_path_b_offload(window_data)
                    
                    t_exec_remote = results["t_exec_remote_ms"]
                    t_net = results["t_net_ms"]
                    
                    if results["status"] != "success":
                        log.warning("Offload failed, enforcing local fallback.")
                        t_exec_local = self.execute_path_a_local(window_data)
                        decision = "FALLBACK_LOCAL"
                
                # 4. Zero-Trust Logging
                self.log_telemetry([
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
                log.info("Waiting for data window (rate limited).")
            except requests.exceptions.ConnectionError:
                log.error("Cannot connect to Sensor Emulator. Retrying in 2s...")
                time.sleep(2)
            except Exception as e:
                log.error(f"Unexpected error: {e}")
                time.sleep(1)


def parse_args():
    parser = argparse.ArgumentParser(description="EdgeTrust Main Scheduler (Client)")
    parser.add_argument("--jetson-ip", type=str, default=os.getenv("JETSON_IP", "100.1.1.4"), help="Jetson Tailscale IP")
    parser.add_argument("--jetson-port", type=str, default=os.getenv("JETSON_PORT", "8000"), help="Jetson Port")
    parser.add_argument("--emulator-url", type=str, default=os.getenv("EMULATOR_URL", "http://127.0.0.1:9000/get_window"), help="Sensor emulator endpoint")
    parser.add_argument("--device-id", type=str, default="rpi-client-scheduler", help="Our Device ID for Zero-Trust validation")
    
    # Debug / Testing Options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable highly verbose debug logging")
    parser.add_argument("--test-connection", action="store_true", help="Hit the Jetson /health endpoint, print telemetry, and then exit immediately")
    
    # Simulation weights
    parser.add_argument("--alpha", type=float, default=float(os.getenv("WEIGHT_ALPHA", "1.0")), help="Execution time weight")
    parser.add_argument("--beta", type=float, default=float(os.getenv("WEIGHT_BETA", "1.0")), help="Network Latency weight")
    parser.add_argument("--gamma", type=float, default=float(os.getenv("WEIGHT_GAMMA", "1.0")), help="Cryptographic Overhead weight")
    parser.add_argument("--delta", type=float, default=float(os.getenv("WEIGHT_DELTA", "1.0")), help="Packet Loss Penalty weight")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)
        log.debug("Verbose logging enabled.")

    scheduler = EdgeTrustScheduler(args)

    if args.test_connection:
        log.info("Running Jetson connection test mode.")
        success = scheduler.test_jetson_connection()
        if success:
            log.info("Connection successful. Exiting due to --test-connection.")
            sys.exit(0)
        else:
            log.error("Connection failed under --test-connection. Exiting with error.")
            sys.exit(1)

    # Otherwise, start normal loop
    scheduler.run()


if __name__ == "__main__":
    main()
