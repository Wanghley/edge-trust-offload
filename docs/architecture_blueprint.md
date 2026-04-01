# EdgeTrust-Offload: Architectural Blueprint

## 1. Project Overview & Technical Goals
**Project Name:** EdgeTrust-Offload
**Core Goal:** To quantify the trade-off between network latency and security overhead in Zero-Trust edge computing environments. Specifically, the project aims to identify the **Crossover Point ($L^*$)**—the network latency threshold at which the overhead of encrypting and authenticating data makes offloading to a more powerful server slower than simply computing the task locally on the constrained edge node.

**Target Workloads for Offloading:**
The project tests workloads typical of continuous health sensing and industrial IoT, transitioning raw time-series data into lightweight features:
1. **1024-point FFT on synthetic ECG (256 Hz) / Motion Data (100 Hz MPU6050):** A compute-heavy task to extract frequency-domain features (dominant frequencies, band energy) for anomaly detection (e.g., machine faults or arrhythmia).
2. **64-tap Low-pass FIR Filter on Temperature Data (1 Hz):** A lightweight task mimicking thermal drift compensation, used to test the minimum overhead floor of the Zero-Trust architecture.

**Performance Bottlenecks Justifying Offload:**
- **Local Constraints (Edge Client):** Microcontrollers (like the ESP32-S3) lack the computational headroom, RAM (e.g., 512KB SRAM), and battery efficiency required to run complex algorithms rapidly. Running these tasks locally drains energy and takes longer (e.g., ~30ms pure processing time for an FFT).
- **Network & Security Costs:** While offloading to a Jetson Nano resolves the compute bottleneck, doing so in a HIPAA/GDPR-compliant setting mandates a Zero-Trust approach. The mandatory encryption/decryption (ChaCha20-Poly1305), authentication, and network transmission (impacted by jitter and packet loss) introduce latency. If this combined overhead ($T_{network} + T_{security}$) exceeds the local compute time, offloading becomes counterproductive.

## 2. Component Mapping (Image vs. Logical Roles)
Based on the provided physical testbed architecture (`image_ad8345.png`) and the simulation requirements (Raspberry Pi to Jetson Nano):

| Physical Component | Logical Role | IP Address (Tailscale) | Description |
| :--- | :--- | :--- | :--- |
| **Raspberry Pi 2** | **Edge Client (Source)** | `100.1.1.2` | Acts as the sensor node. It generates/collects sensor data (e.g., motion/ECG) and runs the adaptive scheduler. It calculates whether offloading is beneficial. If so, it packages the payload and transmits it over the secure tunnel. |
| **NVIDIA Jetson Nano** | **Edge Server (Compute)** | `100.1.1.4` | The high-performance destination node. It acts as the secure offload target, receiving encrypted payloads, decrypting them, and executing heavy processing (like GPU-accelerated FFT or ML analysis) before returning the results securely. |
| **Orange Pi 3B** | **Monitoring/Orchestration** | `100.1.1.3` | Facilitates network telemetry, tracking latency/jitter, and potentially hosting secondary analysis tools like `tegrastats` to observe system health. |
| **Router & Switch** | **Underlay Network** | `192.168.1.0/24` | Provides the physical layer connectivity, occasionally injected with synthetic congestion (using tools like Linux `tc`) to test system resilience to jitter. |

## 3. Offloading API Specification
To facilitate communication between the Raspberry Pi (Client) and the Jetson Nano (Server), a RESTful HTTP protocol operating over the Tailscale TCP/IP tunnel is recommended, matching the existing `main.py` implementation approach while standardizing the payload.

**Endpoint:** `POST http://100.1.1.4:8000/api/v1/offload/fft`

### RPi (Client) Request Payload
The client must send the raw time-series window along with telemetry necessary for the Server to prioritize or log the task.
```json
{
  "device_id": "rpi-client-01",
  "timestamp_ms": 1711928374123,
  "workload_type": "fft_1024",
  "payload": {
    "sampling_rate_hz": 256,
    "precision": "float32",
    "samples": [0.12, 0.45, -0.34, 0.99 /* ... exactly 1024 elements */]
  },
  "metadata": {
    "client_cpu_load_pct": 85.2,
    "battery_level_pct": 42
  }
}
```

### Jetson Nano (Server) Response Payload
The server returns the compressed, analyzed feature set, drastically reducing return-trip bandwidth (e.g., yielding just ~40 bytes of telemetry relative to the 6KB inward payload).
```json
{
  "status": "success",
  "processing_node": "jetson-nano",
  "compute_latency_ms": 4.12,
  "result": {
    "dominant_frequency_hz": [5.0, 12.4, 60.1],
    "total_band_energy": 145.2,
    "anomaly_flag": false,
    "confidence_score": 0.98
  }
}
```

## 4. How Tailscale Simplifies Zero-Trust Architecture
In the EdgeTrust model, data must travel securely across a potentially hostile or dynamically changing local network. Tailscale provides the essential Zero-Trust networking layer with minimal configuration via the following mechanisms:

1. **Identity-Based Access Control (ACLs):** Instead of using fragile IP-address or subnet-based firewall rules, Tailscale secures the connection based on machine identity. The Raspberry Pi and Jetson Nano are authenticated within the Tailnet. Strict ACLs can ensure that *only* the specific Raspberry Pi has port-level access to the Jetson Nano's compute API, enforcing the principle of least privilege.
2. **Elimination of Port Forwarding:** Tailscale handles NAT traversal using STUN/TURN (DERP relays if direct connections fail). The Jetson Nano can run an offloading server without opening any inbound ports on the physical network switch or router, eliminating the attack surface from automated network scanners.
3. **Automated Key Management:** True Zero Trust relies on robust encryption, but managing Public Key Infrastructure (PKI) and rotating certificates on edge IoT nodes is notoriously difficult. Tailscale automatically generates and rotates WireGuard cryptographic keys in the background, abstracting away massive operational complexity.
4. **WireGuard's Predictable Overhead:** Because Tailscale is built on WireGuard, it uses a highly optimized, in-kernel (and soon userspace) ChaCha20-Poly1305 encryption mechanism. This establishes a predictable, deterministic encryption overhead for the $T_{search}$ variable in the offloading decision equation without crippling system performance.
