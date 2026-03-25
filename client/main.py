import time
import psutil
import numpy as np
import requests

# get from .env or config file in production
import os
from dotenv import load_dotenv
load_dotenv()

# --- Configuration (Move to common/constants.py later) ---
JETSON_IP = os.getenv("JETSON_IP")
OFFLOAD_URL = f"http://{JETSON_IP}:8000/analyze"
TRUST_THRESHOLD = os.getenv("TRUST_THRESHOLD", 1.0) # Noise level above which we consider it 'Low Trust'
CPU_OFFLOAD_LIMIT = os.getenv("CPU_OFFLOAD_LIMIT", 80) # CPU usage percentage above which we consider it 'Overloaded'

def generate_health_data():
    """Simulates hand kinematics data with occasional high-frequency noise."""
    t = np.linspace(0, 1, 100)
    # Base signal (simulating a 5Hz hand movement)
    signal = np.sin(2 * np.pi * 5 * t)
    # Inject synthetic noise to simulate 'Low Trust' scenarios
    if np.random.random() > 0.8:
        signal += np.random.normal(0, 2.5, 100)
    return signal.tolist()

def get_offload_decision(data, cpu_usage):
    """
    Implements the Offloading Equation:
    Decision = Offload if (Complexity / Local_Res) > (Network_Delay + Remote_Res)
    """
    signal_noise = np.std(data)
    is_complex = signal_noise > TRUST_THRESHOLD
    is_overloaded = cpu_usage > CPU_OFFLOAD_LIMIT

    if is_complex or is_overloaded:
        reason = "High Noise/Complexity" if is_complex else "High CPU Load"
        return True, reason
    return False, "Nominal"

def main():
    print(f"Starting EdgeTrust-Offload on Raspberry Pi...")
    while True:
        # 1. Collect synthetic health data
        data = generate_health_data()
        current_cpu = psutil.cpu_percent()

        # 2. Evaluate the offloading equation
        should_offload, reason = get_offload_decision(data, current_cpu)

        if should_offload:
            print(f"[OFFLOAD] {reason}. Sending to Jetson Nano...")
            try:
                start = time.time()
                response = requests.post(OFFLOAD_URL, json={"data": data}, timeout=3)
                latency = (time.time() - start) * 1000
                print(f" -> Result: {response.json()['status']} | Latency: {latency:.2f}ms")
            except Exception as e:
                print(f" -> Failed to offload: {e}")
        else:
            print(f"[LOCAL] Processing on Raspberry Pi...")
            # Simulate local processing (e.g., simple threshold check)
            _ = np.mean(data)
            time.sleep(0.05)

        time.sleep(1) # Interval between monitoring windows

if __name__ == "__main__":
    main()