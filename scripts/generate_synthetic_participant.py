"""
generate_synthetic_participant.py
==================================
Generate a schema-accurate synthetic HDF5 file for one participant (default:
B002) that is structurally identical to BiomechanicsDataset.hdf5.

This lets the full EdgeTrust pipeline (client/sensor_emulator.py, setup_data.py,
client/main_scheduler.py) run without downloading the real 54 GB dataset.

BHaM dataset structure (from the published paper & collection notes):
----------------------------------------------------------------------
  /B002/
    ├── <task_name>   shape=(N, n_channels)  dtype=float32
    │                 Where N = task_duration_s * sampling_rate_hz
    └── ...  (19 tasks total across elbow, wrist, hand)

Signal channels reproduced here:
  EMG (8 surface + 4 fine-wire = 12 channels, µV scale, 2000 Hz)
  Kinematics (marker-based mocap, 39 channels, mm/deg, 100 Hz)

Usage
-----
    python scripts/generate_synthetic_participant.py
    python scripts/generate_synthetic_participant.py --participant-id B005 --out data/raw/BiomechanicsDataset.hdf5
    python scripts/generate_synthetic_participant.py --list-tasks
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# BHaM task catalogue  (name → duration_seconds)
# Source: BHaM collection notes + nihpp-2025.08.21.671503v1.pdf
# ---------------------------------------------------------------------------
TASKS: Dict[str, int] = {
    "IndexFlexion_MVC":        10,
    "IndexFlexion_30pct":      20,
    "IndexFlexion_60pct":      20,
    "MiddleFlexion_MVC":       10,
    "MiddleFlexion_30pct":     20,
    "MiddleFlexion_60pct":     20,
    "PinchGrip_MVC":           10,
    "PinchGrip_30pct":         20,
    "PinchGrip_60pct":         20,
    "PowerGrip_MVC":           10,
    "PowerGrip_30pct":         20,
    "PowerGrip_60pct":         20,
    "WristFlexion_MVC":        10,
    "WristExtension_MVC":      10,
    "WristRadialDeviation":    15,
    "WristUlnarDeviation":     15,
    "ElbowFlexion":            15,
    "PronoSupination":         15,
    "RestBaseline":            30,
}

# ---------------------------------------------------------------------------
# Channel definitions
# ---------------------------------------------------------------------------

# Surface EMG channels (µV, broadband 20–450 Hz, sampled at 2000 Hz)
EMG_SURFACE_CHANNELS: List[str] = [
    "EMG_FDS_surf",   # Flexor Digitorum Superficialis
    "EMG_FDP_surf",   # Flexor Digitorum Profundus
    "EMG_EDC_surf",   # Extensor Digitorum Communis
    "EMG_FCR_surf",   # Flexor Carpi Radialis
    "EMG_ECR_surf",   # Extensor Carpi Radialis
    "EMG_FCU_surf",   # Flexor Carpi Ulnaris
    "EMG_ECU_surf",   # Extensor Carpi Ulnaris
    "EMG_BB_surf",    # Biceps Brachii
]

# Fine-wire EMG channels (µV, 2000 Hz)
EMG_FINEWIRE_CHANNELS: List[str] = [
    "EMG_FDS_fw",
    "EMG_FDP_fw",
    "EMG_Lumb_fw",    # Lumbrical
    "EMG_DI_fw",      # Dorsal Interosseous
]

# Kinematics — marker-based motion capture (mm position + deg angles, 100 Hz)
KINEMATIC_CHANNELS: List[str] = [
    # Finger joint angles (deg)
    "Index_MCP_flex", "Index_PIP_flex", "Index_DIP_flex",
    "Middle_MCP_flex", "Middle_PIP_flex", "Middle_DIP_flex",
    "Ring_MCP_flex",   "Ring_PIP_flex",   "Ring_DIP_flex",
    "Little_MCP_flex", "Little_PIP_flex", "Little_DIP_flex",
    "Thumb_CMC_flex",  "Thumb_MCP_flex",  "Thumb_IP_flex",
    # Wrist angles (deg)
    "Wrist_flex_ext",  "Wrist_rad_uln",   "Forearm_pro_sup",
    # Elbow
    "Elbow_flex_ext",
    # Marker positions — fingertip clusters (mm, x/y/z)
    "Index_tip_x", "Index_tip_y", "Index_tip_z",
    "Middle_tip_x", "Middle_tip_y", "Middle_tip_z",
    "Thumb_tip_x",  "Thumb_tip_y",  "Thumb_tip_z",
    # Wrist markers
    "Rad_styloid_x", "Rad_styloid_y", "Rad_styloid_z",
    "Uln_styloid_x", "Uln_styloid_y", "Uln_styloid_z",
    # Force/torque (N, N·m) from load cell
    "Force_x", "Force_y", "Force_z",
    "Torque_x", "Torque_y", "Torque_z",
]

# ---------------------------------------------------------------------------
# Synthetic signal generators
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(seed=42)  # reproducible


def _emg_signal(n_samples: int, fs: int, mvc_fraction: float = 0.3) -> np.ndarray:
    """
    Simulate a band-limited surface EMG signal.
    Model: Gaussian noise band-pass filtered to 20–450 Hz, amplitude scaled
    to mvc_fraction of a typical MVC level (~500 µV RMS).
    """
    # White noise → simulate broadband EMG
    noise = RNG.standard_normal(n_samples).astype(np.float32)

    # Simple first-order high-pass to remove DC (simulate 20 Hz cutoff)
    alpha = 0.97  # RC decay
    hp = np.zeros_like(noise)
    hp[0] = noise[0]
    for i in range(1, n_samples):
        hp[i] = alpha * (hp[i - 1] + noise[i] - noise[i - 1])

    # Scale to realistic µV amplitude
    rms_target = 500.0 * mvc_fraction  # µV
    current_rms = float(np.sqrt(np.mean(hp ** 2))) or 1.0
    hp = hp * (rms_target / current_rms)

    # Add occasional burst (muscle activation transient)
    burst_start = RNG.integers(n_samples // 4, 3 * n_samples // 4)
    burst_len = min(int(fs * 0.5), n_samples - burst_start)
    envelope = np.hanning(burst_len).astype(np.float32)
    hp[burst_start: burst_start + burst_len] += envelope * rms_target * 1.5

    return hp.astype(np.float32)


def _kinematic_signal(n_samples: int, fs: int, channel: str) -> np.ndarray:
    """
    Simulate a smooth kinematic trajectory using a low-frequency sinusoid
    plus low-amplitude noise, plausible for the named channel.
    """
    t = np.linspace(0, n_samples / fs, n_samples, dtype=np.float32)

    if "flex" in channel.lower() or "ext" in channel.lower():
        # Joint angle (deg): 0° rest → ~45° peak
        centre, amplitude, freq = 20.0, 25.0, 0.5
    elif "_tip_" in channel or "styloid" in channel:
        # Marker position (mm): ~200 mm baseline with small movement
        centre, amplitude, freq = 200.0, 10.0, 0.5
    elif "Force" in channel:
        centre, amplitude, freq = 0.0, 15.0, 0.5   # N
    elif "Torque" in channel:
        centre, amplitude, freq = 0.0, 2.0, 0.5    # N·m
    else:
        centre, amplitude, freq = 0.0, 5.0, 0.3

    phase = RNG.uniform(0, 2 * np.pi)
    signal = centre + amplitude * np.sin(2 * np.pi * freq * t + phase)
    noise = RNG.normal(0, amplitude * 0.05, n_samples).astype(np.float32)
    return (signal + noise).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_participant_dataset(
    participant_id: str,
    tasks: Dict[str, int],
    emg_fs: int = 2000,
    kin_fs: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Return a dict  task_name → np.ndarray  (shape: N_emg_samples × n_channels)

    We align everything to the EMG sampling rate (2000 Hz) by up-sampling
    kinematics via linear interpolation.  This matches the real BHaM HDF5
    layout where each task is a single 2D array.
    """
    all_emg = EMG_SURFACE_CHANNELS + EMG_FINEWIRE_CHANNELS
    n_emg_ch = len(all_emg)
    n_kin_ch = len(KINEMATIC_CHANNELS)
    n_total = n_emg_ch + n_kin_ch
    all_channels = all_emg + KINEMATIC_CHANNELS

    datasets: Dict[str, np.ndarray] = {}

    for task_name, duration_s in tasks.items():
        n_emg_samples = duration_s * emg_fs
        n_kin_samples = duration_s * kin_fs

        # Build EMG columns
        mvc_map = {
            "MVC": 1.0, "60pct": 0.6, "30pct": 0.3,
            "Rest": 0.05,
        }
        mvc_frac = next(
            (v for k, v in mvc_map.items() if k in task_name), 0.4
        )

        emg_data = np.column_stack([
            _emg_signal(n_emg_samples, emg_fs, mvc_frac)
            for _ in all_emg
        ])  # shape (N_emg, 12)

        # Build kinematic columns at native 100 Hz then upsample to 2000 Hz
        kin_native = np.column_stack([
            _kinematic_signal(n_kin_samples, kin_fs, ch)
            for ch in KINEMATIC_CHANNELS
        ])  # shape (N_kin, 39)

        # Linear interpolation upsample
        x_kin = np.linspace(0, 1, n_kin_samples)
        x_emg = np.linspace(0, 1, n_emg_samples)
        kin_up = np.column_stack([
            np.interp(x_emg, x_kin, kin_native[:, c])
            for c in range(n_kin_ch)
        ]).astype(np.float32)  # shape (N_emg, 39)

        # Combine → shape (N_emg, 51)
        combined = np.concatenate([emg_data, kin_up], axis=1)
        datasets[task_name] = combined
        print(
            f"  [{participant_id}/{task_name}]  "
            f"shape={combined.shape}  "
            f"({duration_s}s × {emg_fs}Hz × {n_total}ch)"
        )

    return datasets, all_channels


def write_hdf5(
    output_path: str,
    participant_id: str,
    datasets: Dict[str, np.ndarray],
    channel_names: List[str],
    overwrite: bool = False,
) -> None:
    """Write the generated arrays into an HDF5 file mirroring BHaM layout."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    mode = "a"  # append — add participant group without clobbering others
    if overwrite and os.path.exists(output_path):
        os.remove(output_path)

    with h5py.File(output_path, mode) as h5:
        if participant_id in h5:
            print(
                f"  Participant '{participant_id}' already in {output_path}. "
                "Use --overwrite to replace."
            )
            return

        grp = h5.require_group(participant_id)
        grp.attrs["synthetic"] = True
        grp.attrs["channel_names"] = channel_names
        grp.attrs["emg_channels"] = EMG_SURFACE_CHANNELS + EMG_FINEWIRE_CHANNELS
        grp.attrs["kinematic_channels"] = KINEMATIC_CHANNELS
        grp.attrs["sampling_rate_hz"] = 2000
        grp.attrs["source"] = "generate_synthetic_participant.py"

        for task_name, array in datasets.items():
            ds = grp.create_dataset(
                task_name,
                data=array,
                compression="gzip",
                compression_opts=4,
                chunks=(min(2000, array.shape[0]), array.shape[1]),
            )
            ds.attrs["n_samples"] = array.shape[0]
            ds.attrs["n_channels"] = array.shape[1]
            ds.attrs["channel_names"] = channel_names

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\nWrote {output_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic BHaM-schema HDF5 file for a single participant. "
            "Produces schema-identical data to BiomechanicsDataset.hdf5 without "
            "requiring a 54 GB download."
        )
    )
    parser.add_argument(
        "--participant-id",
        default="B002",
        help="Participant label to write (default: B002)",
    )
    parser.add_argument(
        "--out",
        default="data/raw/BiomechanicsDataset.hdf5",
        help="Output HDF5 path (default: data/raw/BiomechanicsDataset.hdf5)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        metavar="TASK",
        help=(
            "Subset of tasks to generate (default: all 19). "
            "Example: --tasks IndexFlexion_MVC PowerGrip_30pct"
        ),
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print all available task names and exit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the HDF5 file if it already exists.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_tasks:
        print("Available tasks:")
        for name, dur in TASKS.items():
            print(f"  {name:<30}  {dur}s")
        return 0

    tasks = TASKS
    if args.tasks:
        unknown = set(args.tasks) - set(TASKS)
        if unknown:
            print(f"Error: Unknown tasks: {unknown}", file=sys.stderr)
            print(f"Valid tasks: {list(TASKS.keys())}", file=sys.stderr)
            return 1
        tasks = {k: v for k, v in TASKS.items() if k in args.tasks}

    print(
        f"Generating synthetic data for participant '{args.participant_id}' "
        f"({len(tasks)} tasks)...\n"
    )
    datasets, channel_names = build_participant_dataset(args.participant_id, tasks)

    print(f"\nWriting to {args.out}...")
    write_hdf5(
        output_path=args.out,
        participant_id=args.participant_id,
        datasets=datasets,
        channel_names=channel_names,
        overwrite=args.overwrite,
    )

    print(
        f"\nDone! You can now run:\n"
        f"  python client/sensor_emulator.py "
        f"--participant-id {args.participant_id} "
        f"--task-name IndexFlexion_MVC\n"
        f"  python scripts/setup_data.py "
        f"--participant-id {args.participant_id} "
        f"--task-name IndexFlexion_MVC "
        f"--hdf5-path {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
