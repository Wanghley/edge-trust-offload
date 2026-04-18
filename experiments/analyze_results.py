#!/usr/bin/env python3
"""
EdgeTrust-Offload Result Analysis & Plotting
=============================================
Reads the JSONL output from run_experiments.py and produces:
  - Console summary table
  - crossover_results.png  (for inclusion in the LaTeX report)
  - security_overhead.png  (S2 vs S3 latency delta)
  - jitter_analysis.png    (CDF of latency per scenario)

Usage
-----
  python experiments/analyze_results.py results/experiment_YYYYMMDD_HHMMSS/
  python experiments/analyze_results.py results/experiment_YYYYMMDD_HHMMSS/ --no-plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless on RPi
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed — plots skipped. pip install matplotlib")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SCENARIO_LABELS = {
    "S1": "Local (RPi 4)",
    "S2": "Insecure offload",
    "S3": "Secure offload (ZTA)",
    "S4": "Secure + congestion",
}

SCENARIO_COLORS = {
    "S1": "#2196F3",
    "S2": "#4CAF50",
    "S3": "#FF9800",
    "S4": "#F44336",
}


def load_raw(exp_dir: Path, scenario: str, complexity: str) -> List[dict]:
    fname_map = {"S1": "s1_local", "S2": "s2_insecure", "S3": "s3_secure", "S4": "s4_congested"}
    fpath = exp_dir / "raw" / f"{fname_map[scenario]}_{complexity}.jsonl"
    if not fpath.exists():
        return []
    records = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_summary(exp_dir: Path) -> List[dict]:
    p = exp_dir / "summary.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def load_crossover(exp_dir: Path) -> dict:
    p = exp_dir / "crossover.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(summaries: List[dict], crossover: dict) -> None:
    print("\n" + "=" * 90)
    print("EdgeTrust-Offload Experiment Results")
    print("=" * 90)
    print(f"{'Scenario':<26} {'Complexity':<10} {'N':<5} {'Mean(ms)':<10} {'Std':<8} {'p50':<8} {'p95':<8} {'p99':<8} {'L*(ms)':<8}")
    print("-" * 90)
    for s in summaries:
        label = SCENARIO_LABELS.get(s["scenario"], s["scenario"])
        print(
            f"{label:<26} {s['complexity']:<10} {s['n_trials']:<5} "
            f"{s['t_total_mean']:<10.1f} {s['t_total_std']:<8.1f} "
            f"{s['t_total_p50']:<8.1f} {s['t_total_p95']:<8.1f} {s['t_total_p99']:<8.1f} "
            f"{s['l_star_mean']:<8.1f}"
        )
    print("=" * 90)

    print("\nCrossover latency L* (ms) — network RTT threshold for beneficial offloading:")
    for scenario, complexities in crossover.items():
        label = SCENARIO_LABELS.get(scenario, scenario)
        for complexity, l_star in complexities.items():
            print(f"  {label:<26} [{complexity:<6}] L* = {l_star:.1f} ms")

    # Security overhead (S3 vs S2)
    s2 = {s["complexity"]: s for s in summaries if s["scenario"] == "S2"}
    s3 = {s["complexity"]: s for s in summaries if s["scenario"] == "S3"}
    if s2 and s3:
        print("\nZero-Trust security overhead (S3 - S2, ms):")
        for cx in ("normal", "high"):
            if cx in s2 and cx in s3:
                delta = s3[cx]["t_total_mean"] - s2[cx]["t_total_mean"]
                print(f"  [{cx:<6}] {delta:+.2f} ms mean  ({s3[cx]['e_crypto_mean']:.2f} ms estimated crypto)")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_crossover(exp_dir: Path, summaries: List[dict]) -> None:
    """Latency vs complexity bar chart + variance box plot."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: mean latency grouped by complexity
    scenarios = ["S1", "S2", "S3", "S4"]
    complexities = ["normal", "high"]
    x = np.arange(len(scenarios))
    width = 0.35

    ax = axes[0]
    for i, cx in enumerate(complexities):
        means = []
        stds = []
        for sc in scenarios:
            match = [s for s in summaries if s["scenario"] == sc and s["complexity"] == cx]
            if match:
                means.append(match[0]["t_total_mean"])
                stds.append(match[0]["t_total_std"])
            else:
                means.append(0)
                stds.append(0)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, label=cx.capitalize(), yerr=stds, capsize=4, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("End-to-end latency (ms)")
    ax.set_title("Latency vs. Scenario and Complexity")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: p95 comparison
    ax2 = axes[1]
    for i, cx in enumerate(complexities):
        p95s = []
        for sc in scenarios:
            match = [s for s in summaries if s["scenario"] == sc and s["complexity"] == cx]
            if match:
                p95s.append(match[0]["t_total_p95"])
            else:
                p95s.append(0)
        offset = (i - 0.5) * width
        ax2.bar(x + offset, p95s, width, label=cx.capitalize(), alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios], rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("p95 latency (ms)")
    ax2.set_title("95th Percentile Latency")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = exp_dir / "crossover_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {out}")


def plot_latency_cdf(exp_dir: Path) -> None:
    """Empirical CDF of end-to-end latency per scenario (normal complexity)."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = ["S1", "S2", "S3", "S4"]
    for sc in scenarios:
        records = load_raw(exp_dir, sc, "normal")
        if not records:
            continue
        vals = sorted(r["t_total_ms"] for r in records)
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, label=SCENARIO_LABELS.get(sc, sc), color=SCENARIO_COLORS[sc], linewidth=2)

    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF — Normal Complexity")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = exp_dir / "latency_cdf.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {out}")


def plot_security_overhead(exp_dir: Path, summaries: List[dict]) -> None:
    """S2 vs S3 per-complexity delta bar chart."""
    if not HAS_MPL:
        return

    s2 = {s["complexity"]: s for s in summaries if s["scenario"] == "S2"}
    s3 = {s["complexity"]: s for s in summaries if s["scenario"] == "S3"}
    if not (s2 and s3):
        return

    cxs = [cx for cx in ("normal", "high") if cx in s2 and cx in s3]
    deltas_mean = [s3[cx]["t_total_mean"] - s2[cx]["t_total_mean"] for cx in cxs]
    crypto_est  = [s3[cx]["e_crypto_mean"] for cx in cxs]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(cxs))
    ax.bar(x - 0.2, deltas_mean, 0.35, label="Measured overhead (S3−S2)", color="#FF9800", alpha=0.85)
    ax.bar(x + 0.2, crypto_est,  0.35, label="Estimated crypto latency", color="#9C27B0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in cxs])
    ax.set_ylabel("Latency delta (ms)")
    ax.set_title("Zero-Trust Encryption Overhead (S3 vs S2)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = exp_dir / "security_overhead.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Analyze EdgeTrust-Offload experiment results")
    p.add_argument("exp_dir", type=Path, help="Experiment output directory")
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = p.parse_args()

    if not args.exp_dir.exists():
        print(f"ERROR: {args.exp_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    summaries = load_summary(args.exp_dir)
    crossover = load_crossover(args.exp_dir)

    if not summaries:
        print(f"No summary.json found in {args.exp_dir}. Run run_experiments.py first.")
        sys.exit(1)

    print_report(summaries, crossover)

    if not args.no_plots:
        plot_crossover(args.exp_dir, summaries)
        plot_latency_cdf(args.exp_dir)
        plot_security_overhead(args.exp_dir, summaries)
        print(f"\nPlots saved in {args.exp_dir}/")


if __name__ == "__main__":
    main()
