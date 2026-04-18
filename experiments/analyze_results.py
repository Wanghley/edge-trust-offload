#!/usr/bin/env python3
"""
EdgeTrust-Offload Result Analysis & Plotting
=============================================
Reads output from run_experiments.py (or a merged directory) and produces:
  - Console summary table
  - crossover_results.png   — mean latency per scenario (for LaTeX report)
  - congestion_sweep.png    — S4 light/mid/heavy comparison
  - latency_cdf.png         — empirical CDF per scenario (normal complexity)
  - security_overhead.png   — S2 vs S3 encryption delta

Usage
-----
  # Standard single-run directory
  python experiments/analyze_results.py results/experiment_20260417_210340/

  # Merged directory (after running the merge script)
  python experiments/analyze_results.py results/experiment_merged/

  # Skip plot generation (table only)
  python experiments/analyze_results.py results/experiment_merged/ --no-plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless — works on RPi with no display
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed — plots skipped.  pip install matplotlib")


# ---------------------------------------------------------------------------
# Scenario metadata — covers base scenarios + three congestion levels
# ---------------------------------------------------------------------------

SCENARIO_LABELS = {
    "S1":       "Local (RPi 3B)",
    "S2":       "Insecure offload (LAN)",
    "S3":       "Secure offload (ZTA)",
    "S4":       "Secure + congestion",
    "s4_light": "Congestion: light\n(10ms/3ms/0.5%)",
    "s4_mid":   "Congestion: medium\n(25ms/8ms/2%)",
    "s4_heavy": "Congestion: heavy\n(60ms/20ms/5%)",
}

SCENARIO_COLORS = {
    "S1":       "#2196F3",
    "S2":       "#4CAF50",
    "S3":       "#FF9800",
    "S4":       "#F44336",
    "s4_light": "#FFCDD2",
    "s4_mid":   "#F44336",
    "s4_heavy": "#B71C1C",
}

# Display order for the main bar chart
SCENARIO_ORDER = ["S1", "S2", "S3", "S4", "s4_light", "s4_mid", "s4_heavy"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _infer_scenario_from_stem(stem: str) -> tuple[str, str]:
    """
    Map a JSONL filename stem to (scenario_key, complexity).
    e.g. 's1_local_normal'   → ('S1', 'normal')
         's4_congested_high' → ('S4', 'high')
         's4_light_normal'   → ('s4_light', 'normal')
    """
    fixed = {
        "s1_local":      "S1",
        "s2_insecure":   "S2",
        "s3_secure":     "S3",
        "s4_congested":  "S4",
        "s4_light":      "s4_light",
        "s4_mid":        "s4_mid",
        "s4_heavy":      "s4_heavy",
    }
    parts = stem.rsplit("_", 1)          # split off the trailing complexity
    scenario_key = fixed.get(parts[0], parts[0])
    complexity = parts[1] if len(parts) == 2 else "unknown"
    return scenario_key, complexity


def load_raw_by_stem(exp_dir: Path, stem: str) -> List[dict]:
    """Load a JSONL file by its full stem (e.g. 's3_secure_normal')."""
    fpath = exp_dir / "raw" / f"{stem}.jsonl"
    if not fpath.exists():
        return []
    records = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_all_raw(exp_dir: Path) -> Dict[str, Dict[str, List[dict]]]:
    """
    Returns  {scenario_key: {complexity: [records]}}
    by scanning all *.jsonl files in raw/.
    """
    raw_dir = exp_dir / "raw"
    result: Dict[str, Dict[str, List[dict]]] = {}
    if not raw_dir.exists():
        return result
    for fpath in sorted(raw_dir.glob("*.jsonl")):
        scenario, complexity = _infer_scenario_from_stem(fpath.stem)
        records = []
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if records:
            result.setdefault(scenario, {})[complexity] = records
    return result


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


def build_summary_from_raw(raw: Dict[str, Dict[str, List[dict]]]) -> List[dict]:
    """Build summary stats directly from raw records (no summary.json needed)."""
    summaries = []
    for scenario, complexities in raw.items():
        for complexity, records in complexities.items():
            if not records:
                continue
            totals  = np.array([r["t_total_ms"]      for r in records])
            execs   = np.array([r["t_exec_local_ms"]  for r in records])
            nets    = np.array([r["t_net_ms"]          for r in records])
            crypts  = np.array([r["e_crypto_ms"]       for r in records])
            lstars  = np.array([r["l_star_ms"]         for r in records])
            summaries.append({
                "scenario":      scenario,
                "complexity":    complexity,
                "n_trials":      len(records),
                "n_success":     sum(1 for r in records if not r.get("error")),
                "t_total_mean":  float(np.mean(totals)),
                "t_total_std":   float(np.std(totals)),
                "t_total_p50":   float(np.percentile(totals, 50)),
                "t_total_p95":   float(np.percentile(totals, 95)),
                "t_total_p99":   float(np.percentile(totals, 99)),
                "t_exec_mean":   float(np.mean(execs)),
                "t_net_mean":    float(np.mean(nets)),
                "e_crypto_mean": float(np.mean(crypts)),
                "l_star_mean":   float(np.mean(lstars)),
            })
    return summaries


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(summaries: List[dict], crossover: dict) -> None:
    # Sort by canonical order, then alphabetically for unknowns
    order = {k: i for i, k in enumerate(SCENARIO_ORDER)}
    summaries_sorted = sorted(
        summaries,
        key=lambda s: (order.get(s["scenario"], 99), s["complexity"])
    )

    print("\n" + "=" * 100)
    print("EdgeTrust-Offload — Experiment Results")
    print("=" * 100)
    print(f"{'Scenario':<32} {'Complexity':<10} {'N':<5} {'Mean(ms)':<10} "
          f"{'Std':<8} {'p50':<8} {'p95':<8} {'p99':<8} {'L*(ms)':<8}")
    print("-" * 100)
    for s in summaries_sorted:
        label = SCENARIO_LABELS.get(s["scenario"], s["scenario"]).replace("\n", " ")
        print(
            f"{label:<32} {s['complexity']:<10} {s['n_trials']:<5} "
            f"{s['t_total_mean']:<10.1f} {s['t_total_std']:<8.1f} "
            f"{s['t_total_p50']:<8.1f} {s['t_total_p95']:<8.1f} "
            f"{s['t_total_p99']:<8.1f} {s['l_star_mean']:<8.1f}"
        )
    print("=" * 100)

    # Crossover summary
    if crossover:
        print("\nCrossover L* (ms) — RTT threshold above which local execution wins:")
        for sc in SCENARIO_ORDER:
            if sc not in crossover:
                continue
            label = SCENARIO_LABELS.get(sc, sc).replace("\n", " ")
            for cx in ("normal", "high"):
                if cx in crossover[sc]:
                    print(f"  {label:<40} [{cx:<6}]  L* = {crossover[sc][cx]:.1f} ms")

    # Zero-Trust overhead: S3 vs S2
    s2 = {s["complexity"]: s for s in summaries if s["scenario"] == "S2"}
    s3 = {s["complexity"]: s for s in summaries if s["scenario"] == "S3"}
    if s2 and s3:
        print("\nZero-Trust encryption overhead (S3 − S2):")
        for cx in ("normal", "high"):
            if cx in s2 and cx in s3:
                delta = s3[cx]["t_total_mean"] - s2[cx]["t_total_mean"]
                est   = s3[cx]["e_crypto_mean"]
                print(f"  [{cx:<6}]  measured Δ = {delta:+.2f} ms   "
                      f"estimated crypto = {est:.2f} ms")

    # Congestion sweep summary
    cong = {s["scenario"]: s for s in summaries
            if s["scenario"] in ("s4_light", "s4_mid", "s4_heavy")
            and s["complexity"] == "normal"}
    s1_normal = next((s for s in summaries
                      if s["scenario"] == "S1" and s["complexity"] == "normal"), None)
    if cong and s1_normal:
        local_mean = s1_normal["t_total_mean"]
        print("\nCongestion sweep vs local baseline (normal complexity):")
        for level in ("s4_light", "s4_mid", "s4_heavy"):
            if level not in cong:
                continue
            mean = cong[level]["t_total_mean"]
            p95  = cong[level]["t_total_p95"]
            wins = "offload wins" if mean < local_mean else "LOCAL WINS  ← crossover"
            label = SCENARIO_LABELS[level].replace("\n", " ")
            print(f"  {label:<40}  mean={mean:.1f}ms  p95={p95:.1f}ms  → {wins}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_crossover(exp_dir: Path, summaries: List[dict]) -> None:
    """Bar chart: mean ± std per scenario, grouped by complexity."""
    if not HAS_MPL:
        return

    # Only include scenarios that are present in summaries
    present = {s["scenario"] for s in summaries}
    scenarios = [sc for sc in SCENARIO_ORDER if sc in present]
    complexities = ["normal", "high"]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(scenarios) * 1.4), 5))
    x = np.arange(len(scenarios))
    width = 0.35

    for ax, metric, title in [
        (axes[0], "t_total_mean", "Mean end-to-end latency"),
        (axes[1], "t_total_p95",  "95th-percentile latency"),
    ]:
        for i, cx in enumerate(complexities):
            vals = []
            stds = []
            for sc in scenarios:
                match = [s for s in summaries
                         if s["scenario"] == sc and s["complexity"] == cx]
                if match:
                    vals.append(match[0][metric])
                    stds.append(match[0]["t_total_std"] if metric == "t_total_mean" else 0)
                else:
                    vals.append(0); stds.append(0)

            offset = (i - 0.5) * width
            colors = [SCENARIO_COLORS.get(sc, "#888888") for sc in scenarios]
            bars = ax.bar(x + offset, vals, width,
                          label=cx.capitalize(),
                          yerr=stds if metric == "t_total_mean" else None,
                          capsize=3, alpha=0.85, color=colors)

        xlabels = [SCENARIO_LABELS.get(sc, sc).replace("\n", "\n") for sc in scenarios]
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Latency (ms)")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = exp_dir / "crossover_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {out}")


def plot_congestion_sweep(exp_dir: Path, summaries: List[dict]) -> None:
    """Line plot: latency across congestion levels + local baseline."""
    if not HAS_MPL:
        return

    cong_scenarios = [sc for sc in ("s4_light", "s4_mid", "s4_heavy")
                      if any(s["scenario"] == sc for s in summaries)]
    if not cong_scenarios:
        return  # no congestion sweep data

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels_short = {"s4_light": "Light\n10ms/3ms/0.5%",
                    "s4_mid":   "Medium\n25ms/8ms/2%",
                    "s4_heavy": "Heavy\n60ms/20ms/5%"}

    for ax, cx in zip(axes, ["normal", "high"]):
        x = np.arange(len(cong_scenarios))

        means = []
        p95s  = []
        stds  = []
        for sc in cong_scenarios:
            match = [s for s in summaries
                     if s["scenario"] == sc and s["complexity"] == cx]
            if match:
                means.append(match[0]["t_total_mean"])
                p95s.append(match[0]["t_total_p95"])
                stds.append(match[0]["t_total_std"])
            else:
                means.append(0); p95s.append(0); stds.append(0)

        ax.bar(x - 0.2, means, 0.35, label="Mean", color="#F44336", alpha=0.8,
               yerr=stds, capsize=4)
        ax.bar(x + 0.2, p95s,  0.35, label="p95",  color="#B71C1C", alpha=0.6)

        # Local baseline reference line
        local = next((s["t_total_mean"] for s in summaries
                      if s["scenario"] == "S1" and s["complexity"] == cx), None)
        if local:
            ax.axhline(local, color="#2196F3", linestyle="--", linewidth=1.5,
                       label=f"Local baseline ({local:.0f} ms)")

        ax.set_xticks(x)
        ax.set_xticklabels([labels_short[sc] for sc in cong_scenarios], fontsize=9)
        ax.set_ylabel("End-to-end latency (ms)")
        ax.set_title(f"Congestion sweep — {cx} complexity")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = exp_dir / "congestion_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {out}")


def plot_latency_cdf(exp_dir: Path, raw: Dict[str, Dict[str, List[dict]]]) -> None:
    """Empirical CDF — normal complexity, all available scenarios."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False

    for sc in SCENARIO_ORDER:
        if sc not in raw or "normal" not in raw[sc]:
            continue
        records = raw[sc]["normal"]
        vals = sorted(r["t_total_ms"] for r in records)
        cdf  = np.arange(1, len(vals) + 1) / len(vals)
        label = SCENARIO_LABELS.get(sc, sc).replace("\n", " ")
        color = SCENARIO_COLORS.get(sc, "#888888")
        lw    = 2.0 if sc in ("S1", "S3") else 1.2
        ls    = "-" if sc.startswith("S") else "--"
        ax.plot(vals, cdf, label=label, color=color, linewidth=lw, linestyle=ls)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF — Normal Complexity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = exp_dir / "latency_cdf.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {out}")


def plot_security_overhead(exp_dir: Path, summaries: List[dict]) -> None:
    """S2 vs S3 encryption overhead bar chart."""
    if not HAS_MPL:
        return

    s2 = {s["complexity"]: s for s in summaries if s["scenario"] == "S2"}
    s3 = {s["complexity"]: s for s in summaries if s["scenario"] == "S3"}
    if not (s2 and s3):
        return

    cxs = [cx for cx in ("normal", "high") if cx in s2 and cx in s3]
    deltas = [s3[cx]["t_total_mean"] - s2[cx]["t_total_mean"] for cx in cxs]
    crypto = [s3[cx]["e_crypto_mean"] for cx in cxs]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(cxs))
    ax.bar(x - 0.2, deltas, 0.35,
           label="Measured overhead (S3 − S2)", color="#FF9800", alpha=0.85)
    ax.bar(x + 0.2, crypto, 0.35,
           label="Estimated crypto (ChaCha20)", color="#9C27B0", alpha=0.85)
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
    p = argparse.ArgumentParser(
        description="Analyze EdgeTrust-Offload experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("exp_dir", type=Path, help="Experiment or merged results directory")
    p.add_argument("--no-plots", action="store_true", help="Print table only, skip plots")
    args = p.parse_args()

    if not args.exp_dir.exists():
        print(f"ERROR: {args.exp_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Load raw records (always available if raw/ dir exists)
    raw = load_all_raw(args.exp_dir)

    # Load or build summary
    summaries = load_summary(args.exp_dir)
    if not summaries:
        if raw:
            print("[INFO] No summary.json found — building from raw JSONL files.")
            summaries = build_summary_from_raw(raw)
            # Persist so subsequent calls are faster
            (args.exp_dir / "summary.json").write_text(
                json.dumps(summaries, indent=2)
            )
        else:
            print("ERROR: No summary.json and no raw/*.jsonl files found.", file=sys.stderr)
            print(f"       Directory: {args.exp_dir}", file=sys.stderr)
            sys.exit(1)

    # Load or build crossover
    crossover = load_crossover(args.exp_dir)
    if not crossover:
        for s in summaries:
            crossover.setdefault(s["scenario"], {})[s["complexity"]] = s["l_star_mean"]
        (args.exp_dir / "crossover.json").write_text(
            json.dumps(crossover, indent=2)
        )

    print_report(summaries, crossover)

    if not args.no_plots:
        plot_crossover(args.exp_dir, summaries)
        plot_congestion_sweep(args.exp_dir, summaries)
        plot_latency_cdf(args.exp_dir, raw)
        plot_security_overhead(args.exp_dir, summaries)
        print(f"\nAll plots saved in: {args.exp_dir}/")


if __name__ == "__main__":
    main()
