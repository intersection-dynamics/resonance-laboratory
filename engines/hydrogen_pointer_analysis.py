#!/usr/bin/env python3
"""
===============================================================================
Hydrogen Pointer Analysis
===============================================================================

Analysis helper for the emergent_hydrogen_pointer_substrate runs.

Expected directory layout:

    <run_dir>/
        params.json
        summary.json
        data/
            timeseries.npz
        logs/
            run.log      (optional)
        figures/
            (this script will create PNGs here)

It will:

  - Print a quick summary based on summary.json and timeseries.npz.
  - Generate:
        figures/radii_vs_time.png
        figures/decoherence_vs_time.png
        figures/radial_profiles.png
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_timeseries(run_dir: str) -> Dict[str, Any]:
    data_path = os.path.join(run_dir, "data", "timeseries.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find timeseries.npz at {data_path}")
    data = np.load(data_path)
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_radii_vs_time(run_dir: str, ts: Dict[str, Any]) -> None:
    times = ts["times"]
    e_r = ts["electron_radius"]
    ph_r = ts["photon_radius"]
    pr_r = ts["proton_radius"]

    plt.figure(figsize=(8, 5))
    plt.plot(times, e_r, label="electron ⟨r⟩")
    plt.plot(times, ph_r, label="photon ⟨r⟩")
    plt.plot(times, pr_r, label="proton ⟨r⟩")

    plt.xlabel("time")
    plt.ylabel("mean radius ⟨r⟩")
    plt.title("Pointer mean radii vs time")
    plt.legend()
    plt.tight_layout()

    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, "radii_vs_time.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_decoherence_vs_time(run_dir: str, ts: Dict[str, Any]) -> None:
    times = ts["times"]
    C = ts["decoherence_C"]

    plt.figure(figsize=(8, 5))
    # Use semilogy if values span many orders of magnitude
    if np.any(C > 0):
        plt.semilogy(times, C, label="pointer coherence C")
    else:
        plt.plot(times, C, label="pointer coherence C")

    plt.xlabel("time")
    plt.ylabel("C (average off-diagonal magnitude)")
    plt.title("Decoherence in pointer basis vs time")
    plt.legend()
    plt.tight_layout()

    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, "decoherence_vs_time.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_radial_profiles(run_dir: str, ts: Dict[str, Any]) -> None:
    r = ts["r_centers"]
    e_prof = ts["electron_radial_mean"]
    ph_prof = ts["photon_radial_mean"]
    pr_prof = ts["proton_radial_mean"]

    plt.figure(figsize=(8, 5))
    plt.plot(r, pr_prof, label="proton", linewidth=2)
    plt.plot(r, e_prof, label="electron", linewidth=2)
    plt.plot(r, ph_prof, label="photon", linewidth=2)

    plt.xlabel("radius r")
    plt.ylabel("average pointer density")
    plt.title("Time-averaged radial pointer profiles")
    plt.legend()
    plt.tight_layout()

    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, "radial_profiles.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main analysis routine
# ---------------------------------------------------------------------------


def analyze_run(run_dir: str) -> None:
    params_path = os.path.join(run_dir, "params.json")
    summary_path = os.path.join(run_dir, "summary.json")

    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.json not found in {run_dir}")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary.json not found in {run_dir}")

    params = load_json(params_path)
    summary = load_json(summary_path)
    ts = load_timeseries(run_dir)

    script = summary.get("script", "unknown_script")
    run_id = summary.get("run_id", "unknown_run")
    metrics = summary.get("metrics", {})
    diagnostics = summary.get("diagnostics", {})

    print("================================================================")
    print("Hydrogen Pointer Analysis")
    print("================================================================")
    print(f"Run dir : {run_dir}")
    print(f"Script  : {script}")
    print(f"Run ID  : {run_id}")
    print()
    print("Parameters (subset):")
    print(f"  Lx, Ly, Lz     : {params.get('Lx')} {params.get('Ly')} {params.get('Lz')}")
    print(f"  d_pointer, d_env: {params.get('d_pointer')} {params.get('d_env')}")
    print(f"  dt, steps      : {params.get('dt')} {params.get('total_steps')}")
    print()

    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k:30s} : {v}")
    print()
    print("Diagnostics:")
    print(f"  converged       : {diagnostics.get('converged')}")
    print(f"  runtime_seconds : {diagnostics.get('runtime_seconds')}")
    if diagnostics.get("warnings"):
        print("  warnings:")
        for w in diagnostics["warnings"]:
            print(f"    - {w}")
    print("================================================================")
    print()

    # Generate plots
    print("Generating plots...")
    plot_radii_vs_time(run_dir, ts)
    plot_decoherence_vs_time(run_dir, ts)
    plot_radial_profiles(run_dir, ts)
    print("Plots saved under figures/ in this run directory.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a hydrogen pointer run from emergent_hydrogen_pointer_substrate."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the run directory (the one containing params.json/summary.json/data/).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    analyze_run(os.path.abspath(args.run_dir))


if __name__ == "__main__":
    main()
