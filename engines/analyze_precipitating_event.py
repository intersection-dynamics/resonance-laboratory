"""
analyze_precipitating_event.py

Quick diagnostic for precipitating_event.py runs.

- Loads a specific run's time_series.npz.
- Computes the range of |Z_i(t) - mean(Z(t))| over all times/sites.
- Suggests a reasonable z-threshold.
- Optionally prints a few time slices so you can see if there's
  any emergent structure in <Z_i(t)>.

This does NOT modify the main simulation. Pure post-processing.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze a Precipitating Event run (time_series.npz)."
    )
    p.add_argument(
        "--run-root",
        type=str,
        required=True,
        help="Path to a specific precipitating_event run root.",
    )
    p.add_argument(
        "--print-slices",
        type=int,
        default=5,
        help="How many time slices to print (spread across the run).",
    )
    return p.parse_args()


def load_time_series(run_root: str) -> Dict[str, Any]:
    data_path = os.path.join(run_root, "data", "time_series.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"time_series.npz not found at {data_path}")
    npz = np.load(data_path)
    times = npz["times"]
    local_z_t = npz["local_z_t"]
    lump_counts = npz["lump_counts"]
    return {"times": times, "local_z_t": local_z_t, "lump_counts": lump_counts}


def load_config(run_root: str) -> Dict[str, Any]:
    params_path = os.path.join(run_root, "params.json")
    if not os.path.exists(params_path):
        return {}
    with open(params_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Precipitating Event Run Analyzer")
    print("=" * 60)
    print(f"Run root: {args.run_root}")
    print("-" * 60)

    series = load_time_series(args.run_root)
    times = series["times"]
    local_z_t = series["local_z_t"]
    lump_counts = series["lump_counts"]

    n_steps, n_sites = local_z_t.shape
    print(f"n_steps: {n_steps}, n_sites: {n_sites}")

    # Compute |Z_i(t) - mean_Z(t)| distribution
    z_profiles = local_z_t
    mean_Z_t = np.mean(z_profiles, axis=1, keepdims=True)
    dev = np.abs(z_profiles - mean_Z_t)

    dev_min = float(np.min(dev))
    dev_max = float(np.max(dev))
    dev_mean = float(np.mean(dev))
    dev_median = float(np.median(dev))

    print("-" * 60)
    print("Deviation |Z_i(t) - mean_Z(t)| across all times/sites:")
    print(f"  min:    {dev_min:.6f}")
    print(f"  max:    {dev_max:.6f}")
    print(f"  mean:   {dev_mean:.6f}")
    print(f"  median: {dev_median:.6f}")

    # Suggest a threshold as some fraction of the max deviation
    suggested_thr_lo = 0.3 * dev_max
    suggested_thr_hi = 0.6 * dev_max
    print("-" * 60)
    print("Suggested z-threshold range based on this run:")
    print(f"  low:    {suggested_thr_lo:.6f}")
    print(f"  high:   {suggested_thr_hi:.6f}")
    print("You can pass this as --z-threshold to precipitating_event.py")
    print("-" * 60)

    # Lump count quick stats
    print("Lump count stats (from the run):")
    print(f"  min:    {int(np.min(lump_counts))}")
    print(f"  max:    {int(np.max(lump_counts))}")
    print(f"  mean:   {float(np.mean(lump_counts)):.3f}")
    print("-" * 60)

    # Print a few slices of Z(t)
    n_to_print = max(1, min(args.print_slices, n_steps))
    idxs = np.linspace(0, n_steps - 1, n_to_print, dtype=int)
    print(f"Printing {n_to_print} time slices of Z_i(t):")
    for idx in idxs:
        t = float(times[idx])
        z_prof = local_z_t[idx]
        z_str = ", ".join(f"{z:.3f}" for z in z_prof)
        print(f"  t={t:7.3f}  Z: [{z_str}]")

    print("=" * 60)


if __name__ == "__main__":
    main()
