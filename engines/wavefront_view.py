#!/usr/bin/env python
"""
wavefront_view.py

Inspect the radial "wavefront" diagnostics from a precipitating_event run.

Looks for:
  - run_root/data/wave_series.npz      (center_site, shells, wave_intensity)
  - run_root/timeseries.npz            (times)

What it does:

  1. Load times, center_site, shells, wave_intensity.
  2. Print a short summary:
       - center site
       - available shells (graph distances)
       - max wave intensity per shell and the time it occurs.
  3. Write a CSV file at run_root/data/wavefront.csv with columns:
       time, wave_shell0, wave_shell1, ...

This gives you a clean "wavefront view": how much <Z> is changing at each
graph radius from the first dominant lump center, as a function of time.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

import numpy as np
import csv


def load_wave_series(run_root: str) -> Dict[str, Any]:
    data_path = os.path.join(run_root, "data", "wave_series.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"wave_series.npz not found at {data_path}. "
            "Make sure precipitating_event.py was run with the newer "
            "--snapshot-stride>0 version that writes wavefront diagnostics."
        )
    arr = np.load(data_path)
    required = ["center_site", "shells", "wave_intensity"]
    for key in required:
        if key not in arr:
            raise KeyError(f"{key} missing from wave_series.npz at {data_path}.")
    center_site = int(arr["center_site"])
    shells = np.array(arr["shells"], dtype=int)
    wave_intensity = np.array(arr["wave_intensity"], dtype=float)
    return {
        "center_site": center_site,
        "shells": shells,
        "wave_intensity": wave_intensity,
    }


def load_times(run_root: str) -> np.ndarray:
    ts_path = os.path.join(run_root, "timeseries.npz")
    if not os.path.exists(ts_path):
        raise FileNotFoundError(
            f"timeseries.npz not found at {ts_path}. "
            "Run precipitating_event.py first."
        )
    arr = np.load(ts_path)
    if "times" not in arr:
        raise KeyError(f"'times' missing from timeseries.npz at {ts_path}.")
    return np.array(arr["times"], dtype=float)


def write_wavefront_csv(
    csv_path: str,
    times: np.ndarray,
    shells: np.ndarray,
    wave_intensity: np.ndarray,
) -> None:
    """
    Write a CSV with columns:
      time, wave_shell{shells[0]}, wave_shell{shells[1]}, ...
    """
    n_steps, n_shells = wave_intensity.shape
    if len(times) != n_steps:
        raise ValueError(
            f"times has length {len(times)}, but wave_intensity has {n_steps} rows."
        )

    header = ["time"] + [f"wave_shell{int(s)}" for s in shells]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t_idx in range(n_steps):
            row = [f"{times[t_idx]:.9g}"]
            row += [f"{wave_intensity[t_idx, j]:.9g}" for j in range(n_shells)]
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="View radial wavefront diagnostics from a precipitating_event run."
    )
    ap.add_argument(
        "--run-root",
        required=True,
        help="Run root for precipitating_event (contains data/wave_series.npz and timeseries.npz).",
    )
    ap.add_argument(
        "--csv-name",
        default="wavefront.csv",
        help="Name of CSV file to write inside run_root/data/.",
    )
    args = ap.parse_args()

    run_root = os.path.abspath(args.run_root)
    print("============================================================")
    print("  Wavefront View (radial |ΔZ| vs time)")
    print("============================================================")
    print(f"run_root:   {run_root}")
    print("------------------------------------------------------------")

    try:
        wave = load_wave_series(run_root)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    try:
        times = load_times(run_root)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    center_site = wave["center_site"]
    shells = wave["shells"]
    wave_intensity = wave["wave_intensity"]

    n_steps, n_shells = wave_intensity.shape

    print(f"center_site: {center_site}")
    print(f"n_steps:     {n_steps}")
    print(f"shells:      {list(int(s) for s in shells)}")
    print("------------------------------------------------------------")

    # Basic per-shell summary
    print("Per-shell max wave intensity:")
    print("shell   max_wave      time_at_max")
    print("----------------------------------")
    for j in range(n_shells):
        shell = int(shells[j])
        shell_vals = wave_intensity[:, j]
        max_idx = int(np.argmax(shell_vals))
        max_val = float(shell_vals[max_idx])
        t_at_max = float(times[max_idx]) if max_idx < len(times) else float("nan")
        print(f"{shell:5d}  {max_val:10.6e}  {t_at_max:11.6f}")
    print("----------------------------------")

    # Write CSV
    data_dir = os.path.join(run_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, args.csv_name)
    write_wavefront_csv(csv_path, times, shells, wave_intensity)
    print(f"Wrote wavefront CSV: {csv_path}")
    print("You can plot this as time vs wave_shell{d} to see the radial waves.")


if __name__ == "__main__":
    main()
