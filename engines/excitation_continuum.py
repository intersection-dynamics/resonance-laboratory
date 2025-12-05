#!/usr/bin/env python
"""
excitation_continuum.py

Continuous, threshold-free readout of excitations above a data-driven 'vacuum'
baseline for a precipitating_event run.

Baseline (per site):
  n_i(t) = (1 - Z_i(t))/2
  n_i^vac = average of n_i(t) over pre-quench times (t <= t_quench_effective)

At each time t:
  excess_i(t) = max(0, n_i(t) - n_i^vac)
  N(t)        = sum_i n_i(t)                  (total excitation)
  E(t)        = sum_i excess_i(t)             (total above-vacuum excitation)
  pct_excess  = 100 * E(t) / max(N(t), eps)
  frac_sites  = (# of i with excess_i(t) > 0) / n_sites

Outputs:
  run_root/data/excitation_continuum.csv

This is compatible with existing precipitating_event outputs (no engine edits).
"""

from __future__ import annotations

import argparse
import json
import os
import csv
from typing import Any, Dict, Tuple

import numpy as np


def load_timeseries(run_root: str) -> Dict[str, np.ndarray]:
    # Prefer data/timeseries.npz; fall back to run_root/timeseries.npz
    p1 = os.path.join(run_root, "data", "timeseries.npz")
    p2 = os.path.join(run_root, "timeseries.npz")
    path = p1 if os.path.exists(p1) else p2
    if not os.path.exists(path):
        raise FileNotFoundError(f"timeseries.npz not found at {p1} or {p2}")
    arr = np.load(path)
    if "times" not in arr or "local_z_t" not in arr:
        raise KeyError("timeseries.npz must contain 'times' and 'local_z_t'")
    return {"times": np.array(arr["times"], dtype=float),
            "local_z_t": np.array(arr["local_z_t"], dtype=float)}


def load_quench_time(run_root: str, times: np.ndarray) -> float:
    summ = os.path.join(run_root, "summary.json")
    if os.path.exists(summ):
        try:
            with open(summ, "r", encoding="utf-8") as f:
                js = json.load(f)
            return float(js["metrics"]["t_quench_effective"])
        except Exception:
            pass
    # Fallback: use the first time as “pre-quench” up to 20% of the run
    # (keeps this script robust even if summary.json is missing)
    k = max(1, int(0.2 * len(times)))
    return float(times[min(k, len(times)-1)])


def compute_excess(times: np.ndarray, local_z_t: np.ndarray, t_quench: float
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      total_N(t), total_E(t), pct_excess(t), frac_sites_excess(t), excess_i(t) matrix
    """
    # Excitation per site
    n_t = 0.5 * (1.0 - local_z_t)  # shape (T, N)
    T, N = n_t.shape

    # Pre-quench mask
    pre_mask = times <= (t_quench + 1e-12)
    if not np.any(pre_mask):
        # if quench is at first sample, take first 10% as baseline
        k = max(1, int(0.1 * T))
        pre_mask = np.zeros(T, dtype=bool)
        pre_mask[:k] = True

    # Baseline per site
    n_vac = n_t[pre_mask].mean(axis=0)  # shape (N,)

    # Excess above baseline
    excess_ti = n_t - n_vac[None, :]
    excess_ti[excess_ti < 0.0] = 0.0

    total_N = n_t.sum(axis=1)                 # (T,)
    total_E = excess_ti.sum(axis=1)           # (T,)
    eps = 1e-12
    pct_excess = 100.0 * total_E / np.maximum(total_N, eps)

    frac_sites_excess = (excess_ti > 0.0).sum(axis=1) / float(N)

    return total_N, total_E, pct_excess, frac_sites_excess, excess_ti


def write_csv(out_path: str,
              times: np.ndarray,
              total_N: np.ndarray,
              total_E: np.ndarray,
              pct_excess: np.ndarray,
              frac_sites_excess: np.ndarray,
              excess_ti: np.ndarray) -> None:
    T, N = excess_ti.shape
    header = ["time", "total_excitation", "total_excess", "percent_above_vacuum", "frac_sites_excess"]
    header += [f"excess_site{i}" for i in range(N)]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for t_idx in range(T):
            row = [
                f"{times[t_idx]:.9g}",
                f"{total_N[t_idx]:.9g}",
                f"{total_E[t_idx]:.9g}",
                f"{pct_excess[t_idx]:.9g}",
                f"{frac_sites_excess[t_idx]:.9g}",
            ]
            row += [f"{excess_ti[t_idx, j]:.9g}" for j in range(N)]
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Threshold-free excitation report (percent above vacuum baseline).")
    ap.add_argument("--run-root", required=True,
                    help="Run folder from precipitating_event (contains summary.json and timeseries.npz).")
    ap.add_argument("--csv-name", default="excitation_continuum.csv",
                    help="Filename to write under run_root/data/.")
    args = ap.parse_args()

    run_root = os.path.abspath(args.run_root)
    ts = load_timeseries(run_root)
    times = ts["times"]
    local_z_t = ts["local_z_t"]

    t_quench = load_quench_time(run_root, times)

    total_N, total_E, pct_excess, frac_sites_excess, excess_ti = compute_excess(
        times, local_z_t, t_quench
    )

    out_csv = os.path.join(run_root, "data", args.csv_name)
    write_csv(out_csv, times, total_N, total_E, pct_excess, frac_sites_excess, excess_ti)

    # Console summary
    t_idx_peak = int(np.argmax(pct_excess))
    print("============================================================")
    print("  Excitation Continuum (percent above vacuum)")
    print("============================================================")
    print(f"run_root:        {run_root}")
    print(f"t_quench_eff:    {t_quench:.6f}")
    print("------------------------------------------------------------")
    print(f"peak pct_excess: {pct_excess[t_idx_peak]:.6f}% at t={times[t_idx_peak]:.6f}")
    print(f"final pct_excess:{pct_excess[-1]:.6f}%")
    print(f"CSV written to:  {out_csv}")


if __name__ == "__main__":
    main()
