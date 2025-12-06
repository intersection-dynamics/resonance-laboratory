#!/usr/bin/env python3
"""
history_wave_diagnostic.py

Compare two substrate runs (A and B) that differ by a small local "history"
(e.g. a tiny local unitary applied in run B), and diagnose how that difference
propagates through the graph.

Inputs (explicit, no guessing):
  --npz-a    path to timeseries.npz for run A
  --npz-b    path to timeseries.npz for run B

Optional:
  --geometry    geometry npz with graph_dist (for radial analysis)
  --source-site integer index of the site where the "history kick" was applied
  --csv         path to CSV file to write (if omitted, no CSV is written)

We assume each timeseries.npz contains:
  - times: 1D array shape (T,)
  - n_t:   2D array shape (T, N) with local excitations n_i(t)
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Tuple, List, Optional

import numpy as np


# ---------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------
def load_timeseries_npz(path: str) -> Dict[str, np.ndarray]:
    """Load times and n_t from a single timeseries.npz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"timeseries.npz not found at {path}")
    arr = np.load(path)
    if "times" not in arr or "n_t" not in arr:
        raise KeyError(f"{path} must contain 'times' and 'n_t'.")
    times = np.array(arr["times"], dtype=float)
    n_t = np.array(arr["n_t"], dtype=float)
    return {"times": times, "n_t": n_t}


def load_graph_dist(geom_path: str) -> np.ndarray:
    """Load graph_dist from a geometry npz."""
    data = np.load(geom_path, allow_pickle=True)
    if "graph_dist" not in data:
        raise KeyError(f"Geometry {geom_path} must contain 'graph_dist'.")
    gd = np.asarray(data["graph_dist"], dtype=float)
    if gd.ndim != 2 or gd.shape[0] != gd.shape[1]:
        raise ValueError("graph_dist must be a square matrix.")
    return gd


def radial_shells(
    graph_dist: np.ndarray, source_site: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Given a graph_dist matrix and a source site, build shells:
      shell d = all sites with int(graph_dist[source, i]) == d

    Returns:
      d_values: 1D array of distinct shell distances (sorted)
      shells: list of index arrays, shells[k] = sites in shell d_values[k]
    """
    N = graph_dist.shape[0]
    if source_site < 0 or source_site >= N:
        raise ValueError(f"source_site {source_site} out of bounds for N={N}")
    d_float = graph_dist[source_site, :]
    d_int = np.round(d_float).astype(int)
    d_values = np.unique(d_int)
    shells: List[np.ndarray] = []
    for d in d_values:
        idx = np.where(d_int == d)[0]
        shells.append(idx)
    return d_values, shells


# ---------------------------------------------------------------------
# Core diagnostics
# ---------------------------------------------------------------------
def compute_delta_metrics(
    nA_t: np.ndarray,
    nB_t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Δn_i(t) and global norms.

    Returns:
      delta_ti      : (T, N) = nB - nA
      abs_delta_ti  : (T, N) = |delta|
      L1_t          : (T,)   = sum_i |Δn_i|
      L2_t          : (T,)   = sqrt(sum_i Δn_i^2)
    """
    if nA_t.shape != nB_t.shape:
        raise ValueError(f"Shape mismatch: nA_t {nA_t.shape}, nB_t {nB_t.shape}")
    delta_ti = nB_t - nA_t
    abs_delta_ti = np.abs(delta_ti)
    L1_t = abs_delta_ti.sum(axis=1)
    L2_t = np.sqrt((delta_ti ** 2).sum(axis=1))
    return delta_ti, abs_delta_ti, L1_t, L2_t


def radial_summary(
    abs_delta_ti: np.ndarray,
    times: np.ndarray,
    graph_dist: np.ndarray,
    source_site: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Using geometry and source site, compute:

      shell_L1_t:  (T, S) = sum_{i in shell s} |Δn_i|
      shell_d:     (S,)   = integer shell distances
      R90_t:       (T,)   = minimal radius R so that cumulative shell_L1 >= 90% total L1
      Rmax90:      float  = max R90_t over all t

    Returns:
      shell_d, shell_L1_t, R90_t, Rmax90
    """
    T, N = abs_delta_ti.shape
    gd = graph_dist
    d_values, shells = radial_shells(gd, source_site)
    S = len(shells)

    shell_L1_t = np.zeros((T, S), dtype=float)
    for s_idx, idx in enumerate(shells):
        if idx.size == 0:
            continue
        shell_L1_t[:, s_idx] = abs_delta_ti[:, idx].sum(axis=1)

    total_L1_t = shell_L1_t.sum(axis=1)
    R90_t = np.full(T, np.nan, dtype=float)

    for t_idx in range(T):
        L_total = total_L1_t[t_idx]
        if L_total <= 0:
            continue
        cumulative = 0.0
        radius90 = d_values[-1]
        for s_idx, d in enumerate(d_values):
            cumulative += shell_L1_t[t_idx, s_idx]
            if cumulative >= 0.9 * L_total:
                radius90 = d
                break
        R90_t[t_idx] = float(radius90)

    if np.all(np.isnan(R90_t)):
        Rmax90 = float("nan")
    else:
        Rmax90 = float(np.nanmax(R90_t))

    return d_values, shell_L1_t, R90_t, Rmax90


def write_csv(
    csv_path: str,
    times: np.ndarray,
    L1_t: np.ndarray,
    L2_t: np.ndarray,
    max_site_t: np.ndarray,
    max_val_t: np.ndarray,
    shell_d: Optional[np.ndarray],
    shell_L1_t: Optional[np.ndarray],
    R90_t: Optional[np.ndarray],
) -> None:
    """
    Write a CSV with:

      time, L1, L2, max_site, max_abs, (optional R90), (optional shell_L1_d0, ...)

    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = ["time", "L1", "L2", "max_site", "max_abs"]
        if R90_t is not None:
            header.append("R90")
        if shell_d is not None and shell_L1_t is not None:
            for d in shell_d:
                header.append(f"shell_L1_d{int(d)}")
        writer.writerow(header)

        T = times.shape[0]
        for t_idx in range(T):
            row = [
                f"{times[t_idx]:.9g}",
                f"{L1_t[t_idx]:.9g}",
                f"{L2_t[t_idx]:.9g}",
                int(max_site_t[t_idx]),
                f"{max_val_t[t_idx]:.9g}",
            ]
            if R90_t is not None:
                row.append(
                    "" if np.isnan(R90_t[t_idx]) else f"{R90_t[t_idx]:.9g}"
                )
            if shell_d is not None and shell_L1_t is not None:
                for s_idx in range(shell_L1_t.shape[1]):
                    row.append(f"{shell_L1_t[t_idx, s_idx]:.9g}")
            writer.writerow(row)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "History wave diagnostic: compare two timeseries.npz files and "
            "see how Δn_i(t) spreads through the graph."
        )
    )
    ap.add_argument(
        "--npz-a",
        required=True,
        help="Path to timeseries.npz for run A (baseline).",
    )
    ap.add_argument(
        "--npz-b",
        required=True,
        help="Path to timeseries.npz for run B (perturbed).",
    )
    ap.add_argument(
        "--geometry",
        help="Optional geometry npz with graph_dist for radial analysis.",
    )
    ap.add_argument(
        "--source-site",
        type=int,
        help="Index of site where the 'history kick' was applied (for shells).",
    )
    ap.add_argument(
        "--csv",
        help="Optional path to write history_wave.csv. If omitted, no CSV is written.",
    )
    args = ap.parse_args()

    npz_a = os.path.abspath(args.npz_a)
    npz_b = os.path.abspath(args.npz_b)

    print("============================================================")
    print("  History Wave Diagnostic (Δn_i(t) between two runs)")
    print("============================================================")
    print(f"npz A: {npz_a}")
    print(f"npz B: {npz_b}")
    print("------------------------------------------------------------")

    tsA = load_timeseries_npz(npz_a)
    tsB = load_timeseries_npz(npz_b)

    timesA, nA_t = tsA["times"], tsA["n_t"]
    timesB, nB_t = tsB["times"], tsB["n_t"]

    if timesA.shape != timesB.shape or not np.allclose(timesA, timesB):
        raise ValueError("Runs A and B must have matching time grids.")

    times = timesA
    T, N = nA_t.shape
    print(f"time steps: {T}")
    print(f"sites:      {N}")
    print("------------------------------------------------------------")

    delta_ti, abs_delta_ti, L1_t, L2_t = compute_delta_metrics(nA_t, nB_t)

    total_L1 = L1_t
    total_L2 = L2_t
    max_L1_idx = int(np.argmax(total_L1))
    max_L2_idx = int(np.argmax(total_L2))

    max_site_t = np.argmax(abs_delta_ti, axis=1)
    max_val_t = np.max(abs_delta_ti, axis=1)

    print("Global Δn norms over time:")
    print(f"  L1: min={total_L1.min():.6g}, max={total_L1.max():.6g}")
    print(f"  L2: min={total_L2.min():.6g}, max={total_L2.max():.6g}")
    print(f"  max L1 at t={times[max_L1_idx]:.6g}")
    print(f"  max L2 at t={times[max_L2_idx]:.6g}")
    print("------------------------------------------------------------")

    shell_d = None
    shell_L1_t = None
    R90_t = None
    Rmax90 = None

    if args.geometry is not None and args.source_site is not None:
        geom_path = os.path.abspath(args.geometry)
        gd = load_graph_dist(geom_path)
        if gd.shape[0] != N:
            raise ValueError(
                f"graph_dist size {gd.shape[0]} != N_sites {N} for the runs."
            )
        shell_d, shell_L1_t, R90_t, Rmax90 = radial_summary(
            abs_delta_ti, times, gd, args.source_site
        )

        print("Radial shell analysis (L1 per distance shell):")
        print("  shells (graph distance):", [int(d) for d in shell_d])
        print("  R90_t (radius enclosing 90% of |Δn|) summary:")
        if np.all(np.isnan(R90_t)):
            print("    R90_t is NaN for all times (no signal).")
        else:
            print(f"    min R90 = {np.nanmin(R90_t):.3g}")
            print(f"    max R90 = {Rmax90:.3g}")
        print("------------------------------------------------------------")

        print("Per-shell max L1(|Δn|):")
        print("shell  max_L1      time_at_max")
        print("--------------------------------")
        for s_idx, d in enumerate(shell_d):
            shell_vals = shell_L1_t[:, s_idx]
            t_idx = int(np.argmax(shell_vals))
            print(
                f"{int(d):5d}  {shell_vals[t_idx]:10.6g}  {times[t_idx]:10.6g}"
            )
        print("--------------------------------")

    if args.csv:
        csv_path = os.path.abspath(args.csv)
        write_csv(
            csv_path,
            times,
            total_L1,
            total_L2,
            max_site_t,
            max_val_t,
            shell_d,
            shell_L1_t,
            R90_t,
        )
        print(f"Wrote CSV: {csv_path}")

    print("Done.")


if __name__ == "__main__":
    main()
