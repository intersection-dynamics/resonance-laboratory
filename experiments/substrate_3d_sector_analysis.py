"""
experiments/substrate_3d_sector_analysis.py

Sector-occupation analysis for the full-Hilbert-space 3D substrate engine.

Default workflow (from repo root):
    python experiments\\substrate_3d_fullspace_exp.py ...
    python experiments\\substrate_3d_sector_analysis.py

By default, this script reads:
    outputs/fullspace_results_latest.json

You can override with:
    --input path\to\some_other_results.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Core analysis logic (same as before)
# ---------------------------------------------------------------------------

def analyze_sector_occupations(
    times: np.ndarray,
    Pk: np.ndarray,
    min_avg_occupancy: float = 0.05,
    min_peak_occupancy: float = 0.10,
) -> Dict[str, Any]:
    if times.ndim != 1:
        raise ValueError(f"times must be 1D, got shape {times.shape}")
    if Pk.ndim != 2:
        raise ValueError(f"Pk must be 2D, got shape {Pk.shape}")
    if Pk.shape[0] != times.shape[0]:
        raise ValueError(
            f"Pk.shape[0] ({Pk.shape[0]}) != times.shape[0] ({times.shape[0]})"
        )

    n_steps, N_sectors = Pk.shape

    avg_Pk = Pk.mean(axis=0)
    max_Pk = Pk.max(axis=0)

    significant_sectors: List[int] = [
        int(k) for k in range(N_sectors) if avg_Pk[k] >= min_avg_occupancy
    ]

    if N_sectors > 1:
        nonzero_indices = np.arange(1, N_sectors)
        k_dom = int(nonzero_indices[np.argmax(avg_Pk[1:])])
    else:
        k_dom = 0

    peak_info: Dict[str, Any] = {}
    for k in range(N_sectors):
        k_max = float(max_Pk[k])
        if k_max < min_peak_occupancy:
            continue
        if k_max <= 0.0:
            continue
        rel_tol = 0.01
        abs_eps = 1e-6
        mask = (Pk[:, k] >= (1.0 - rel_tol) * k_max - abs_eps)
        peak_indices = np.where(mask)[0]
        peak_times = times[peak_indices].tolist()
        peak_info[str(k)] = {
            "max_Pk": k_max,
            "peak_indices": peak_indices.tolist(),
            "peak_times": peak_times,
        }

    diagnostics: Dict[str, Any] = {
        "n_steps": int(n_steps),
        "N_sectors": int(N_sectors),
        "min_avg_occupancy": float(min_avg_occupancy),
        "min_peak_occupancy": float(min_peak_occupancy),
    }

    analysis: Dict[str, Any] = {
        "avg_Pk": avg_Pk.tolist(),
        "max_Pk": max_Pk.tolist(),
        "significant_sectors": significant_sectors,
        "dominant_sector": k_dom,
        "peak_info": peak_info,
        "diagnostics": diagnostics,
    }

    return analysis


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_results_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "times" not in data or "sector_occupations" not in data:
        raise KeyError(
            f"JSON at {path} must contain 'times' and 'sector_occupations' keys."
        )
    return data


def _save_analysis_json(path: str, analysis: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, sort_keys=True)


def _print_summary(analysis: Dict[str, Any]) -> None:
    avg_Pk = np.array(analysis["avg_Pk"], dtype=float)
    max_Pk = np.array(analysis["max_Pk"], dtype=float)
    sig = analysis["significant_sectors"]
    k_dom = analysis["dominant_sector"]
    peak_info = analysis["peak_info"]
    diag = analysis["diagnostics"]

    N_sectors = diag["N_sectors"]

    print("=" * 72)
    print(" Sector Occupation Analysis")
    print("=" * 72)
    print(f"N_sectors (0..N_sites): {N_sectors}")
    print(f"Significant sectors (avg >= {diag['min_avg_occupancy']}): {sig}")
    print(f"Dominant sector (nonzero k with maximal avg_Pk): k = {k_dom}")
    print()

    print("Average occupation per sector (k, avg_Pk, max_Pk):")
    for k in range(N_sectors):
        print(f"  k={k:2d}  avg={avg_Pk[k]:.6f}  max={max_Pk[k]:.6f}")
    print()

    print("Peak times for well-populated sectors:")
    if not peak_info:
        print("  (No sector exceeded min_peak_occupancy threshold.)")
    else:
        for k_str, info in peak_info.items():
            k = int(k_str)
            max_k = info["max_Pk"]
            times = info["peak_times"]
            print(f"  k={k:2d}  max={max_k:.6f}  peak_times={times}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze sector occupations from full-space 3D substrate engine.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help=(
            "Path to JSON file with 'times' and 'sector_occupations'. "
            "If omitted, defaults to outputs/fullspace_results_latest.json."
        ),
    )
    parser.add_argument(
        "--min-avg-occupancy",
        type=float,
        default=0.05,
        help="Min time-averaged occupation to call a sector 'significant'.",
    )
    parser.add_argument(
        "--min-peak-occupancy",
        type=float,
        default=0.10,
        help="Min max occupation to report detailed peak times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input:
        input_path = args.input
    else:
        input_path = os.path.join("outputs", "fullspace_results_latest.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input JSON not found at '{input_path}'. "
            "Run substrate_3d_fullspace_exp.py first."
        )

    data = _load_results_json(input_path)
    times_np = np.array(data["times"], dtype=np.float64)
    Pk_np = np.array(data["sector_occupations"], dtype=np.float64)

    analysis = analyze_sector_occupations(
        times=times_np,
        Pk=Pk_np,
        min_avg_occupancy=args.min_avg_occupancy,
        min_peak_occupancy=args.min_peak_occupancy,
    )

    _print_summary(analysis)

    base, ext = os.path.splitext(input_path)
    out_path = base + ".analysis.json"
    _save_analysis_json(out_path, analysis)
    print(f"\nAnalysis written to: {out_path}")


if __name__ == "__main__":
    main()
