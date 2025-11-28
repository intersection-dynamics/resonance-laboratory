#!/usr/bin/env python3
"""
basin_scan.py
=============
Scan the "basin of stability" for the Unified Substrate Framework.

We vary:
  - number of modes (n_modes)
  - hopping coupling strength
  - max evolution time (t_max)

For each parameter combo we:
  - build a Substrate and local-chain Hamiltonian
  - run test_propagation()
  - measure how well the single-excitation norm is preserved over time:
        norm(t) ≈ sum_m history[m, t]
  - record the maximum deviation from 1.0 and mark stable/unstable.

Outputs:
  - A JSON file (basin_results.json) with all scan records
  - A simple text summary on stdout

This is meant as a coarse map of where the dynamics behave nicely
(numerically and physically) vs where they get sketchy.
"""

import argparse
import json
import os
from typing import List, Dict, Any

import numpy as np

import substrate  # assumes substrate.py is in the same directory
from substrate import Substrate, linear_chain, hopping_hamiltonian, test_propagation


def parse_csv_floats(s: str) -> List[float]:
    """Parse comma-separated floats."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_ints(s: str) -> List[int]:
    """Parse comma-separated ints."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def scan_basin(
    modes_list: List[int],
    coupling_list: List[float],
    tmax_list: List[float],
    n_steps: int,
    threshold: float,
) -> Dict[str, Any]:
    """
    Perform the basin-of-stability scan.

    Returns a dict with:
      - 'params': the scan ranges
      - 'records': list of per-point results
    """
    records: List[Dict[str, Any]] = []

    for n_modes in modes_list:
        sub = Substrate(n_modes, dim_per_mode=2)

        for coupling in coupling_list:
            connectivity = linear_chain(n_modes)
            H = hopping_hamiltonian(sub, connectivity, coupling=coupling)

            for t_max in tmax_list:
                # Use the central mode as source
                source = n_modes // 2

                result = test_propagation(
                    sub, H, source=source, t_max=t_max, n_steps=n_steps
                )
                history = result["history"]  # shape (n_modes, n_steps), numpy array
                times = result["times"]

                # For a single excitation, sum over modes should be ~1 at all times
                total = history.sum(axis=0)
                norm_deviation = float(np.max(np.abs(total - 1.0)))

                # Basic sanity: numbers finite and norm reasonably preserved
                stable = bool(
                    np.all(np.isfinite(history)) and np.isfinite(total).all()
                    and norm_deviation <= threshold
                )

                records.append(
                    {
                        "n_modes": n_modes,
                        "coupling": coupling,
                        "t_max": t_max,
                        "n_steps": n_steps,
                        "threshold": threshold,
                        "max_norm_deviation": norm_deviation,
                        "stable": stable,
                        "arrival": {
                            str(k): float(v) for k, v in result["arrival"].items()
                        },
                        "times": times.tolist(),
                        "total_norm_over_time": total.tolist(),
                    }
                )

    return {
        "params": {
            "modes_list": modes_list,
            "coupling_list": coupling_list,
            "tmax_list": tmax_list,
            "n_steps": n_steps,
            "threshold": threshold,
        },
        "records": records,
    }


def summarize(records: List[Dict[str, Any]]) -> None:
    """Print a compact textual summary to stdout."""
    print("\nBASIN OF STABILITY SUMMARY")
    print("=" * 32)
    header = f"{'modes':>5} {'J':>6} {'t_max':>6} {'max|Δnorm|':>12} {'stable':>8}"
    print(header)
    print("-" * len(header))

    for rec in records:
        print(
            f"{rec['n_modes']:5d} "
            f"{rec['coupling']:6.3f} "
            f"{rec['t_max']:6.2f} "
            f"{rec['max_norm_deviation']:12.3e} "
            f"{str(rec['stable']):>8}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Scan basin of stability for the substrate hopping dynamics."
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="4,6,8",
        help="Comma-separated list of mode counts to scan (default: 4,6,8)",
    )
    parser.add_argument(
        "--couplings",
        type=str,
        default="0.25,0.5,1.0,2.0",
        help="Comma-separated list of hopping couplings J (default: 0.25,0.5,1.0,2.0)",
    )
    parser.add_argument(
        "--tmax",
        type=str,
        default="4.0,8.0,12.0",
        help="Comma-separated list of max times t_max (default: 4.0,8.0,12.0)",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=120,
        help="Number of time steps in propagation (default: 120)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Max allowed deviation from norm=1.0 to count as stable (default: 1e-3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="basin_results.json",
        help="Output JSON file (default: basin_results.json)",
    )

    args = parser.parse_args()

    modes_list = parse_csv_ints(args.modes)
    coupling_list = parse_csv_floats(args.couplings)
    tmax_list = parse_csv_floats(args.tmax)

    scan = scan_basin(
        modes_list=modes_list,
        coupling_list=coupling_list,
        tmax_list=tmax_list,
        n_steps=args.nsteps,
        threshold=args.threshold,
    )

    summarize(scan["records"])

    out_path = os.path.abspath(args.output)
    with open(out_path, "w") as f:
        json.dump(scan, f, indent=2)

    print(f"\nWrote basin scan results to: {out_path}")


if __name__ == "__main__":
    main()
