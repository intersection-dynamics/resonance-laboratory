#!/usr/bin/env python3
"""
cosmic_inflation_experiment.py

Probe "inflation" in the Hilbert substrate framework.

This script:
  - builds a dense, highly connected substrate (early "over-entangled" state)
  - runs evolution with sparsifying defrag + cutoff
  - measures how an effective "space size" grows over time

Metrics recorded to CSV:
  t_step                 : integer time step
  t_physical             : t_step * dt
  avg_degree             : mean number of neighbors per node
  degree_50              : median degree
  degree_90              : 90th percentile degree
  graph_radius_50        : median graph distance from center node
  graph_radius_90        : 90th percentile graph distance from center node
  graph_diameter_est     : max finite BFS distance from center
  reachable_fraction     : fraction of nodes reachable from center
  eff_dist_mean          : mean 1 / |J_ij| over all nonzero couplings
  eff_dist_90            : 90th percentile of 1 / |J_ij|
  a_graph_90             : graph_radius_90 / graph_radius_90(t=0)
  a_eff_90               : eff_dist_90     / eff_dist_90(t=0)

Interpretation:
  - If defrag truly "inflates space", you should see:
      * avg_degree dropping (graph gets sparser)
      * graph_radius_90 and/or eff_dist_90 growing
      * a_* scale factors increasing by large factors.
  - If graph_radius_90 suddenly drops while reachable_fraction also drops,
    it means the center node got stranded in a tiny component, not that
    the whole universe collapsed.
"""

import argparse
import csv
from typing import List, Tuple

import numpy as np

from substrate import Config, Substrate


# ------------------------------------------------------------------------
# Graph / distance diagnostics
# ------------------------------------------------------------------------

def graph_distance_profile(sub: Substrate, center: int = 0) -> Tuple[float, float, float, float]:
    """
    Compute graph distance statistics from a chosen center node.

    Uses BFS on sub._neighbors (topological graph, ignores |J|).
    IMPORTANT: no artificial max-radius cap; we explore the full component.

    Returns
    -------
    r50 : float
        Median finite distance from center.
    r90 : float
        90th percentile finite distance from center.
    diam_est : float
        Maximum finite distance from center (an estimate of graph diameter).
    reachable_fraction : float
        Fraction of nodes reachable (finite distance) from center.
    """
    from collections import deque

    n = sub.n_nodes
    neighbors = sub._neighbors  # list-of-lists of ints

    center = int(center)
    if center < 0 or center >= n:
        raise ValueError(f"Center index {center} out of range 0..{n-1}")

    # BFS distances
    dist = np.full(n, np.inf, dtype=float)
    dist[center] = 0.0

    q = deque([center])
    while q:
        i = q.popleft()
        di = dist[i]
        for j in neighbors[i]:
            if not np.isfinite(dist[j]):
                dist[j] = di + 1.0
                q.append(j)

    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        # Center is completely isolated
        return float("inf"), float("inf"), float("inf"), 0.0

    r50 = float(np.quantile(finite, 0.5))
    r90 = float(np.quantile(finite, 0.9))
    diam_est = float(np.max(finite))
    reachable_fraction = float(finite.size) / float(n)
    return r50, r90, diam_est, reachable_fraction


def effective_distance_metrics(sub: Substrate) -> Tuple[float, float]:
    """
    Coupling-based "distance" metrics derived from |J_ij|.

    For all nonzero couplings J_ij, define a local "length" L_ij = 1 / |J_ij|.
    Then:
      eff_mean = mean(L_ij)
      eff_90   = 90th percentile of L_ij

    Intuition:
      - If couplings are strong and dense, L_ij is small (tight, opaque).
      - If couplings dilute and weaken, L_ij grows (looser, more "space").
    """
    J = sub.couplings
    try:
        # CuPy backend
        J_np = J.get()
    except AttributeError:
        # NumPy backend
        J_np = np.asarray(J)

    mag = np.abs(J_np)
    mask = mag > 0.0
    if not np.any(mask):
        return float("nan"), float("nan")

    lengths = 1.0 / mag[mask]
    eff_mean = float(np.mean(lengths))
    eff_90 = float(np.quantile(lengths, 0.9))
    return eff_mean, eff_90


def degree_stats(sub: Substrate) -> Tuple[float, float, float]:
    """
    Compute degree-based statistics.

    Returns
    -------
    avg_degree, degree_50, degree_90
    """
    degrees = np.array([len(nbrs) for nbrs in sub._neighbors], dtype=float)
    if degrees.size == 0:
        return 0.0, 0.0, 0.0

    avg_degree = float(np.mean(degrees))
    degree_50 = float(np.quantile(degrees, 0.5))
    degree_90 = float(np.quantile(degrees, 0.9))
    return avg_degree, degree_50, degree_90


# ------------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------------

def run_inflation_experiment(
    n_nodes: int,
    internal_dim: int,
    steps: int,
    record_stride: int,
    out_csv: str,
    seed: int,
    connectivity: float,
    monogamy_budget: float,
    defrag_rate: float,
    dt: float,
    pressure_p: float,
    pressure_q: float,
    defrag_cutoff: float,
    center_node: int = 0,
) -> None:
    """
    Build a dense substrate, run evolution with sparsifying defrag, and record
    inflation-related metrics to CSV.
    """
    # --------------------------------------------------------------------
    # Build config and substrate
    # --------------------------------------------------------------------
    cfg = Config(
        n_nodes=n_nodes,
        internal_dim=internal_dim,
        monogamy_budget=monogamy_budget,
        defrag_rate=defrag_rate,
        dt=dt,
        seed=seed,
        connectivity=connectivity,
        pressure_p=pressure_p,
        pressure_q=pressure_q,
        defrag_cutoff=defrag_cutoff,
    )

    print("[inflation] Building substrate...")
    sub = Substrate(cfg)

    # --------------------------------------------------------------------
    # Time evolution + measurement
    # --------------------------------------------------------------------
    t_steps: List[int] = []
    t_phys: List[float] = []
    avg_degrees: List[float] = []
    degree_50s: List[float] = []
    degree_90s: List[float] = []
    r50s: List[float] = []
    r90s: List[float] = []
    diam_ests: List[float] = []
    reach_fracs: List[float] = []
    eff_means: List[float] = []
    eff_90s: List[float] = []

    print("[inflation] Starting evolution...")
    for step in range(steps + 1):
        # Measure at stride steps (including t=0)
        if step % record_stride == 0:
            t = step * dt

            avg_deg, deg50, deg90 = degree_stats(sub)
            r50, r90, diam, reach_frac = graph_distance_profile(sub, center=center_node)
            eff_mean, eff_90 = effective_distance_metrics(sub)

            t_steps.append(step)
            t_phys.append(t)
            avg_degrees.append(avg_deg)
            degree_50s.append(deg50)
            degree_90s.append(deg90)
            r50s.append(r50)
            r90s.append(r90)
            diam_ests.append(diam)
            reach_fracs.append(reach_frac)
            eff_means.append(eff_mean)
            eff_90s.append(eff_90)

            print(
                f"[t={t:8.3f}] step={step:6d} "
                f"deg(avg,90)={avg_deg:6.2f},{deg90:6.2f} "
                f"r90={r90:5.2f} diam~{diam:5.2f} "
                f"reach={reach_frac:5.2f} "
                f"eff_90={eff_90:8.3f}"
            )

        # Stop before extra evolve at final step
        if step >= steps:
            break

        # One evolution step (unitary + defrag)
        sub.evolve(n_steps=1)

    # --------------------------------------------------------------------
    # Compute scale factors
    # --------------------------------------------------------------------
    if len(r90s) > 0 and np.isfinite(r90s[0]) and r90s[0] != 0.0:
        a_graph_90 = [r / r90s[0] if np.isfinite(r) else float("nan") for r in r90s]
    else:
        a_graph_90 = [float("nan")] * len(r90s)

    if len(eff_90s) > 0 and not np.isnan(eff_90s[0]) and eff_90s[0] != 0.0:
        a_eff_90 = [e / eff_90s[0] if not np.isnan(e) else float("nan") for e in eff_90s]
    else:
        a_eff_90 = [float("nan")] * len(eff_90s)

    # --------------------------------------------------------------------
    # Write CSV
    # --------------------------------------------------------------------
    print(f"[inflation] Writing results to {out_csv} ...")
    fieldnames = [
        "t_step",
        "t_physical",
        "avg_degree",
        "degree_50",
        "degree_90",
        "graph_radius_50",
        "graph_radius_90",
        "graph_diameter_est",
        "reachable_fraction",
        "eff_dist_mean",
        "eff_dist_90",
        "a_graph_90",
        "a_eff_90",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i in range(len(t_steps)):
            writer.writerow([
                t_steps[i],
                t_phys[i],
                avg_degrees[i],
                degree_50s[i],
                degree_90s[i],
                r50s[i],
                r90s[i],
                diam_ests[i],
                reach_fracs[i],
                eff_means[i],
                eff_90s[i],
                a_graph_90[i],
                a_eff_90[i],
            ])

    # --------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------
    print("[inflation] Done.")
    if len(r90s) > 1 and len(eff_90s) > 1:
        print(
            f"[inflation] graph_radius_90: {r90s[0]:.3f} -> {r90s[-1]:.3f} "
            f"(a_graph_90 ~ {a_graph_90[-1]:.3f})"
        )
        print(
            f"[inflation] eff_dist_90   : {eff_90s[0]:.3f} -> {eff_90s[-1]:.3f} "
            f"(a_eff_90   ~ {a_eff_90[-1]:.3f})"
        )


# ------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cosmic inflation experiment in Hilbert substrate.")
    p.add_argument("--n_nodes", type=int, default=128, help="Number of nodes in the substrate.")
    p.add_argument("--internal_dim", type=int, default=3, help="Local Hilbert space dimension.")
    p.add_argument("--steps", type=int, default=5000, help="Total evolution steps.")
    p.add_argument("--record_stride", type=int, default=10, help="Record metrics every N steps.")
    p.add_argument("--out_csv", type=str, default="inflation_history.csv", help="Output CSV path.")
    p.add_argument("--seed", type=int, default=2025, help="Random seed.")
    p.add_argument("--connectivity", type=float, default=1.0, help="Initial edge probability (dense ~1.0).")
    p.add_argument("--monogamy_budget", type=float, default=1.0, help="Per-node L1 budget for couplings.")
    p.add_argument("--defrag_rate", type=float, default=0.5, help="Defrag sharpening aggressiveness.")
    p.add_argument("--dt", type=float, default=0.1, help="Physical time step for unitary evolution.")
    p.add_argument("--pressure_p", type=float, default=0.5, help="(Unused) pressure exponent p (0 < p < 1).")
    p.add_argument("--pressure_q", type=float, default=2.0, help="(Unused) pressure exponent q (> 1).")
    p.add_argument("--defrag_cutoff", type=float, default=1e-2, help="Cutoff for |J_ij| below which edges are removed.")
    p.add_argument("--center_node", type=int, default=0, help="Center node index for radius measurements.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inflation_experiment(
        n_nodes=args.n_nodes,
        internal_dim=args.internal_dim,
        steps=args.steps,
        record_stride=args.record_stride,
        out_csv=args.out_csv,
        seed=args.seed,
        connectivity=args.connectivity,
        monogamy_budget=args.monogamy_budget,
        defrag_rate=args.defrag_rate,
        dt=args.dt,
        pressure_p=args.pressure_p,
        pressure_q=args.pressure_q,
        defrag_cutoff=args.defrag_cutoff,
        center_node=args.center_node,
    )
