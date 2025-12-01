#!/usr/bin/env python3
"""
sweep_topology_params.py

Parameter sweep over the topology / defrag parameters for the Hilbert substrate.

We vary:
  - pressure_p       (sparsification exponent, 0 < p < 1)
  - pressure_q       (row amplification exponent, q > 1)
  - defrag_cutoff    (hard pruning threshold)

and record, over time:
  - number of edges
  - degree statistics
  - sample graph diameter
  - coupling statistics

All runs are logged into a single CSV:

    topology_param_sweep.csv

Columns:
    run_id,
    pressure_p,
    pressure_q,
    defrag_cutoff,
    seed,
    step,
    n_edges,
    mean_degree,
    min_degree,
    max_degree,
    sample_diameter,
    mean_coupling,
    std_coupling,
    max_coupling

This is intended for "phase diagram" style plots: how diameter, degree, etc.
behave as a function of (p, q, cutoff).
"""

import csv
import itertools
import numpy as np
from substrate import Config, Substrate


# ---------------------------------------------------------------------------
# Utility: graph statistics
# ---------------------------------------------------------------------------

def measure_graph_stats(substrate: Substrate, max_radius: int = 32) -> dict:
    """Compute basic graph statistics and a sampled effective diameter."""
    n = substrate.n_nodes
    neighbors = substrate._neighbors

    # Degree distribution
    degrees = [len(neighbors[i]) for i in range(n)]
    mean_degree = float(np.mean(degrees)) if len(degrees) > 0 else 0.0
    min_degree = int(min(degrees)) if len(degrees) > 0 else 0
    max_degree = int(max(degrees)) if len(degrees) > 0 else 0

    # Sample-based graph diameter (over a subset of nodes)
    sample_size = min(20, n)
    sample_nodes = np.random.choice(n, size=sample_size, replace=False)

    max_dist = 0
    for i in sample_nodes:
        for j in sample_nodes:
            if i < j:
                d = substrate.graph_distance(i, j, max_radius=max_radius)
                if not np.isinf(d):
                    max_dist = max(max_dist, int(d))

    # Count edges
    n_edges = substrate.num_edges

    # Coupling magnitude distribution
    J_xp = substrate.couplings
    if hasattr(J_xp, "get"):
        J_np = J_xp.get()
    else:
        J_np = np.asarray(J_xp)
    mags = np.abs(J_np).flatten()
    mags = mags[mags > 0]

    return {
        "n_edges": int(n_edges),
        "mean_degree": mean_degree,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "sample_diameter": max_dist,
        "mean_coupling": float(np.mean(mags)) if len(mags) > 0 else 0.0,
        "std_coupling": float(np.std(mags)) if len(mags) > 0 else 0.0,
        "max_coupling": float(np.max(mags)) if len(mags) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

def run_param_sweep(
    csv_path: str = "topology_param_sweep.csv",
    n_nodes: int = 128,
    connectivity: float = 0.30,
    defrag_rate: float = 0.2,
    n_steps: int = 2000,
    checkpoint_every: int = 20,
    base_seed: int = 42,
) -> None:
    """
    Run a sweep over (pressure_p, pressure_q, defrag_cutoff) and log
    graph statistics to a single CSV.

    You can tune the parameter lists below as needed.
    """

    # Parameter ranges to explore
    pressure_p_list = [0.3, 0.5, 0.8]     # sparsification strength
    pressure_q_list = [1.5, 2.0]          # row pressure amplification
    cutoff_list     = [0.0, 0.01, 0.02]   # hard pruning

    param_grid = list(itertools.product(pressure_p_list, pressure_q_list, cutoff_list))

    print("=" * 70)
    print("Parameter sweep over substrate topology")
    print(f"  n_nodes          = {n_nodes}")
    print(f"  connectivity     = {connectivity}")
    print(f"  defrag_rate      = {defrag_rate}")
    print(f"  n_steps          = {n_steps}")
    print(f"  checkpoint_every = {checkpoint_every}")
    print(f"  total runs       = {len(param_grid)}")
    print("=" * 70)

    # Open CSV and write header
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_id",
                "pressure_p",
                "pressure_q",
                "defrag_cutoff",
                "seed",
                "step",
                "n_edges",
                "mean_degree",
                "min_degree",
                "max_degree",
                "sample_diameter",
                "mean_coupling",
                "std_coupling",
                "max_coupling",
            ]
        )

        # Loop over parameter combinations
        for run_id, (p, q, cutoff) in enumerate(param_grid):
            seed = base_seed + run_id  # change seed per run for diversity

            print("\n" + "-" * 70)
            print(f"Run {run_id}: p={p}, q={q}, cutoff={cutoff}, seed={seed}")
            print("-" * 70)

            config = Config(
                n_nodes=n_nodes,
                internal_dim=3,
                connectivity=connectivity,
                defrag_rate=defrag_rate,
                pressure_p=p,
                pressure_q=q,
                defrag_cutoff=cutoff,
                seed=seed,
            )

            sub = Substrate(config)

            # Initial stats at step 0
            stats = measure_graph_stats(sub)
            writer.writerow(
                [
                    run_id,
                    p,
                    q,
                    cutoff,
                    seed,
                    0,
                    stats["n_edges"],
                    stats["mean_degree"],
                    stats["min_degree"],
                    stats["max_degree"],
                    stats["sample_diameter"],
                    stats["mean_coupling"],
                    stats["std_coupling"],
                    stats["max_coupling"],
                ]
            )
            print(
                f"  step 0: "
                f"diam={stats['sample_diameter']}, "
                f"edges={stats['n_edges']}, "
                f"mean_deg={stats['mean_degree']:.2f}"
            )

            # Evolve and log at checkpoints
            for step in range(1, n_steps + 1):
                sub.evolve(n_steps=1)

                if step % checkpoint_every == 0:
                    stats = measure_graph_stats(sub)
                    writer.writerow(
                        [
                            run_id,
                            p,
                            q,
                            cutoff,
                            seed,
                            step,
                            stats["n_edges"],
                            stats["mean_degree"],
                            stats["min_degree"],
                            stats["max_degree"],
                            stats["sample_diameter"],
                            stats["mean_coupling"],
                            stats["std_coupling"],
                            stats["max_coupling"],
                        ]
                    )
                    print(
                        f"  step {step}: "
                        f"diam={stats['sample_diameter']}, "
                        f"edges={stats['n_edges']}, "
                        f"mean_deg={stats['mean_degree']:.2f}"
                    )

    print("\nSweep complete.")
    print(f"CSV written to: {csv_path}")


def main() -> None:
    run_param_sweep(
        csv_path="topology_param_sweep.csv",
        n_nodes=128,
        connectivity=0.30,
        defrag_rate=0.2,
        n_steps=2000,
        checkpoint_every=20,
        base_seed=42,
    )


if __name__ == "__main__":
    main()
