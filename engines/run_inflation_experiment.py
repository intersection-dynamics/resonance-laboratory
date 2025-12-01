#!/usr/bin/env python3
"""
run_inflation_experiment.py

Targeted experiment to look for "actual inflation" in the substrate:
large growth in graph diameter under entanglement-redistribution dynamics.

This:

  - Uses a more aggressive parameter set (strong sparsification + cutoff)
  - Runs longer
  - Uses a better diameter estimator:
      from several random sources, BFS to ALL nodes, take max distance

Outputs a CSV:

    inflation_history.csv

with columns:
    step,
    n_edges,
    mean_degree,
    min_degree,
    max_degree,
    est_diameter,
    reachable_fraction,
    mean_coupling,
    std_coupling,
    max_coupling
"""

import csv
import numpy as np
from substrate import Config, Substrate


def estimate_diameter_fullish(
    sub: Substrate,
    n_sources: int = 16,
    max_radius: int = 512,
) -> tuple[int, float]:
    """
    Estimate graph diameter more seriously than the quick sample method.

    - Pick up to n_sources random source nodes.
    - For each source, BFS to ALL nodes.
    - Track the maximum finite distance seen.
    - Also track fraction of node pairs that are mutually reachable.

    Returns
    -------
    est_diameter : int
        Maximum finite distance encountered.
    reachable_fraction : float
        Fraction of (source,target) pairs that were reachable within max_radius.
    """
    from collections import deque

    n = sub.n_nodes
    neighbors = sub._neighbors

    # Choose distinct sources
    n_sources = min(n_sources, n)
    sources = np.random.choice(n, size=n_sources, replace=False)

    max_dist = 0
    reachable_pairs = 0
    total_pairs = n_sources * n

    for s in sources:
        # BFS from s
        dist = {s: 0}
        q = deque([s])

        while q:
            i = q.popleft()
            d_i = dist[i]
            if d_i >= max_radius:
                continue
            for j in neighbors[i]:
                if j not in dist:
                    dist[j] = d_i + 1
                    q.append(j)

        # Update statistics
        for t in range(n):
            if t in dist:
                reachable_pairs += 1
                if dist[t] > max_dist:
                    max_dist = dist[t]

    reachable_fraction = reachable_pairs / float(total_pairs) if total_pairs > 0 else 0.0
    return int(max_dist), float(reachable_fraction)


def measure_graph_stats_inflation(sub: Substrate) -> dict:
    """
    Compute graph stats + better diameter estimate for the inflation run.
    """
    n = sub.n_nodes
    neighbors = sub._neighbors

    degrees = [len(neighbors[i]) for i in range(n)]
    mean_degree = float(np.mean(degrees)) if degrees else 0.0
    min_degree = int(min(degrees)) if degrees else 0
    max_degree = int(max(degrees)) if degrees else 0

    est_diam, reachable_frac = estimate_diameter_fullish(sub, n_sources=16, max_radius=512)

    # Coupling magnitude distribution
    J_xp = sub.couplings
    if hasattr(J_xp, "get"):
        J_np = J_xp.get()
    else:
        J_np = np.asarray(J_xp)
    mags = np.abs(J_np).flatten()
    mags = mags[mags > 0]

    mean_c = float(np.mean(mags)) if len(mags) > 0 else 0.0
    std_c = float(np.std(mags)) if len(mags) > 0 else 0.0
    max_c = float(np.max(mags)) if len(mags) > 0 else 0.0

    return {
        "n_edges": sub.num_edges,
        "mean_degree": mean_degree,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "est_diameter": est_diam,
        "reachable_fraction": reachable_frac,
        "mean_coupling": mean_c,
        "std_coupling": std_c,
        "max_coupling": max_c,
    }


def run_inflation_experiment(
    csv_path: str = "inflation_history.csv",
    n_nodes: int = 256,
    connectivity: float = 0.25,
    defrag_rate: float = 0.25,
    pressure_p: float = 0.25,
    pressure_q: float = 2.0,
    defrag_cutoff: float = 0.03,
    n_steps: int = 4000,
    checkpoint_every: int = 20,
    seed: int = 42,
) -> None:
    """
    Run a single, aggressive "inflation" experiment and log stats to CSV.
    """

    print("=" * 70)
    print("Inflation experiment")
    print("- Target: large growth in graph diameter")
    print(f"  n_nodes      = {n_nodes}")
    print(f"  connectivity = {connectivity}")
    print(f"  defrag_rate  = {defrag_rate}")
    print(f"  pressure_p   = {pressure_p}")
    print(f"  pressure_q   = {pressure_q}")
    print(f"  cutoff       = {defrag_cutoff}")
    print(f"  n_steps      = {n_steps}")
    print("=" * 70)

    config = Config(
        n_nodes=n_nodes,
        internal_dim=3,
        connectivity=connectivity,
        defrag_rate=defrag_rate,
        pressure_p=pressure_p,
        pressure_q=pressure_q,
        defrag_cutoff=defrag_cutoff,
        seed=seed,
    )

    sub = Substrate(config)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "n_edges",
                "mean_degree",
                "min_degree",
                "max_degree",
                "est_diameter",
                "reachable_fraction",
                "mean_coupling",
                "std_coupling",
                "max_coupling",
            ]
        )

        # Initial
        stats = measure_graph_stats_inflation(sub)
        writer.writerow(
            [
                0,
                stats["n_edges"],
                stats["mean_degree"],
                stats["min_degree"],
                stats["max_degree"],
                stats["est_diameter"],
                stats["reachable_fraction"],
                stats["mean_coupling"],
                stats["std_coupling"],
                stats["max_coupling"],
            ]
        )
        print(
            f"step 0: diam={stats['est_diameter']}, "
            f"reachable={stats['reachable_fraction']:.3f}, "
            f"edges={stats['n_edges']}, "
            f"mean_deg={stats['mean_degree']:.2f}"
        )

        # Evolution
        for step in range(1, n_steps + 1):
            sub.evolve(n_steps=1)

            if step % checkpoint_every == 0:
                stats = measure_graph_stats_inflation(sub)
                writer.writerow(
                    [
                        step,
                        stats["n_edges"],
                        stats["mean_degree"],
                        stats["min_degree"],
                        stats["max_degree"],
                        stats["est_diameter"],
                        stats["reachable_fraction"],
                        stats["mean_coupling"],
                        stats["std_coupling"],
                        stats["max_coupling"],
                    ]
                )
                print(
                    f"step {step}: "
                    f"diam={stats['est_diameter']}, "
                    f"reachable={stats['reachable_fraction']:.3f}, "
                    f"edges={stats['n_edges']}, "
                    f"mean_deg={stats['mean_degree']:.2f}"
                )

    print(f"\nInflation run complete. CSV written to: {csv_path}")


def main() -> None:
    run_inflation_experiment()


if __name__ == "__main__":
    main()
