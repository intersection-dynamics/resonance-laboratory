#!/usr/bin/env python3
"""
emergent_particles.py
=====================

Toy "particle" extractor on top of the precipitating_event quench+cooling lab.

Philosophy
----------

  - There are no particles; only a Hilbert-space wavefunction psi over nodes.
  - Geometry is emergent from the Lieb-Robinson (LR) metric embedding (3D).
  - The precipitating event produces topological vortices on cycles of the LR graph.
  - We interpret *clusters of nonzero-vorticity cycles* as proto-"particles".

Pipeline
--------

1. Use precipitating_event.run_simulation(..., return_data=True) to:
     - build geometry from asset_dir (lr_metrics + embedding),
     - run the quench + cooling schedule,
     - return times, psi_t, Q_net, etc.

2. Independently rebuild:
     - adjacency from graph_dist == 1,
     - a list of short cycles (length 3..max_cycle_len) on the graph.

3. At a chosen snapshot time t_snap:
     - Take psi_snap = psi_t[idx_snap],
     - Extract phases theta_i = arg(psi_i),
     - Compute integer vortex charge q for each cycle:
          q = round(sum_edges delta_theta / (2*pi)).

4. Select all cycles with q != 0.

5. Merge overlapping nonzero-charged cycles into *clusters*:
     - Treat each cycle's node set as a small patch.
     - Union-find or iterative merging: clusters whose node sets intersect are merged.
     - Each cluster C has:
          nodes(C), cycles(C), total_charge(C) = sum_over_cycles q,
          center(C) = avg_{i in nodes(C)} X_i,
          radius(C) = rms distance from center,
          weight(C) = sum_{i in nodes(C)} |psi_i|^2    (probability mass).

6. Print a human-readable summary and save:
     emergent_particles.npz with:
        snapshot_time, idx_snap,
        cluster_charge, cluster_weight, cluster_radius,
        cluster_centers (M,3), cluster_node_indices (object array of int lists),
        config used (from precipitating_event) and seeds.

This is *not* adding extra ontology; it is just reading off persistent,
topologically-nontrivial eddies in the frozen Hilbert sea.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Set, Tuple, Optional, Any

import numpy as np

# We reuse the quench lab's configuration and geometry utilities.
from precipitating_event import (
    PrecipConfig,
    run_simulation,
    load_metrics,
    load_embedding,
    build_adjacency_from_graph_dist,
    find_cycles,
    phase_field,
    cycle_vorticity,
)


# ---------------------------------------------------------------------------
# Cluster construction from nonzero-vorticity cycles
# ---------------------------------------------------------------------------


def build_clusters_from_cycles(
    cycles: List[List[int]],
    q: np.ndarray,
    X: np.ndarray,
    psi: np.ndarray,
    min_abs_q: int = 1,
) -> Tuple[
    List[Set[int]],  # cluster_nodes
    List[List[int]],  # cluster_cycle_indices
    np.ndarray,  # cluster_charge
    np.ndarray,  # cluster_weight
    np.ndarray,  # cluster_radius
    np.ndarray,  # cluster_centers
]:
    """
    Given:
      - cycles: list of cycles (each is a list of node indices),
      - q: integer vortex charges for each cycle,
      - X: coordinates for each node (N,3),
      - psi: wavefunction at snapshot (length N),

    Construct "particle" clusters by merging cycles with |q| >= min_abs_q
    that share at least one node.

    Returns:
      cluster_nodes         : list of sets of node indices
      cluster_cycle_indices : list of lists of cycle indices for each cluster
      cluster_charge        : array of net integer charge per cluster
      cluster_weight        : array of probability mass per cluster
      cluster_radius        : array of RMS radius in configuration space
      cluster_centers       : array (M,3) of cluster centers
    """
    N = X.shape[0]
    if psi.shape[0] != N:
        raise ValueError("psi length and X shape mismatch.")

    # Select nonzero-vorticity cycles with |q| >= min_abs_q
    active_indices = [i for i, qi in enumerate(q) if abs(qi) >= min_abs_q]
    if not active_indices:
        return [], [], np.zeros(0, dtype=int), np.zeros(0), np.zeros(0), np.zeros((0, 3))

    # Each active cycle defines an initial cluster
    clusters_nodes: List[Set[int]] = []
    clusters_cycle_indices: List[List[int]] = []
    for idx in active_indices:
        cyc = cycles[idx]
        clusters_nodes.append(set(cyc))
        clusters_cycle_indices.append([idx])

    # Iteratively merge clusters that share nodes
    merged = True
    while merged:
        merged = False
        new_nodes: List[Set[int]] = []
        new_cycles: List[List[int]] = []
        used = [False] * len(clusters_nodes)

        for i in range(len(clusters_nodes)):
            if used[i]:
                continue
            base_nodes = set(clusters_nodes[i])
            base_cycles = list(clusters_cycle_indices[i])
            used[i] = True

            # Try to merge with later clusters that intersect base_nodes
            changed = True
            while changed:
                changed = False
                for j in range(i + 1, len(clusters_nodes)):
                    if used[j]:
                        continue
                    if base_nodes.intersection(clusters_nodes[j]):
                        base_nodes.update(clusters_nodes[j])
                        base_cycles.extend(clusters_cycle_indices[j])
                        used[j] = True
                        changed = True
                        merged = True

            new_nodes.append(base_nodes)
            new_cycles.append(base_cycles)

        clusters_nodes = new_nodes
        clusters_cycle_indices = new_cycles

    # Compute cluster properties
    M = len(clusters_nodes)
    cluster_charge = np.zeros(M, dtype=int)
    cluster_weight = np.zeros(M, dtype=float)
    cluster_radius = np.zeros(M, dtype=float)
    cluster_centers = np.zeros((M, 3), dtype=float)

    prob = np.abs(psi) ** 2

    for k in range(M):
        nodes_k = sorted(clusters_nodes[k])
        cycles_k = clusters_cycle_indices[k]

        # Net charge = sum of q over all cycles in cluster
        q_sum = 0
        for idx in cycles_k:
            q_sum += int(q[idx])
        cluster_charge[k] = q_sum

        # Weight = sum of probability mass on cluster nodes
        w = float(np.sum(prob[nodes_k]))
        cluster_weight[k] = w

        # Center = average position of nodes (simple mean)
        coords = X[nodes_k]  # (n_k,3)
        center = np.mean(coords, axis=0)
        cluster_centers[k] = center

        # Radius = RMS distance from center
        diffs = coords - center[None, :]
        r2 = np.mean(np.sum(diffs**2, axis=1))
        cluster_radius[k] = float(np.sqrt(r2))

    return (
        clusters_nodes,
        clusters_cycle_indices,
        cluster_charge,
        cluster_weight,
        cluster_radius,
        cluster_centers,
    )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_emergent_particles(args: argparse.Namespace) -> None:
    """
    Run a precipitating_event simulation, then extract "particles" from
    a chosen snapshot by clustering nonzero-vorticity cycles.
    """
    # Build precipitating-event config
    cfg = PrecipConfig(
        asset_dir=args.asset_dir,
        metrics_file=args.metrics_file,
        embedding_smooth_file=args.embedding_smooth_file,
        embedding_file=args.embedding_file,
        hot_strength=args.hot_strength,
        J_order=args.J_order,
        rot_strength=args.rot_strength,
        Omega_x=args.Omega_x,
        Omega_y=args.Omega_y,
        Omega_z=args.Omega_z,
        t_max=args.t_max,
        n_steps=args.n_steps,
        quench_fraction=args.quench_fraction,
        seed=args.seed,
        init_random_phase=not args.no_random_phase,
        max_cycle_len=args.max_cycle_len,
        output_dir=args.output_dir,
    )

    print("=== Emergent Particles Analysis ===")
    print("Calling precipitating_event.run_simulation with config:")
    print(cfg)
    print()

    result = run_simulation(cfg, return_data=True)
    if result is None:
        raise RuntimeError("run_simulation returned None despite return_data=True")

    times = result["times"]      # (T,)
    psi_t = result["psi_t"]      # (T,N)
    X = result["X"]              # (N,3)
    Q_net = result["Q_net"]      # (T,)
    config_dict = result["config"]

    T, N = psi_t.shape
    print(f"Simulation returned T={T} snapshots, N={N} nodes.")
    print(f"Q_net[0] = {Q_net[0]}, Q_net[-1] = {Q_net[-1]}")
    print()

    # Decide snapshot index
    if args.snapshot_time < 0.0:
        idx_snap = T - 1
        t_snap = float(times[idx_snap])
        print(f"Using final snapshot at t={t_snap:.3f} (index {idx_snap}).")
    else:
        t_target = args.snapshot_time
        idx_snap = int(np.argmin(np.abs(times - t_target)))
        t_snap = float(times[idx_snap])
        print(
            f"Requested snapshot t≈{t_target:.3f}; "
            f"using closest t={t_snap:.3f} (index {idx_snap})."
        )

    psi_snap = psi_t[idx_snap]

    # Rebuild adjacency and cycles on the LR graph
    print("\nRebuilding adjacency and cycles for topology analysis...")
    _, _, graph_dist = load_metrics(cfg.asset_dir, cfg.metrics_file)
    adjacency = build_adjacency_from_graph_dist(graph_dist)

    # Just to sanity check
    print("Adjacency (first few nodes):")
    max_print = min(N, 8)
    for i in range(max_print):
        print(f"  {i}: {adjacency[i]}")
    if N > max_print:
        print(f"  ... ({N - max_print} more nodes)")
    print()

    print(f"Finding cycles up to length {cfg.max_cycle_len}...")
    cycles = find_cycles(adjacency, cfg.max_cycle_len)
    print(f"  Found {len(cycles)} cycles.")
    print()

    # Compute vorticity on cycles at snapshot
    theta = phase_field(psi_snap)
    q = cycle_vorticity(theta, cycles)

    n_nonzero = int(np.sum(np.abs(q) >= args.min_abs_q))
    print(
        f"Nonzero-vorticity cycles with |q| >= {args.min_abs_q}: "
        f"{n_nonzero} / {len(cycles)}"
    )

    (
        clusters_nodes,
        clusters_cycle_indices,
        cluster_charge,
        cluster_weight,
        cluster_radius,
        cluster_centers,
    ) = build_clusters_from_cycles(
        cycles,
        q,
        X,
        psi_snap,
        min_abs_q=args.min_abs_q,
    )

    M = len(clusters_nodes)
    print(f"\nIdentified {M} 'particle' clusters.")

    if M == 0:
        print("No clusters found with nonzero total vorticity at this snapshot.")
    else:
        print("\nClusters (sorted by descending |charge|, then weight):")
        idx_sorted = list(range(M))
        idx_sorted.sort(
            key=lambda k: (abs(cluster_charge[k]), cluster_weight[k]), reverse=True
        )

        header = (
            "  idx | Q_cluster | weight   | radius   | center_x  center_y  center_z | n_nodes | n_cycles"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for rank, k in enumerate(idx_sorted):
            Qc = cluster_charge[k]
            w = cluster_weight[k]
            r = cluster_radius[k]
            cx, cy, cz = cluster_centers[k]
            n_nodes = len(clusters_nodes[k])
            n_cycles = len(clusters_cycle_indices[k])
            print(
                f"  {k:3d} | {Qc:8d} | {w:7.4f} | {r:7.4f} | "
                f"{cx:8.4f} {cy:8.4f} {cz:8.4f} | "
                f"{n_nodes:7d} | {n_cycles:8d}"
            )

    # Prepare output directory inside asset_dir/output_dir
    out_dir = os.path.join(cfg.asset_dir, cfg.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Save data to npz
    npz_path = os.path.join(out_dir, "emergent_particles.npz")
    # cluster_node_indices as object array of lists
    cluster_node_indices = np.empty(M, dtype=object)
    cluster_cycle_ids = np.empty(M, dtype=object)
    for k in range(M):
        cluster_node_indices[k] = np.array(sorted(clusters_nodes[k]), dtype=int)
        cluster_cycle_ids[k] = np.array(sorted(clusters_cycle_indices[k]), dtype=int)

    np.savez(
        npz_path,
        snapshot_time=t_snap,
        snapshot_index=idx_snap,
        times=times,
        Q_net=Q_net,
        cluster_charge=cluster_charge,
        cluster_weight=cluster_weight,
        cluster_radius=cluster_radius,
        cluster_centers=cluster_centers,
        cluster_node_indices=cluster_node_indices,
        cluster_cycle_indices=cluster_cycle_ids,
        config=config_dict,
        seed=cfg.seed,
    )

    # Save a human-readable summary
    summary_path = os.path.join(out_dir, "emergent_particles_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Emergent Particles Summary\n")
        f.write("==========================\n\n")
        f.write(f"asset_dir      : {cfg.asset_dir}\n")
        f.write(f"metrics_file   : {cfg.metrics_file}\n")
        f.write(f"embedding_file : {cfg.embedding_file}\n")
        f.write(f"embedding_smooth_file : {cfg.embedding_smooth_file}\n")
        f.write(f"seed           : {cfg.seed}\n")
        f.write(f"t_max, n_steps : {cfg.t_max}, {cfg.n_steps}\n")
        f.write(f"quench_fraction: {cfg.quench_fraction}\n")
        f.write(f"Omega          : ({cfg.Omega_x}, {cfg.Omega_y}, {cfg.Omega_z})\n")
        f.write(f"max_cycle_len  : {cfg.max_cycle_len}\n")
        f.write(f"min_abs_q      : {args.min_abs_q}\n")
        f.write(f"output_dir     : {cfg.output_dir}\n\n")

        f.write(f"Snapshot index : {idx_snap}\n")
        f.write(f"Snapshot time  : {t_snap:.6f}\n")
        f.write(f"Q_net(0)       : {int(Q_net[0])}\n")
        f.write(f"Q_net(final)   : {int(Q_net[-1])}\n\n")

        f.write(f"Total cycles          : {len(cycles)}\n")
        f.write(
            f"Nonzero-vorticity cycles (|q| >= {args.min_abs_q}): "
            f"{int(np.sum(np.abs(q) >= args.min_abs_q))}\n\n"
        )

        f.write(f"Identified clusters   : {M}\n\n")
        if M > 0:
            f.write(
                "idx | Q_cluster | weight   | radius   | center_x  center_y  center_z | n_nodes | n_cycles\n"
            )
            f.write(
                "----+----------+----------+----------+--------------------------------+---------+---------\n"
            )
            idx_sorted = list(range(M))
            idx_sorted.sort(
                key=lambda k: (abs(cluster_charge[k]), cluster_weight[k]), reverse=True
            )
            for k in idx_sorted:
                Qc = cluster_charge[k]
                w = cluster_weight[k]
                r = cluster_radius[k]
                cx, cy, cz = cluster_centers[k]
                n_nodes = len(clusters_nodes[k])
                n_cycles = len(clusters_cycle_indices[k])
                f.write(
                    f"{k:3d} | {Qc:8d} | {w:7.4f} | {r:7.4f} | "
                    f"{cx:8.4f} {cy:8.4f} {cz:8.4f} | "
                    f"{n_nodes:7d} | {n_cycles:8d}\n"
                )

    print(f"\nSaved emergent_particles.npz to {npz_path}")
    print(f"Saved emergent_particles_summary.txt to {summary_path}")
    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract emergent 'particle' clusters from a precipitating_event "
            "quench+cooling run, using topological vortices on the LR graph."
        )
    )

    # Geometry / asset inputs
    parser.add_argument(
        "--asset-dir",
        type=str,
        default="cubic_universe_L3",
        help="Directory with lr_metrics.npz and LR embedding files.",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="lr_metrics.npz",
        help="Metrics file name inside asset-dir.",
    )
    parser.add_argument(
        "--embedding-smooth-file",
        type=str,
        default="lr_embedding_3d_smooth.npz",
        help="Preferred smoothed embedding file name.",
    )
    parser.add_argument(
        "--embedding-file",
        type=str,
        default="lr_embedding_3d.npz",
        help="Fallback embedding file name.",
    )

    # Hamiltonian parameters (mirroring precipitating_event)
    parser.add_argument(
        "--hot-strength",
        type=float,
        default=1.0,
        help="Strength for disordered H_hot.",
    )
    parser.add_argument(
        "--J-order",
        type=float,
        default=1.0,
        help="Coupling for ordered H_order.",
    )
    parser.add_argument(
        "--rot-strength",
        type=float,
        default=0.3,
        help="Overall scale for rotation bias H_rot.",
    )
    parser.add_argument(
        "--Omega-x",
        type=float,
        default=0.0,
        help="Rotation vector Omega_x.",
    )
    parser.add_argument(
        "--Omega-y",
        type=float,
        default=0.0,
        help="Rotation vector Omega_y.",
    )
    parser.add_argument(
        "--Omega-z",
        type=float,
        default=1.0,
        help="Rotation vector Omega_z (use 1.0 for strongly spinning run).",
    )

    # Quench schedule
    parser.add_argument(
        "--t-max",
        type=float,
        default=10.0,
        help="Total evolution time.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=4000,
        help="Number of RK4 time steps.",
    )
    parser.add_argument(
        "--quench-fraction",
        type=float,
        default=0.4,
        help="Fraction of t_max for lambda(t) to ramp from 0 to 1.",
    )

    # Initial state / seed
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for H_hot and initial state.",
    )
    parser.add_argument(
        "--no-random-phase",
        action="store_true",
        help="If set, start from localized |0> instead of random phases.",
    )

    # Topology detection
    parser.add_argument(
        "--max-cycle-len",
        type=int,
        default=4,
        help="Maximum cycle length to consider for vorticity (min 3).",
    )
    parser.add_argument(
        "--min-abs-q",
        type=int,
        default=1,
        help="Minimum absolute vortex charge |q| to include a cycle in clustering.",
    )

    # Snapshot selection
    parser.add_argument(
        "--snapshot-time",
        type=float,
        default=-1.0,
        help=(
            "Snapshot time to analyze; if negative, use final snapshot "
            "(t_max)."
        ),
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="emergent_particles_L3_Omega1_seed",
        help=(
            "Output subdirectory inside asset-dir. "
            "If left as default, you may want to make it seed-specific."
        ),
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # If the user leaves the default output-dir, auto-append the seed
    if args.output_dir == "emergent_particles_L3_Omega1_seed":
        args.output_dir = f"emergent_particles_L3_Omega1_seed{args.seed}"

    run_emergent_particles(args)


if __name__ == "__main__":
    main()
