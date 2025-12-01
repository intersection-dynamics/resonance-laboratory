#!/usr/bin/env python3
"""
cosmic_birth_experiment.py

"T = 0 infinite density" -> space birth -> particle emergence (toy experiment)

This driver is meant to probe the *earliest* part of your story:

    - At T = 0, the substrate is effectively "metricless":
        information density is maximized, couplings are dense,
        and there's no meaningful notion of distance yet.

    - Under strictly unitary evolution + defrag (no new physics terms),
      the couplings reorganize to respect no-signaling, which we interpret
      as an *effective emergence / inflation of distance*.

    - Later, proton-like and electron-like patterns appear and can bind.

We do NOT change substrate.py, pattern_detector.py, or the rules.
We only:

  - override the initial couplings to be dense and "overcrowded",
  - define a *coupling-based effective distance metric*,
  - track that metric over time,
  - and track proton/electron candidates and their separation.

Outputs
-------
- Prints a running log of effective space metrics and particle scores.
- Writes a per-frame CSV file: cosmic_birth_history.csv

Columns include (per time sample):

    time
    center_node
    eff_dist_mean
    eff_dist_90
    graph_radius_90
    proton_id
    proton_score
    electron_id
    electron_score
    photon_id
    photon_score
    graph_dist_pe

You can then look for:

  - eff_dist_mean / eff_dist_90 growing with time   -> "space pushing out"
  - proton/electron scores stabilizing and d_pe shrinking
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from substrate import Config, Substrate
import pattern_detector as pd
from scan_defrag_photon_packets import graph_distance


# =============================================================================
# Config for this driver
# =============================================================================

@dataclass
class CosmicBirthConfig:
    # Substrate / graph parameters
    n_nodes: int = 128
    internal_dim: int = 3
    monogamy_budget: float = 1.0
    defrag_rate: float = 0.1
    dt: float = 0.1
    seed: int = 2025

    # We still pass a connectivity to Config, but we *override* couplings
    # to be dense (all-to-all) after construction. So connectivity is mostly
    # a placeholder here.
    connectivity: float = 1.0

    # Evolution controls
    total_steps: int = 5000
    record_stride: int = 10

    # Effective graph radius for simple BFS-based summary
    max_graph_radius: int = 8

    # Output
    output_csv: str = "cosmic_birth_history.csv"

    # Dense initial couplings
    dense_init: bool = True  # if True, overwrite substrate.couplings to dense random
    dense_init_scale: float = 1.0  # overall scale before monogamy normalization


# =============================================================================
# Helpers
# =============================================================================

def _to_numpy(x) -> np.ndarray:
    """Convert xp array (numpy or cupy) to NumPy ndarray."""
    if hasattr(x, "get"):  # CuPy ndarray
        return x.get()
    return np.asarray(x)


def make_dense_initial_couplings(substrate: Substrate, scale: float = 1.0) -> None:
    """
    Overwrite the substrate couplings with an "infinite density" / dense state.

    - All off-diagonal entries are non-zero random complex values.
    - Matrix is symmetrized.
    - Each row is scaled so that sum |J_ij| ~ monogamy_budget.

    This gives a highly connected, metricless-ish starting point where
    every node "talks" strongly to every other node.
    """
    xp = substrate.xp
    n = substrate.n_nodes

    rng = np.random.default_rng(substrate.config.seed)

    # Random dense complex matrix
    J_real = rng.normal(loc=0.0, scale=scale, size=(n, n))
    J_imag = rng.normal(loc=0.0, scale=scale, size=(n, n))
    J_np = J_real + 1j * J_imag

    # Remove self-couplings
    np.fill_diagonal(J_np, 0.0)

    # Symmetrize
    J_np = 0.5 * (J_np + J_np.T.conj())

    # Normalize rows to match monogamy_budget
    row_scale = np.sum(np.abs(J_np), axis=1, keepdims=True)
    row_scale[row_scale == 0] = 1.0
    target = substrate.monogamy_budget
    J_np *= (target / row_scale)

    # Assign back to substrate (xp array)
    substrate.couplings = xp.asarray(J_np, dtype=xp.complex128)

    # Recompute neighbor lists based on non-zero couplings
    J_abs = np.abs(J_np)
    neighbors: List[List[int]] = []
    for i in range(n):
        nbrs = np.nonzero(J_abs[i] > 0)[0].tolist()
        neighbors.append(nbrs)

    substrate._neighbors = neighbors  # type: ignore[attr-defined]

    print(
        f"[cosmic_birth] Dense initial couplings set: "
        f"{n} nodes, fully connected."
    )


def coupling_distance_metrics(substrate: Substrate) -> Tuple[int, float, float]:
    """
    Compute a *coupling-based effective distance* metric.

    Steps:
      - Convert substrate.couplings to NumPy, take absolute value.
      - Choose a "center" node = one with maximal total |J_ij|.
      - Define edge weights w_ij = 1 / (|J_ij| + eps) for |J_ij|>0, else inf.
      - Run Dijkstra from center to get shortest-path distances.
      - Return:
          center_index,
          mean effective distance to other nodes,
          90% quantile of the distance distribution (eff_dist_90).

    If graph is disconnected (shouldn't be with dense init), infinities are
    ignored in the averages.
    """
    J = _to_numpy(substrate.couplings)
    n = J.shape[0]
    if n == 0:
        return 0, 0.0, 0.0

    J_abs = np.abs(J)
    np.fill_diagonal(J_abs, 0.0)

    row_sum = np.sum(J_abs, axis=1)
    if np.all(row_sum == 0.0):
        return 0, 0.0, 0.0

    center = int(np.argmax(row_sum))

    # Edge weights (larger |J| = shorter edge; zero |J| = no edge)
    eps = 1e-9
    with np.errstate(divide="ignore"):
        W = 1.0 / (J_abs + eps)

    # No edge where |J| == 0
    W[J_abs <= 0.0] = np.inf
    np.fill_diagonal(W, 0.0)

    # Dijkstra from center
    dist = np.full(n, np.inf, dtype=float)
    visited = np.zeros(n, dtype=bool)
    dist[center] = 0.0

    for _ in range(n):
        # pick unvisited node with smallest dist
        mask = ~visited
        if not mask.any():
            break
        candidates = dist.copy()
        candidates[~mask] = np.inf
        i = int(np.argmin(candidates))
        if visited[i] or not np.isfinite(dist[i]):
            break
        visited[i] = True

        # relax neighbors
        row = W[i]
        nbrs = np.where(np.isfinite(row))[0]
        for j in nbrs:
            if visited[j]:
                continue
            alt = dist[i] + row[j]
            if alt < dist[j]:
                dist[j] = alt

    # Exclude self and infinities
    valid = dist[(np.arange(n) != center) & np.isfinite(dist)]
    if valid.size == 0:
        return center, 0.0, 0.0

    eff_mean = float(np.mean(valid))

    # 90% quantile
    sorted_d = np.sort(valid)
    idx_90 = int(0.9 * (sorted_d.size - 1))
    eff_90 = float(sorted_d[idx_90])

    return center, eff_mean, eff_90


def bfs_radius_90(
    neighbors: List[List[int]],
    center: int,
    max_radius: int,
) -> float:
    """
    Simple BFS-based "graph radius": the smallest shell radius R such that
    at least 90% of reachable nodes are within distance <= R.

    This is purely topological (ignores coupling magnitudes).
    """
    from collections import deque

    n = len(neighbors)
    visited = np.zeros(n, dtype=bool)
    dist = np.full(n, np.inf, dtype=float)

    q = deque()
    visited[center] = True
    dist[center] = 0.0
    q.append(center)

    while q:
        i = q.popleft()
        d = dist[i]
        if d >= max_radius:
            continue
        for j in neighbors[i]:
            if not visited[j]:
                visited[j] = True
                dist[j] = d + 1.0
                q.append(j)

    valid = dist[np.isfinite(dist)]
    if valid.size == 0:
        return 0.0

    sorted_d = np.sort(valid)
    idx_90 = int(0.9 * (sorted_d.size - 1))
    return float(sorted_d[idx_90])


# =============================================================================
# Main experiment
# =============================================================================

def run_cosmic_birth(cfg: CosmicBirthConfig) -> None:
    """
    Run a single long substrate evolution from a dense "infinite density"
    initial condition, tracking:

      - coupling-based effective distance metrics
      - graph-based radius metric
      - proton/electron/photon candidate scores
      - proton-electron graph distance

    This is aimed at probing:

      "Does the substrate's own dynamics *push space into existence*,
       before and while particles emerge and bind?"
    """
    base_cfg = Config(
        n_nodes=cfg.n_nodes,
        internal_dim=cfg.internal_dim,
        monogamy_budget=cfg.monogamy_budget,
        defrag_rate=cfg.defrag_rate,
        dt=cfg.dt,
        seed=cfg.seed,
        connectivity=cfg.connectivity,
    )

    print("================================================================")
    print("  Cosmic Birth Experiment (T=0 dense / infinite density toy)")
    print("----------------------------------------------------------------")
    print(f"  n_nodes         = {cfg.n_nodes}")
    print(f"  internal_dim    = {cfg.internal_dim}")
    print(f"  monogamy_budget = {cfg.monogamy_budget}")
    print(f"  defrag_rate     = {cfg.defrag_rate}")
    print(f"  dt              = {cfg.dt}")
    print(f"  seed            = {cfg.seed}")
    print(f"  connectivity    = {cfg.connectivity}  (Config-level)")
    print(f"  total_steps     = {cfg.total_steps}")
    print(f"  record_stride   = {cfg.record_stride}")
    print(f"  max_graph_radius= {cfg.max_graph_radius}")
    print(f"  dense_init      = {cfg.dense_init}")
    print("================================================================\n")

    substrate = Substrate(base_cfg)
    neighbors = substrate._neighbors  # list[list[int]]
    n_nodes = substrate.n_nodes
    dt = base_cfg.dt

    if cfg.dense_init:
        make_dense_initial_couplings(substrate, scale=cfg.dense_init_scale)
        neighbors = substrate._neighbors  # refresh local alias

    # Prepare CSV
    fieldnames = [
        "time",
        "center_node",
        "eff_dist_mean",
        "eff_dist_90",
        "graph_radius_90",
        "proton_id",
        "proton_score",
        "electron_id",
        "electron_score",
        "photon_id",
        "photon_score",
        "graph_dist_pe",
    ]
    f = open(cfg.output_csv, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    print("[cosmic_birth] Beginning evolution...\n")

    t = 0.0
    for step in range(1, cfg.total_steps + 1):
        substrate.evolve(n_steps=1, defrag_rate=cfg.defrag_rate)
        t += dt

        if (step % cfg.record_stride) != 0 and step != cfg.total_steps:
            continue

        # 1) Effective distance metrics from couplings
        center, eff_mean, eff_90 = coupling_distance_metrics(substrate)
        graph_r90 = bfs_radius_90(neighbors, center, cfg.max_graph_radius)

        # 2) Particle candidates & scores
        feats, scores, cands = pd.analyze_substrate(substrate)
        p_id = cands.proton_id
        e_id = cands.electron_id
        ph_id = cands.photon_id

        p_score = cands.proton_score
        e_score = cands.electron_score
        ph_score = cands.photon_score

        # 3) Simple graph distance between proton & electron candidates
        d_pe = graph_distance(neighbors, p_id, e_id, max_radius=cfg.max_graph_radius)

        # Log to stdout
        print(
            f"[t={t:.3f}] center={center}, "
            f"eff_mean={eff_mean:.3f}, eff_90={eff_90:.3f}, "
            f"graph_r90={graph_r90:.2f}, "
            f"p_id={p_id} (score={p_score:.3f}), "
            f"e_id={e_id} (score={e_score:.3f}), "
            f"d_p-e={d_pe}"
        )

        # Write CSV row
        row = {
            "time": t,
            "center_node": center,
            "eff_dist_mean": eff_mean,
            "eff_dist_90": eff_90,
            "graph_radius_90": graph_r90,
            "proton_id": p_id,
            "proton_score": p_score,
            "electron_id": e_id,
            "electron_score": e_score,
            "photon_id": ph_id,
            "photon_score": ph_score,
            "graph_dist_pe": d_pe,
        }
        writer.writerow(row)

    f.close()
    print(f"\n[cosmic_birth] History written to: {cfg.output_csv}")


# =============================================================================
# CLI entry
# =============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="T=0 dense / cosmic birth experiment: "
                    "probe effective distance growth and particle emergence."
    )
    parser.add_argument("--n_nodes", type=int, default=128)
    parser.add_argument("--internal_dim", type=int, default=3)
    parser.add_argument("--monogamy_budget", type=float, default=1.0)
    parser.add_argument("--defrag_rate", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--connectivity", type=float, default=1.0)
    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--record_stride", type=int, default=10)
    parser.add_argument("--max_graph_radius", type=int, default=8)
    parser.add_argument("--output_csv", type=str, default="cosmic_birth_history.csv")
    parser.add_argument("--dense_init", type=int, default=1)
    parser.add_argument("--dense_init_scale", type=float, default=1.0)

    args = parser.parse_args()

    cfg = CosmicBirthConfig(
        n_nodes=args.n_nodes,
        internal_dim=args.internal_dim,
        monogamy_budget=args.monogamy_budget,
        defrag_rate=args.defrag_rate,
        dt=args.dt,
        seed=args.seed,
        connectivity=args.connectivity,
        total_steps=args.total_steps,
        record_stride=args.record_stride,
        max_graph_radius=args.max_graph_radius,
        output_csv=args.output_csv,
        dense_init=bool(args.dense_init),
        dense_init_scale=args.dense_init_scale,
    )

    run_cosmic_birth(cfg)


if __name__ == "__main__":
    main()
