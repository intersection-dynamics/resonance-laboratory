#!/usr/bin/env python3
"""
fermion_topology_probe.py

Probe the graph topology of "fermion-like" vs "boson-like" clusters
from a Hilbert substrate run.

Pipeline:

  1. Read pattern_detector params.json to reconstruct the Substrate dynamics.
  2. Read particle_cluster_analysis cluster_assignments.csv to know,
     for each (step, node), which cluster each node belongs to.
  3. Optionally read particle_cluster_analysis summary.json to auto-select
     fermion-like and boson-like clusters based on electron-like vs photon-like
     fractions.
  4. Re-run the Substrate evolution from t=0 up to max step seen in
     cluster_assignments, and at each recorded step:
       - Identify fermion nodes (in fermion_clusters).
       - Identify boson nodes (in boson_clusters).
       - Build an adjacency mask from couplings |J_ij| > eps.
       - Compute simple topological statistics:
           * edge fraction where both ends are fermion nodes (FF)
           * edge fraction where both ends are boson nodes (BB)
           * edge fraction where one fermion, one boson (FB)
           * mean graph distance between fermion nodes (approx via BFS)
       - Compare FF vs what you'd expect from random choice of nodes
         with the same fermion fraction (binomial null model).
  5. Write a CSV time series + a summary.json with averages.

Usage (Windows example):

  python fermion_topology_probe.py ^
      --pattern-run 20251201T190712Z_proton_electron_photon_scan ^
      --cluster-run 20251201T201234Z_quark_gluon_probe ^
      --output-root outputs ^
      --tag fermion_topology

If you don't supply fermion/boson cluster IDs, the script will:
  - Load cluster summary.json for cluster-run
  - Tag clusters as "fermion-like" if:
        electron_like_fraction >= 0.5 and photon_like_fraction <= 0.2
  - Tag clusters as "boson-like" if:
        photon_like_fraction >= 0.5
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

from substrate import Config, Substrate  # type: ignore


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_run_id(tag: str | None) -> str:
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if tag:
        safe_tag = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in str(tag)
        )
        return f"{now}_{safe_tag}"
    return now


def _write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------
# Cluster metadata loading
# ---------------------------------------------------------------------


@dataclass
class ClusterMeta:
    fermion_clusters: List[int]
    boson_clusters: List[int]


def _load_cluster_meta(summary_path: str) -> ClusterMeta:
    """
    Load cluster summary.json from particle_cluster_analysis and infer
    fermion-like vs boson-like cluster IDs.

    Heuristics:
      - fermion-like: electron_like_fraction >= 0.5 and photon_like_fraction <= 0.2
      - boson-like:   photon_like_fraction >= 0.5
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    fermion_clusters: List[int] = []
    boson_clusters: List[int] = []

    clusters = summary.get("clusters", [])
    for c in clusters:
        cid = int(c.get("cluster_id"))
        species = c.get("species_domination", {})
        e_frac = float(species.get("electron_like_fraction", 0.0))
        ph_frac = float(species.get("photon_like_fraction", 0.0))

        if e_frac >= 0.5 and ph_frac <= 0.2:
            fermion_clusters.append(cid)
        if ph_frac >= 0.5:
            boson_clusters.append(cid)

    return ClusterMeta(fermion_clusters=fermion_clusters, boson_clusters=boson_clusters)


# ---------------------------------------------------------------------
# Cluster assignments loading
# ---------------------------------------------------------------------


@dataclass
class AssignmentData:
    steps: np.ndarray
    nodes: np.ndarray
    clusters: np.ndarray


def _load_cluster_assignments(csv_path: str) -> AssignmentData:
    """
    Load cluster_assignments.csv from particle_cluster_analysis.

    CSV columns:
      step,t,node_id,cluster_id,proton_score,electron_score,photon_score
    """
    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    steps = data["step"].astype(int)
    nodes = data["node_id"].astype(int)
    clusters = data["cluster_id"].astype(int)
    return AssignmentData(steps=steps, nodes=nodes, clusters=clusters)


# ---------------------------------------------------------------------
# Topology metrics
# ---------------------------------------------------------------------


def _build_adjacency_from_couplings(J: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """
    Build a boolean adjacency matrix A from couplings J by thresholding |J_ij| > eps.
    Ensures no self-edges.
    """
    mag = np.abs(J)
    if eps > 0.0:
        A = mag > eps
    else:
        A = mag > 0.0
    np.fill_diagonal(A, False)
    return A


def _edge_fractions(
    A: np.ndarray,
    fermion_mask: np.ndarray,
    boson_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Compute fractions of edges in each category:

      FF: both ends fermion
      BB: both ends boson
      FB: one fermion, one boson
      other: all other edges

    Returns dict with fractions relative to total number of edges in A
    (counting each undirected edge once).
    """
    n = A.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    edge_mask = A[iu, ju]
    if not np.any(edge_mask):
        return {
            "frac_FF": 0.0,
            "frac_BB": 0.0,
            "frac_FB": 0.0,
            "frac_other": 0.0,
            "n_edges": 0,
        }

    i_edges = iu[edge_mask]
    j_edges = ju[edge_mask]
    n_edges = i_edges.shape[0]

    fi = fermion_mask[i_edges]
    fj = fermion_mask[j_edges]
    bi = boson_mask[i_edges]
    bj = boson_mask[j_edges]

    is_FF = fi & fj
    is_BB = bi & bj
    is_FB = (fi & bj) | (fj & bi)

    n_FF = int(np.sum(is_FF))
    n_BB = int(np.sum(is_BB))
    n_FB = int(np.sum(is_FB))
    n_other = n_edges - n_FF - n_BB - n_FB

    return {
        "frac_FF": n_FF / n_edges,
        "frac_BB": n_BB / n_edges,
        "frac_FB": n_FB / n_edges,
        "frac_other": n_other / n_edges,
        "n_edges": n_edges,
    }


def _random_expected_ff_fraction(
    n_nodes: int,
    fermion_fraction: float,
) -> float:
    """
    Under a simple binomial null model, the probability that
    a random undirected edge connects two fermions is ~f^2,
    where f is the fraction of fermion nodes.

    This ignores degree inhomogeneities but is fine as a baseline.
    """
    f = np.clip(fermion_fraction, 0.0, 1.0)
    return float(f * f)


def _mean_fermion_distance(
    A: np.ndarray,
    fermion_mask: np.ndarray,
    max_sample: int = 16,
    max_depth: int = 5,
) -> float:
    """
    Approximate mean graph distance between fermion nodes using BFS.

    We:
      - sample up to max_sample fermion nodes
      - run BFS out to max_depth
      - collect distances to other fermion nodes discovered
      - return the mean of those finite distances

    If there are no fermions or no finite fermion-fermion distances, returns NaN.
    """
    fermion_indices = np.where(fermion_mask)[0]
    n_f = fermion_indices.shape[0]
    if n_f <= 1:
        return float("nan")

    if n_f <= max_sample:
        seeds = fermion_indices
    else:
        rng = np.random.default_rng(0)
        seeds = rng.choice(fermion_indices, size=max_sample, replace=False)

    n = A.shape[0]
    dists: List[int] = []

    for s in seeds:
        # BFS from s
        dist = -np.ones(n, dtype=int)
        dist[s] = 0
        frontier = [s]
        depth = 0
        while frontier and depth < max_depth:
            next_frontier: List[int] = []
            for u in frontier:
                neighbors = np.where(A[u])[0]
                for v in neighbors:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        next_frontier.append(v)
            frontier = next_frontier
            depth += 1

        # Collect distances to other fermion nodes
        for f_idx in fermion_indices:
            if f_idx == s:
                continue
            d = dist[f_idx]
            if d > 0:
                dists.append(int(d))

    if not dists:
        return float("nan")

    return float(np.mean(dists))


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Probe topology of fermion-like vs boson-like clusters "
            "from a Hilbert substrate pattern_detector run."
        )
    )

    p.add_argument(
        "--pattern-root",
        type=str,
        default="outputs/pattern_detector_scan",
        help="Root directory where pattern_detector runs are stored.",
    )
    p.add_argument(
        "--pattern-run",
        type=str,
        required=True,
        help="pattern_detector run ID (e.g. 20251201T190712Z_proton_electron_photon_scan).",
    )

    p.add_argument(
        "--cluster-root",
        type=str,
        default="outputs/particle_cluster_analysis",
        help="Root directory where particle_cluster_analysis runs are stored.",
    )
    p.add_argument(
        "--cluster-run",
        type=str,
        required=True,
        help="particle_cluster_analysis run ID containing summary.json and cluster_assignments.csv.",
    )

    p.add_argument(
        "--fermion-clusters",
        type=str,
        default="",
        help=(
            "Comma-separated list of cluster IDs to treat as fermion-like. "
            "If empty, will infer from cluster summary."
        ),
    )
    p.add_argument(
        "--boson-clusters",
        type=str,
        default="",
        help=(
            "Comma-separated list of cluster IDs to treat as boson-like. "
            "If empty, will infer from cluster summary."
        ),
    )

    p.add_argument(
        "--max-step",
        type=int,
        default=-1,
        help=(
            "Maximum step to analyze (inclusive). "
            "If negative, use max step from cluster_assignments.csv."
        ),
    )

    p.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Threshold on |J_ij| to define adjacency (0.0 = any nonzero coupling).",
    )

    p.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory for fermion_topology_probe outputs.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="fermion_topology_probe",
        help="Tag for this analysis run.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Locate pattern_detector run and params
    pattern_run_dir = os.path.join(args.pattern_root, args.pattern_run)
    params_path = os.path.join(pattern_run_dir, "params.json")
    if not os.path.isfile(params_path):
        raise FileNotFoundError(f"pattern params.json not found at: {params_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        pattern_params = json.load(f)

    # Locate cluster run
    cluster_run_dir = os.path.join(args.cluster_root, args.cluster_run)
    cluster_summary_path = os.path.join(cluster_run_dir, "summary.json")
    cluster_csv_path = os.path.join(cluster_run_dir, "data", "cluster_assignments.csv")
    if not os.path.isfile(cluster_summary_path):
        raise FileNotFoundError(f"cluster summary.json not found at: {cluster_summary_path}")
    if not os.path.isfile(cluster_csv_path):
        raise FileNotFoundError(f"cluster_assignments.csv not found at: {cluster_csv_path}")

    # Fermion/boson cluster sets
    meta = _load_cluster_meta(cluster_summary_path)
    fermion_clusters_auto = meta.fermion_clusters
    boson_clusters_auto = meta.boson_clusters

    def _parse_cluster_list(s: str) -> List[int]:
        s = s.strip()
        if not s:
            return []
        return [int(tok) for tok in s.split(",") if tok.strip()]

    fermion_clusters = _parse_cluster_list(args.fermion_clusters)
    boson_clusters = _parse_cluster_list(args.boson_clusters)

    if not fermion_clusters:
        fermion_clusters = fermion_clusters_auto
    if not boson_clusters:
        boson_clusters = boson_clusters_auto

    if not fermion_clusters:
        raise RuntimeError("No fermion-like clusters found or specified.")
    if not boson_clusters:
        raise RuntimeError("No boson-like clusters found or specified.")

    # Load cluster assignments
    assign = _load_cluster_assignments(cluster_csv_path)
    unique_steps = np.unique(assign.steps)
    if args.max_step >= 0:
        unique_steps = unique_steps[unique_steps <= args.max_step]
    max_step = int(unique_steps.max())

    # Set up output
    run_id = _make_run_id(args.tag)
    run_root = os.path.join(args.output_root, "fermion_topology_probe", run_id)
    data_dir = os.path.join(run_root, "data")
    _ensure_dir(run_root)
    _ensure_dir(data_dir)

    print("=" * 60)
    print("  Fermion Topology Probe")
    print("=" * 60)
    print(f"pattern run: {pattern_run_dir}")
    print(f"cluster run: {cluster_run_dir}")
    print(f"fermion clusters: {fermion_clusters}")
    print(f"boson clusters:   {boson_clusters}")
    print(f"analyzing steps up to: {max_step}")
    print(f"eps for adjacency: {args.eps}")
    print(f"output: {run_root}")
    print("=" * 60)

    # Reconstruct Substrate from Config
    cfg = Config(
        n_nodes=int(pattern_params["n_nodes"]),
        internal_dim=int(pattern_params["internal_dim"]),
        monogamy_budget=float(pattern_params["monogamy_budget"]),
        defrag_rate=float(pattern_params["defrag_rate"]),
        dt=float(pattern_params["dt"]),
        seed=int(pattern_params["seed"]),
        connectivity=float(pattern_params["connectivity"]),
        use_gpu=bool(pattern_params.get("use_gpu", False)),
    )
    sub = Substrate(cfg)

    # Prepare CSV
    topo_csv_path = os.path.join(data_dir, "fermion_topology_timeseries.csv")
    with open(topo_csv_path, "w", encoding="utf-8") as f:
        f.write(
            "step,t,"
            "n_edges,"
            "fermion_fraction,boson_fraction,"
            "frac_FF,frac_BB,frac_FB,frac_other,"
            "ff_fraction_random,"
            "mean_fermion_distance\n"
        )

    dt = float(pattern_params["dt"])

    # Map from step to indices in assignments
    step_to_mask: Dict[int, np.ndarray] = {}
    for s in unique_steps:
        step_to_mask[int(s)] = (assign.steps == int(s))

    all_ff: List[float] = []
    all_bb: List[float] = []
    all_fd: List[float] = []

    # Evolve substrate and measure at recorded steps
    current_step = 0
    for target_step in unique_steps:
        target_step = int(target_step)
        if target_step > current_step:
            sub.evolve(n_steps=target_step - current_step)
            current_step = target_step

        t = current_step * dt

        mask = step_to_mask[current_step]
        if not np.any(mask):
            continue

        node_ids = assign.nodes[mask]
        cluster_ids = assign.clusters[mask]

        n_nodes = cfg.n_nodes
        node_cluster = -np.ones(n_nodes, dtype=int)
        for nid, cid in zip(node_ids, cluster_ids):
            node_cluster[int(nid)] = int(cid)

        fermion_mask = np.isin(node_cluster, fermion_clusters)
        boson_mask = np.isin(node_cluster, boson_clusters)

        fermion_fraction = float(np.sum(fermion_mask)) / float(n_nodes)
        boson_fraction = float(np.sum(boson_mask)) / float(n_nodes)

        J = sub.couplings
        if hasattr(J, "get"):
            J_np = J.get()
        else:
            J_np = np.asarray(J)

        A = _build_adjacency_from_couplings(J_np, eps=args.eps)
        edge_stats = _edge_fractions(A, fermion_mask, boson_mask)
        ff_rand = _random_expected_ff_fraction(n_nodes, fermion_fraction)
        mean_fd = _mean_fermion_distance(A, fermion_mask)

        all_ff.append(edge_stats["frac_FF"])
        all_bb.append(edge_stats["frac_BB"])
        all_fd.append(mean_fd)

        with open(topo_csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{current_step},{t:.8f},"
                f"{edge_stats['n_edges']},"
                f"{fermion_fraction:.6f},{boson_fraction:.6f},"
                f"{edge_stats['frac_FF']:.6f},"
                f"{edge_stats['frac_BB']:.6f},"
                f"{edge_stats['frac_FB']:.6f},"
                f"{edge_stats['frac_other']:.6f},"
                f"{ff_rand:.6f},"
                f"{mean_fd if np.isfinite(mean_fd) else float('nan')}\n"
            )

    # Summary
    def _finite_mean(xs: List[float]) -> float:
        arr = np.array(xs, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(arr.mean())

    summary = {
        "run_id": _make_run_id(args.tag),
        "pattern_run": args.pattern_run,
        "cluster_run": args.cluster_run,
        "fermion_clusters": fermion_clusters,
        "boson_clusters": boson_clusters,
        "eps": args.eps,
        "mean_frac_FF": _finite_mean(all_ff),
        "mean_frac_BB": _finite_mean(all_bb),
        "mean_mean_fermion_distance": _finite_mean(all_fd),
    }
    _write_json(os.path.join(run_root, "summary.json"), summary)

    print("=" * 60)
    print("Fermion topology probe complete.")
    print(f"Timeseries CSV: {topo_csv_path}")
    print(f"Summary:        {os.path.join(run_root, 'summary.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
