#!/usr/bin/env python3
"""
================================================================================
SUBSTRATE C2 CONFIGURATION-SPACE ANALYSIS (WITH DEFECTS)
================================================================================

Purpose
-------
Given your Hilbert substrate graph (SubstrateGraph from substrate.py), we:

  1. Build the base entanglement graph (all nodes).
  2. Evolve the substrate so excitations can emerge.
  3. Detect high-excitation "defect" nodes via ExcitationDetector.
  4. Build two configuration-space graphs for *two hard-core defects*:
       - C2_full:  all nodes allowed
       - C2_def:   only defect nodes allowed
  5. Measure:
       - Effective dimension of base graph
       - Effective dimension of C2_full and C2_def
       - Short-cycle counts in C2_full and C2_def

This lets us ask:
  "Is the *configuration space of actual excitations* closer to 1D, 2D, or 3D,
   even if the raw entanglement graph is messy?"

Usage
-----
Typical invocation from the repo root:

  python substrate_C2_analysis.py ^
      --n-nodes 25 ^
      --internal-dim 2 ^
      --connectivity 0.25 ^
      --n-evolution-steps 50 ^
      --dt 0.1 ^
      --excitation-threshold 0.05 ^
      --max-defects 12 ^
      --tag defects_scan

Outputs
-------
Creates a run folder:

  outputs/substrate_C2_analysis/<timestamp>_<tag>/

with:

  - params.json
  - metadata.json
  - summary.json
  - logs/run.log
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, Tuple, List

import numpy as np

from substrate import (
    SubstrateGraph,
    ExcitationDetector,
    TopologyAnalyzer,
)


# =============================================================================
# BASE GRAPH → INTEGER ADJACENCY
# =============================================================================

def build_integer_adjacency_from_substrate(
    substrate: SubstrateGraph,
) -> Tuple[Dict[int, Set[int]], Dict[str, int], List[str]]:
    """
    Convert SubstrateGraph nodes/edges into an integer-labelled adjacency dict.

    Returns:
        adj:        dict[int, set[int]]  (base graph adjacency)
        id_to_idx:  map from node.id (str) -> int index
        idx_to_id:  list of node.id indexed by int index
    """
    # Stable ordering of nodes
    node_ids = list(substrate.nodes.keys())
    node_ids.sort()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    idx_to_id = node_ids[:]  # same order

    adj: Dict[int, Set[int]] = {i: set() for i in range(len(node_ids))}

    for edge in substrate.edges.values():
        na = edge.node_a.id
        nb = edge.node_b.id
        i = id_to_idx[na]
        j = id_to_idx[nb]
        if i == j:
            continue
        adj[i].add(j)
        adj[j].add(i)

    return adj, id_to_idx, idx_to_id


# =============================================================================
# TWO-DEFECT CONFIGURATION SPACE C2
# =============================================================================

def build_two_defect_config_graph(
    base_adj: Dict[int, Set[int]]
) -> Tuple[Dict[int, Set[int]], List[Tuple[int, int]]]:
    """
    Build the two-defect configuration-space graph C2 from a base adjacency.

    Nodes of C2 are unordered pairs (i, j) with i < j, where i, j are base nodes.
    Edges of C2 connect configurations that differ by moving exactly one defect
    along one base-graph edge, while respecting hard-core exclusion (i != j).

    Args:
        base_adj: dict int -> set[int], adjacency of base graph

    Returns:
        cfg_adj:   dict int -> set[int], adjacency of C2
        cfg_pairs: list of (i, j) giving the base-node pair for each C2 node index
    """
    base_nodes = sorted(base_adj.keys())

    # All unordered pairs i < j
    cfg_pairs: List[Tuple[int, int]] = []
    for idx_i, i in enumerate(base_nodes):
        for j in base_nodes[idx_i + 1:]:
            cfg_pairs.append((i, j))

    pair_to_idx: Dict[Tuple[int, int], int] = {p: k for k, p in enumerate(cfg_pairs)}
    cfg_adj: Dict[int, Set[int]] = {k: set() for k in range(len(cfg_pairs))}

    # For each configuration (a, b), move one defect along a base edge
    for k, (a, b) in enumerate(cfg_pairs):
        # Move defect at a
        for a2 in base_adj[a]:
            if a2 == b:
                continue  # hard-core: cannot occupy same site
            p2 = tuple(sorted((a2, b)))
            j = pair_to_idx.get(p2)
            if j is not None and j != k:
                cfg_adj[k].add(j)
                cfg_adj[j].add(k)

        # Move defect at b
        for b2 in base_adj[b]:
            if b2 == a:
                continue  # hard-core
            p2 = tuple(sorted((a, b2)))
            j = pair_to_idx.get(p2)
            if j is not None and j != k:
                cfg_adj[k].add(j)
                cfg_adj[j].add(k)

    return cfg_adj, cfg_pairs


def restrict_adjacency(
    base_adj: Dict[int, Set[int]],
    allowed_nodes: Set[int],
) -> Dict[int, Set[int]]:
    """
    Restrict a base adjacency to a subset of nodes.

    Keeps only nodes in allowed_nodes and only edges between allowed_nodes.
    """
    if not allowed_nodes:
        return {}

    sub_adj: Dict[int, Set[int]] = {}
    for i in allowed_nodes:
        nbrs = base_adj.get(i, set())
        sub_adj[i] = {j for j in nbrs if j in allowed_nodes}
    return sub_adj


# =============================================================================
# GRAPH METRICS: EFFECTIVE DIMENSION & CYCLE COUNT
# =============================================================================

def graph_effective_dimension(adj: Dict[int, Set[int]]) -> float:
    """
    Estimate effective dimension from graph structure via neighborhood growth:

        |B(r)| ~ r^d  =>  log |B(r)| ~ d log r

    We:
        - BFS from an arbitrary root
        - count how many nodes lie at each distance
        - fit log(cumulative count) vs log(distance)

    Returns:
        slope ~ effective dimension d
    """
    if not adj:
        return 0.0

    # Pick an arbitrary root
    start = next(iter(adj.keys()))

    # BFS to get distances
    from collections import deque
    dist: Dict[int, int] = {start: 0}
    q = deque([start])

    while q:
        v = q.popleft()
        dv = dist[v]
        for w in adj[v]:
            if w not in dist:
                dist[w] = dv + 1
                q.append(w)

    # distance_counts[d] = number of nodes at distance d
    distance_counts: Dict[int, int] = {}
    for d in dist.values():
        distance_counts[d] = distance_counts.get(d, 0) + 1

    # Need at least two shells beyond distance 0
    distances = sorted(d for d in distance_counts.keys() if d > 0)
    if len(distances) < 2:
        return 0.0

    cumulative = []
    total = 0
    for d in distances:
        total += distance_counts[d]
        cumulative.append(total)

    if len(distances) < 2:
        return 0.0

    log_d = np.log(np.array(distances, dtype=float))
    log_c = np.log(np.array(cumulative, dtype=float))

    # Simple linear fit
    slope, _ = np.polyfit(log_d, log_c, 1)
    return float(slope)


def count_short_cycles(adj: Dict[int, Set[int]], max_length: int = 6) -> int:
    """
    Very simple cycle counter for undirected graphs, restricted to short cycles.

    We perform DFS from each node up to depth max_length and count distinct
    simple cycles (no repeated internal vertices) that return to the start.

    Returns:
        number of distinct cycles of length <= max_length
    """
    if not adj:
        return 0

    cycles: Set[Tuple[int, ...]] = set()
    nodes_sorted = sorted(adj.keys())

    def dfs(start: int, current: int, visited: List[int]):
        if len(visited) > max_length:
            return

        for nxt in adj[current]:
            if nxt == start and len(visited) >= 3:
                # Found a cycle; canonicalize by sorted node list
                cyc_nodes = tuple(sorted(visited))
                cycles.add(cyc_nodes)
            elif nxt not in visited and nxt > start:
                dfs(start, nxt, visited + [nxt])

    for start in nodes_sorted:
        dfs(start, start, [start])

    return len(cycles)


# =============================================================================
# I/O AND EXPERIMENT SCHEMA
# =============================================================================

def make_run_dir(output_root: str, tag: str) -> Tuple[Path, str]:
    """
    Create a run directory under outputs/ following the Software Guide.

        outputs/<experiment_name>/<run_id>/

    where:
        experiment_name = "substrate_C2_analysis"
        run_id = timestamp + optional tag
    """
    experiment_name = "substrate_C2_analysis"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{tag}" if tag else timestamp

    root = Path(output_root)
    run_dir = root / experiment_name / run_id

    # Subfolders per Software Guide
    (run_dir / "data").mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=False)

    return run_dir, run_id


def save_json(path: Path, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# =============================================================================
# CORE EXPERIMENT LOGIC
# =============================================================================

def run_experiment(params: dict) -> dict:
    """
    Core logic: build substrate, evolve, detect defects, build C2 graphs, compute metrics.
    Returns a results dict that can be saved to summary.json.
    """
    # Seed
    seed = params.get("seed", None)
    if seed is not None:
        np.random.seed(seed)

    n_nodes = int(params["n_nodes"])
    internal_dim = int(params["internal_dim"])
    connectivity = float(params["connectivity"])
    n_evolution_steps = int(params["n_evolution_steps"])
    dt = float(params["dt"])
    excitation_threshold = float(params["excitation_threshold"])
    max_defects = int(params["max_defects"])

    # ------------------------------------------------------------------
    # 1. Build substrate from noise
    # ------------------------------------------------------------------
    substrate = SubstrateGraph(
        n_nodes=n_nodes,
        internal_dim=internal_dim,
        connectivity=connectivity,
    )

    # Base graph metrics using your TopologyAnalyzer
    topo = TopologyAnalyzer(substrate)
    base_dim = topo.effective_dimension()
    base_pi1 = topo.fundamental_group_probe()

    # Integer adjacency (all nodes)
    base_adj, id_to_idx, idx_to_id = build_integer_adjacency_from_substrate(substrate)
    base_edges = sum(len(nbrs) for nbrs in base_adj.values()) // 2

    # ------------------------------------------------------------------
    # 2. Evolve substrate to let structure/excitations emerge
    # ------------------------------------------------------------------
    if n_evolution_steps > 0:
        substrate.evolve(dt=dt, n_steps=n_evolution_steps)

    # ------------------------------------------------------------------
    # 3. Detect excitations / defects
    # ------------------------------------------------------------------
    detector = ExcitationDetector(substrate)
    excitation_map = detector.excitation_map()

    # First, threshold-based selection
    defects = [
        node for node in substrate.nodes.values()
        if detector.node_excitation(node) > excitation_threshold
    ]

    # If too many, keep top-K by excitation
    if len(defects) > max_defects:
        defects = sorted(
            defects,
            key=lambda n: detector.node_excitation(n),
            reverse=True,
        )[:max_defects]

    # If none above threshold, pick the top-K nodes anyway (so we always have something)
    if not defects:
        sorted_nodes = sorted(
            substrate.nodes.values(),
            key=lambda n: detector.node_excitation(n),
            reverse=True,
        )
        defects = sorted_nodes[:max_defects]

    defect_ids = [node.id for node in defects]
    defect_indices: Set[int] = {id_to_idx[nid] for nid in defect_ids}

    # ------------------------------------------------------------------
    # 4. Build C2: full and defect-restricted
    # ------------------------------------------------------------------
    # Full C2 (all nodes)
    cfg_adj_full, cfg_pairs_full = build_two_defect_config_graph(base_adj)
    cfg_dim_full = graph_effective_dimension(cfg_adj_full)
    cfg_cycles_full = count_short_cycles(cfg_adj_full, max_length=6)
    cfg_edges_full = sum(len(nbrs) for nbrs in cfg_adj_full.values()) // 2

    # Defect-restricted base adjacency
    base_adj_def = restrict_adjacency(base_adj, defect_indices)

    # Defect C2
    cfg_adj_def, cfg_pairs_def = build_two_defect_config_graph(base_adj_def)
    cfg_dim_def = graph_effective_dimension(cfg_adj_def)
    cfg_cycles_def = count_short_cycles(cfg_adj_def, max_length=6)
    cfg_edges_def = sum(len(nbrs) for nbrs in cfg_adj_def.values()) // 2

    # ------------------------------------------------------------------
    # 5. Package results
    # ------------------------------------------------------------------
    results = {
        "framework_version": "0.2.0",
        "script": "substrate_C2_analysis.py",
        "params": params,
        "metrics": {
            "base": {
                "n_nodes": n_nodes,
                "n_edges": base_edges,
                "effective_dimension": base_dim,
                "fundamental_group_probe": base_pi1,
            },
            "C2_full": {
                "n_nodes": len(cfg_adj_full),
                "n_edges": cfg_edges_full,
                "effective_dimension": cfg_dim_full,
                "short_cycle_count_max6": cfg_cycles_full,
            },
            "C2_defects": {
                "n_defect_nodes": len(defect_indices),
                "defect_node_ids": defect_ids,
                "n_nodes": len(cfg_adj_def),
                "n_edges": cfg_edges_def,
                "effective_dimension": cfg_dim_def,
                "short_cycle_count_max6": cfg_cycles_def,
            },
        },
        "diagnostics": {
            "excitation_map": excitation_map,
            "excitation_threshold": excitation_threshold,
            "max_defects": max_defects,
            "note": (
                "Defect nodes selected by excitation level; C2_defects is the "
                "two-defect configuration space restricted to those nodes."
            ),
        },
        "verdicts": {
            "C2_full_dimension_hint": (
                "1D-ish" if cfg_dim_full < 1.5 else
                "2D-ish" if cfg_dim_full < 2.2 else
                "3D-ish or higher"
            ),
            "C2_defects_dimension_hint": (
                "insufficient" if len(cfg_adj_def) == 0 else
                "1D-ish" if cfg_dim_def < 1.5 else
                "2D-ish" if cfg_dim_def < 2.2 else
                "3D-ish or higher"
            ),
        },
    }

    return results


# =============================================================================
# HUMAN-READABLE SUMMARY
# =============================================================================

def print_summary(results: dict):
    m = results["metrics"]
    base = m["base"]
    C2f = m["C2_full"]
    C2d = m["C2_defects"]

    print("======================================================================")
    print("SUBSTRATE C2 CONFIGURATION-SPACE ANALYSIS (WITH DEFECTS)")
    print("======================================================================\n")

    print("Base entanglement graph:")
    print(f"  Nodes:               {base['n_nodes']}")
    print(f"  Edges:               {base['n_edges']}")
    print(f"  Effective dimension: {base['effective_dimension']:.3f}")
    print(f"  π1 probe:            {base['fundamental_group_probe']}")
    print()

    print("Two-defect configuration space C2 (ALL nodes):")
    print(f"  C2_full nodes (pairs):      {C2f['n_nodes']}")
    print(f"  C2_full edges (moves):      {C2f['n_edges']}")
    print(f"  C2_full effective dim:      {C2f['effective_dimension']:.3f}")
    print(f"  C2_full short cycles (≤6):  {C2f['short_cycle_count_max6']}")
    print(f"  Dimensionality hint:        {results['verdicts']['C2_full_dimension_hint']}")
    print()

    print("Two-defect configuration space C2 (DEFECT nodes only):")
    print(f"  Defect nodes (sites):       {C2d['n_defect_nodes']}")
    print(f"  C2_def nodes (pairs):       {C2d['n_nodes']}")
    print(f"  C2_def edges (moves):       {C2d['n_edges']}")
    print(f"  C2_def effective dim:       {C2d['effective_dimension']:.3f}")
    print(f"  C2_def short cycles (≤6):   {C2d['short_cycle_count_max6']}")
    print(f"  Dimensionality hint:        {results['verdicts']['C2_defects_dimension_hint']}")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze two-defect configuration-space topology for the Hilbert substrate, including a defect-restricted sector."
    )
    parser.add_argument("--n-nodes", type=int, default=16,
                        help="Number of substrate nodes (default: 16)")
    parser.add_argument("--internal-dim", type=int, default=2,
                        help="Internal Hilbert dimension per node (1, 2, or 3; default: 2)")
    parser.add_argument("--connectivity", type=float, default=0.3,
                        help="Probability of initial entanglement edge between nodes (default: 0.3)")
    parser.add_argument("--n-evolution-steps", type=int, default=50,
                        help="Number of evolution steps before defect detection (default: 50)")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Time step for substrate.evolve (default: 0.1)")
    parser.add_argument("--excitation-threshold", type=float, default=0.05,
                        help="Excitation threshold for defect selection (default: 0.05)")
    parser.add_argument("--max-defects", type=int, default=16,
                        help="Maximum number of defect nodes to keep (default: 16)")
    parser.add_argument("--output-root", type=str, default="outputs",
                        help="Root directory for outputs (default: outputs)")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional run tag appended to timestamp")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: None)")

    args = parser.parse_args()

    # Build params dict
    params = {
        "n_nodes": args.n_nodes,
        "internal_dim": args.internal_dim,
        "connectivity": args.connectivity,
        "n_evolution_steps": args.n_evolution_steps,
        "dt": args.dt,
        "excitation_threshold": args.excitation_threshold,
        "max_defects": args.max_defects,
        "seed": args.seed,
    }

    # Create run directory
    run_dir, run_id = make_run_dir(args.output_root, args.tag)

    # Simple logging: print + log file
    log_path = run_dir / "logs" / "run.log"

    def log(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("======================================================================")
    log("SUBSTRATE C2 CONFIGURATION-SPACE ANALYSIS (WITH DEFECTS)")
    log("======================================================================")
    log(f"Run ID:               {run_id}")
    log(f"n_nodes:              {args.n_nodes}")
    log(f"internal_dim:         {args.internal_dim}")
    log(f"connectivity:         {args.connectivity}")
    log(f"n_evolution_steps:    {args.n_evolution_steps}")
    log(f"dt:                   {args.dt}")
    log(f"excitation_threshold: {args.excitation_threshold}")
    log(f"max_defects:          {args.max_defects}")
    log(f"seed:                 {args.seed}")
    log("------------------------------------------------------------------")

    # Run experiment
    results = run_experiment(params)

    # Print human-readable summary to console (and therefore run.log)
    print_summary(results)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n[Console summary printed above]\n")

    # Save params.json, metadata.json, summary.json
    save_json(run_dir / "params.json", params)

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "script": "substrate_C2_analysis.py",
    }
    save_json(run_dir / "metadata.json", metadata)

    summary = {
        **results,
        "run_id": run_id,
        "timestamp": metadata["timestamp"],
    }
    save_json(run_dir / "summary.json", summary)

    log("Done. Summary written to summary.json")


if __name__ == "__main__":
    main()
