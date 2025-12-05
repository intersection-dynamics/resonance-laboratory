"""
fermion_transition_graph.py

Build a directed transition graph over internal states for a single
Precipitating Event run.

Expected files under run_root:

  analysis_microstates/microstates.json
  analysis_microstates/internal_clusters.json

Formats:

  microstates.json:
    {
      "geometry": "...",
      "n_sites": ...,
      "n_times": ...,
      "n_unique_microstates": ...,
      "run_root": "...",
      "microstates": [
        {
          "pattern_bits": "01010110",
          "center_site": 4,
          "times": [...],
          "count": 3,
          ...
        },
        ...
      ]
    }

  internal_clusters.json:
    {
      "center_sites": {...},
      "cluster_radius": ...,
      "n_sites": ...,
      "n_states": ...,
      "n_times": ...,
      "run_root": "...",
      "transitions": {
        "states": [
          {
            "state_id": 0,
            "pattern_bits": "11100010",
            "center_site": 4,
            "count": 1,
            "internal_change_count": 0,
            ...
          },
          ...
        ],
        "edges": [
          { "from_state": 0, "to_state": 1, "count": 1 },
          ...
        ]
      }
    }

The script:

  * Loads microstates and internal states.
  * Builds weighted adjacency over internal states.
  * Runs Tarjan SCC to find strongly connected components.
  * Computes degree-based hub stats.
  * Writes outputs into:

      run_root/analysis_transition_graph/
        adjacency.npy
        state_index.json
        scc_assignments.json
        graph_summary.json
        transition_graph.png

Usage:

  python fermion_transition_graph.py --run-root \
      outputs\\precipitating_event\\20251204_004044_hilbert_quench_12_z0p02

You can also tweak edge filtering with:

  --min-edge-count 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------


@dataclass
class Microstate:
    index: int
    pattern_bits: str
    center_site: int
    count: int
    times: List[float]


@dataclass
class InternalState:
    idx: int          # compact 0..N-1
    state_id: int     # original state_id from JSON
    pattern_bits: str
    center_site: int
    count: int
    internal_change_count: int


@dataclass
class Transition:
    src_idx: int
    dst_idx: int
    count: int


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_microstates(path: str) -> List[Microstate]:
    """
    Load microstates.json robustly (expects dict with "microstates").
    """
    raw = load_json(path)

    if isinstance(raw, dict) and "microstates" in raw:
        records = raw["microstates"]
    elif isinstance(raw, list):
        # Fallback: legacy format as raw list
        records = raw
    else:
        raise ValueError(
            f"microstates.json at {path} must be a dict with 'microstates' or a list."
        )

    microstates: List[Microstate] = []

    for idx, rec in enumerate(records):
        pattern_bits = str(rec.get("pattern_bits", ""))
        center_site = int(rec.get("center_site", -1))
        times = rec.get("times", [])
        count = int(rec.get("count", len(times) if times else 1))
        microstates.append(
            Microstate(
                index=idx,
                pattern_bits=pattern_bits,
                center_site=center_site,
                count=count,
                times=list(times),
            )
        )

    return microstates


def load_internal_clusters(
    path: str,
) -> Tuple[List[InternalState], List[Transition]]:
    """
    Load internal_clusters.json in the current format:

      {
        "center_sites": {...},
        "transitions": {
          "states": [...],
          "edges": [...]
        },
        ...
      }
    """
    raw = load_json(path)

    if not isinstance(raw, dict) or "transitions" not in raw:
        raise ValueError(
            f"internal_clusters.json at {path} must be a dict with 'transitions'."
        )

    tblock = raw["transitions"]
    if not isinstance(tblock, dict) or "states" not in tblock or "edges" not in tblock:
        raise ValueError(
            f"internal_clusters.json at {path} has 'transitions' but "
            f"must contain 'states' and 'edges' inside it."
        )

    states_raw = tblock["states"]
    edges_raw = tblock["edges"]

    # Build compact index mapping state_id -> idx
    by_id: Dict[int, dict] = {}
    for rec in states_raw:
        sid = int(rec["state_id"])
        by_id[sid] = rec

    sorted_ids = sorted(by_id.keys())
    id_to_idx: Dict[int, int] = {sid: i for i, sid in enumerate(sorted_ids)}

    internal_states: List[InternalState] = []
    for i, sid in enumerate(sorted_ids):
        rec = by_id[sid]
        pattern_bits = str(rec.get("pattern_bits", ""))
        center_site = int(rec.get("center_site", -1))
        count = int(rec.get("count", 1))
        internal_change_count = int(rec.get("internal_change_count", 0))

        internal_states.append(
            InternalState(
                idx=i,
                state_id=sid,
                pattern_bits=pattern_bits,
                center_site=center_site,
                count=count,
                internal_change_count=internal_change_count,
            )
        )

    transitions: List[Transition] = []
    for rec in edges_raw:
        src_id = int(rec["from_state"])
        dst_id = int(rec["to_state"])
        count = int(rec.get("count", 1))
        if src_id not in id_to_idx or dst_id not in id_to_idx:
            # Ignore edges that reference unknown states
            continue
        transitions.append(
            Transition(
                src_idx=id_to_idx[src_id],
                dst_idx=id_to_idx[dst_id],
                count=count,
            )
        )

    return internal_states, transitions


# ---------------------------------------------------------------------
# Graph utilities (no networkx)
# ---------------------------------------------------------------------


def build_adjacency(n_states: int, transitions: List[Transition]) -> np.ndarray:
    """Weighted adjacency matrix A[i,j] = total count of transitions i->j."""
    A = np.zeros((n_states, n_states), dtype=float)
    for tr in transitions:
        if 0 <= tr.src_idx < n_states and 0 <= tr.dst_idx < n_states:
            A[tr.src_idx, tr.dst_idx] += float(tr.count)
    return A


def build_adj_list(A: np.ndarray, min_weight: float = 1.0) -> List[List[int]]:
    """Adjacency list for edges with weight >= min_weight."""
    n = A.shape[0]
    adj_list: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        row = A[i]
        js = np.where(row >= min_weight)[0]
        adj_list[i] = [int(j) for j in js]
    return adj_list


def tarjan_scc(adj_list: List[List[int]]) -> List[List[int]]:
    """
    Tarjan's algorithm for strongly connected components.
    """
    n = len(adj_list)
    index = 0
    indices = [-1] * n
    lowlink = [0] * n
    stack: List[int] = []
    on_stack = [False] * n
    sccs: List[List[int]] = []

    def strongconnect(v: int) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack[v] = True

        for w in adj_list[v]:
            if indices[w] == -1:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp: List[int] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in range(n):
        if indices[v] == -1:
            strongconnect(v)

    return sccs


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------


def plot_transition_graph(
    out_path: str,
    states: List[InternalState],
    A: np.ndarray,
    scc_assign: List[int],
    min_edge_weight: float = 1.0,
) -> None:
    """
    Simple circular layout plot of the internal-state transition graph.
    """
    n = len(states)
    if n == 0:
        return

    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    xs = np.cos(angles)
    ys = np.sin(angles)

    out_deg = A.sum(axis=1)
    in_deg = A.sum(axis=0)
    tot_deg = out_deg + in_deg
    base_size = 100.0
    sizes = base_size * (1.0 + np.log1p(tot_deg))

    scc_ids = np.array(scc_assign, dtype=int)
    max_scc = max(scc_ids) if scc_ids.size > 0 else 0
    if max_scc <= 0:
        colors = scc_ids * 0.0
    else:
        colors = scc_ids / max_scc

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw edges
    for i in range(n):
        for j in range(n):
            w = A[i, j]
            if w >= min_edge_weight:
                ax.annotate(
                    "",
                    xy=(xs[j], ys[j]),
                    xytext=(xs[i], ys[i]),
                    arrowprops=dict(
                        arrowstyle="-",
                        alpha=0.15 + 0.4 * min(1.0, w / min_edge_weight),
                        linewidth=0.5,
                    ),
                )

    # Draw nodes
    sc = ax.scatter(xs, ys, s=sizes, c=colors, cmap="viridis", alpha=0.9, edgecolors="k")

    # Label top hubs by degree
    order = np.argsort(-tot_deg)
    for rank in range(min(10, n)):
        i = int(order[rank])
        st = states[i]
        ax.text(
            xs[i] * 1.1,
            ys[i] * 1.1,
            str(st.state_id),
            ha="center",
            va="center",
            fontsize=8,
        )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("SCC index (normalized)")

    fig.suptitle("Internal-State Transition Graph (circular layout)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build internal-state transition graph "
        "from analysis_microstates outputs."
    )
    parser.add_argument(
        "--run-root",
        required=True,
        help="Precipitating Event run root directory.",
    )
    parser.add_argument(
        "--min-edge-count",
        type=int,
        default=1,
        help="Minimum transition count to keep an edge in SCC / plot.",
    )
    args = parser.parse_args()

    run_root = os.path.abspath(args.run_root)

    print("============================================================")
    print("  Fermion Internal-State Transition Graph Builder")
    print("============================================================")
    print(f"run_root: {run_root}")
    print("------------------------------------------------------------")

    micro_path = os.path.join(run_root, "analysis_microstates", "microstates.json")
    clusters_path = os.path.join(run_root, "analysis_microstates", "internal_clusters.json")

    print(f"microstates file:      {micro_path}")
    print(f"internal_clusters file:{clusters_path}")
    print("------------------------------------------------------------")

    if not os.path.isfile(micro_path):
        raise FileNotFoundError(f"microstates.json not found at {micro_path}")
    if not os.path.isfile(clusters_path):
        raise FileNotFoundError(f"internal_clusters.json not found at {clusters_path}")

    out_dir = os.path.join(run_root, "analysis_transition_graph")
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    microstates = load_microstates(micro_path)
    states, transitions = load_internal_clusters(clusters_path)

    n_micro = len(microstates)
    n_states = len(states)
    n_trans = len(transitions)

    print(f"Loaded microstates:     {n_micro}")
    print(f"Loaded internal states: {n_states}")
    print(f"Loaded transitions:     {n_trans}")
    print("------------------------------------------------------------")

    # Build adjacency
    A = build_adjacency(n_states, transitions)
    np.save(os.path.join(out_dir, "adjacency.npy"), A)

    # Degrees
    out_deg = A.sum(axis=1)
    in_deg = A.sum(axis=0)
    tot_deg = out_deg + in_deg

    # SCCs
    adj_list = build_adj_list(A, min_weight=float(args.min_edge_count))
    sccs = tarjan_scc(adj_list)
    scc_assign = [-1] * n_states
    for scc_idx, comp in enumerate(sccs):
        for v in comp:
            scc_assign[v] = scc_idx

    n_scc = len(sccs)
    largest_scc = max((len(c) for c in sccs), default=0)

    # Save state index and SCC assignments
    state_index = {
        str(st.idx): {
            "state_id": st.state_id,
            "pattern_bits": st.pattern_bits,
            "center_site": st.center_site,
            "count": st.count,
            "internal_change_count": st.internal_change_count,
        }
        for st in states
    }
    with open(os.path.join(out_dir, "state_index.json"), "w", encoding="utf-8") as f:
        json.dump(state_index, f, indent=2)

    scc_assign_out = {str(i): int(scc_assign[i]) for i in range(n_states)}
    with open(os.path.join(out_dir, "scc_assignments.json"), "w", encoding="utf-8") as f:
        json.dump(scc_assign_out, f, indent=2)

    # Top hubs
    order = np.argsort(-tot_deg)
    top_hubs = []
    for rank in range(min(10, n_states)):
        i = int(order[rank])
        st = states[i]
        top_hubs.append(
            {
                "rank": rank + 1,
                "idx": i,
                "state_id": st.state_id,
                "pattern_bits": st.pattern_bits,
                "center_site": st.center_site,
                "total_degree": float(tot_deg[i]),
                "in_degree": float(in_deg[i]),
                "out_degree": float(out_deg[i]),
                "scc_index": int(scc_assign[i]),
            }
        )

    summary = {
        "run_root": run_root,
        "n_microstates": n_micro,
        "n_internal_states": n_states,
        "n_transitions": n_trans,
        "n_scc": n_scc,
        "largest_scc_size": largest_scc,
        "min_edge_count": int(args.min_edge_count),
        "top_hubs": top_hubs,
    }
    with open(os.path.join(out_dir, "graph_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot
    png_path = os.path.join(out_dir, "transition_graph.png")
    plot_transition_graph(
        out_path=png_path,
        states=states,
        A=A,
        scc_assign=scc_assign,
        min_edge_weight=float(args.min_edge_count),
    )

    print("Analysis complete.")
    print(f"  adjacency.npy        -> {os.path.join(out_dir, 'adjacency.npy')}")
    print(f"  state_index.json     -> {os.path.join(out_dir, 'state_index.json')}")
    print(f"  scc_assignments.json -> {os.path.join(out_dir, 'scc_assignments.json')}")
    print(f"  graph_summary.json   -> {os.path.join(out_dir, 'graph_summary.json')}")
    print(f"  transition_graph.png -> {png_path}")
    print("============================================================")


if __name__ == "__main__":
    main()
