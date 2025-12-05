"""
find_two_lump_frames.py

Inspect a precipitating_event_snap run and report time frames where:
  - There are exactly two lumps
  - Each lump has size >= --min-lump-size
  - Their centers are at least --min-center-distance apart (graph distance)

Also cross-references with snapshots.npz (if present) to tell you
which of those good frames you actually have full wavefunction snapshots for.

Usage example:

  python find_two_lump_frames.py ^
    --run-root outputs\\precipitating_event_snap\\20251204_...._hilbert_quench_12_snap ^
    --geometry lr_embedding_3d_12.npz ^
    --min-lump-size 2 ^
    --min-center-distance 2
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Tuple

import numpy as np


def load_geometry(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    keys = set(data.keys())
    # coords
    if "coords" in keys:
        coords = data["coords"]
    elif "X" in keys:
        coords = data["X"]
    else:
        raise ValueError(f"Geometry NPZ must contain 'coords' or 'X': {path}")

    # edges or graph_dist
    if "edges" in keys:
        edges = data["edges"].astype(int)
        graph_dist = None
    elif "graph_dist" in keys:
        gd = data["graph_dist"]
        if gd.ndim != 2 or gd.shape[0] != gd.shape[1]:
            raise ValueError("graph_dist must be a square matrix.")
        n = gd.shape[0]
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(gd[i, j] - 1.0) < 1e-8:
                    edge_list.append((i, j))
        edges = np.array(edge_list, dtype=int)
        graph_dist = gd
    else:
        raise ValueError(f"Geometry NPZ must contain 'edges' or 'graph_dist': {path}")

    return {"coords": coords, "edges": edges, "graph_dist": graph_dist}


def bfs_graph_dist(n_sites: int, edges: np.ndarray) -> np.ndarray:
    """
    If there is no graph_dist in the geometry file, build one by BFS.
    dist[i,j] = shortest number of edges between i and j.
    """
    adj = [[] for _ in range(n_sites)]
    for i, j in edges:
        i = int(i)
        j = int(j)
        adj[i].append(j)
        adj[j].append(i)

    dist = np.full((n_sites, n_sites), np.inf, dtype=float)
    for src in range(n_sites):
        dist[src, src] = 0.0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            du = dist[src, u]
            for v in adj[u]:
                if dist[src, v] > du + 1.0:
                    dist[src, v] = du + 1.0
                    queue.append(v)
    return dist


def load_lumps(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_snapshots_indices(path: str) -> List[int]:
    if not os.path.exists(path):
        return []
    data = np.load(path)
    idx = data["indices"]
    return [int(i) for i in idx]


def center_of_lump(lump: List[int], coords: np.ndarray) -> int:
    """
    Define the "center" of a lump as the site in the lump whose
    coords are closest to the mean coordinates of the lump.
    """
    if len(lump) == 1:
        return int(lump[0])
    pts = coords[lump]  # shape (len(lump), 3)
    mean = pts.mean(axis=0)
    d2 = np.sum((pts - mean) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return int(lump[idx])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find good two-lump frames in a precipitating_event_snap run."
    )
    ap.add_argument(
        "--run-root",
        required=True,
        help="Run root for precipitating_event_snap (folder with params.json, lumps.json, snapshots.npz).",
    )
    ap.add_argument(
        "--geometry",
        required=True,
        help="Geometry NPZ used for the run (e.g., lr_embedding_3d_12.npz).",
    )
    ap.add_argument(
        "--min-lump-size",
        type=int,
        default=2,
        help="Minimum size (number of sites) for each lump.",
    )
    ap.add_argument(
        "--min-center-distance",
        type=int,
        default=2,
        help="Minimum graph distance between lump centers.",
    )
    args = ap.parse_args()

    run_root = os.path.abspath(args.run_root)
    geom_path = os.path.abspath(args.geometry)

    print("============================================================")
    print("  Two-Lump Frame Finder")
    print("============================================================")
    print(f"run_root:  {run_root}")
    print(f"geometry:  {geom_path}")
    print("------------------------------------------------------------")

    # Load geometry
    g = load_geometry(geom_path)
    coords = g["coords"]
    edges = g["edges"]
    graph_dist = g["graph_dist"]

    n_sites = coords.shape[0]
    print(f"n_sites: {n_sites}, n_edges: {edges.shape[0]}")

    if graph_dist is None:
        print("No graph_dist in geometry; building BFS-based distances...")
        graph_dist = bfs_graph_dist(n_sites, edges)
    else:
        print("Using graph_dist from geometry file.")

    # Load lumps
    lumps_path = os.path.join(run_root, "lumps.json")
    if not os.path.exists(lumps_path):
        raise FileNotFoundError(f"lumps.json not found in {run_root}")
    lumps_data = load_lumps(lumps_path)
    times = lumps_data["times"]
    lump_counts = lumps_data["lump_counts"]
    lump_memberships = lumps_data["lump_memberships"]

    n_steps = len(times)
    print(f"n_steps in lumps.json: {n_steps}")
    print("------------------------------------------------------------")

    # Load snapshot indices (if any)
    snapshots_path = os.path.join(run_root, "snapshots.npz")
    snap_indices = load_snapshots_indices(snapshots_path)
    snap_index_set = set(snap_indices)
    if snap_indices:
        print(f"Found snapshots.npz with indices: {snap_indices}")
    else:
        print("No snapshots.npz found or no indices recorded.")
    print("------------------------------------------------------------")

    # Scan for good frames
    good_frames: List[Dict] = []
    min_size = int(args.min_lump_size)
    min_dist = float(args.min_center_distance)

    for step_idx in range(n_steps):
        c = int(lump_counts[step_idx])
        if c != 2:
            continue
        lumps = lump_memberships[step_idx]
        if len(lumps) != 2:
            continue

        L0 = [int(v) for v in lumps[0]]
        L1 = [int(v) for v in lumps[1]]
        if len(L0) < min_size or len(L1) < min_size:
            continue

        c0 = center_of_lump(L0, coords)
        c1 = center_of_lump(L1, coords)
        d01 = graph_dist[c0, c1]

        if not np.isfinite(d01) or d01 < min_dist:
            continue

        good_frames.append(
            {
                "step": step_idx,
                "time": float(times[step_idx]),
                "lump_sizes": (len(L0), len(L1)),
                "centers": (c0, c1),
                "center_distance": float(d01),
                "has_snapshot": (step_idx in snap_index_set),
            }
        )

    # Report
    if not good_frames:
        print("No frames matched the two-lump criteria.")
        return

    print("Good two-lump frames:")
    print("step  time     sizes   centers   dist  snapshot?")
    print("----- -------- ------- --------- ----- ---------")
    for rec in good_frames:
        step = rec["step"]
        t = rec["time"]
        s0, s1 = rec["lump_sizes"]
        c0, c1 = rec["centers"]
        d = rec["center_distance"]
        has_snap = rec["has_snapshot"]
        print(
            f"{step:4d}  {t:7.3f}   ({s0},{s1})  ({c0},{c1})  {d:4.1f}   {'YES' if has_snap else 'no'}"
        )

    print("------------------------------------------------------------")
    print(f"Total good frames: {len(good_frames)}")
    have_snap = [g for g in good_frames if g["has_snapshot"]]
    if have_snap:
        print("Frames WITH snapshots:")
        for rec in have_snap:
            print(
                f"  step {rec['step']} at t={rec['time']:.3f}, sizes={rec['lump_sizes']}, centers={rec['centers']}"
            )
    else:
        print("No good frames coincide with saved snapshots; you may want to re-run with appropriate --snapshot-indices.")


if __name__ == "__main__":
    main()
