"""
geometry_upscale_64.py

Upscale an 8-site emergent geometry (cube-like) to a 64-site grid.

Inputs:
  - base geometry npz (e.g. lr_embedding_3d.npz) with:
      * graph_dist : (8, 8) shortest-path distances
      * coords     : (8, 3) optional, node positions in 3D

Idea:
  - Interpret the 8 coords as corners of a cube-like region in 3D.
  - Build a 4 x 4 x 4 regular grid inside the same bounding box.
  - Connect grid points to their nearest neighbors (within ~one lattice spacing).
  - Compute new graph_dist (64 x 64) by BFS on this adjacency.
  - Save new npz with:
      * graph_dist : (64, 64)
      * coords     : (64, 3)

This gives you a "super-resolved" emergent geometry that preserves the
large-scale shape of the original cube, but with more sites for richer dynamics.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple, List, Dict

import numpy as np


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def load_base_geometry(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load base geometry npz.
    Returns:
      graph_dist: (N, N)
      coords: (N, 3)
    If coords are missing, assumes an 8-node cube.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Base geometry not found: {path}")

    npz = np.load(path)
    if "graph_dist" not in npz.files:
        raise ValueError("Base geometry npz must contain 'graph_dist'.")

    graph_dist = np.array(npz["graph_dist"], dtype=float)
    n_sites = graph_dist.shape[0]

    if "coords" in npz.files:
        coords = np.array(npz["coords"], dtype=float)
        if coords.shape != (n_sites, 3):
            raise ValueError(
                f"'coords' has shape {coords.shape}, expected ({n_sites}, 3)."
            )
    else:
        # Fallback: assume it's the 8-node cube and place them at corners.
        if n_sites != 8:
            raise ValueError(
                "Base geometry has no 'coords' and n_sites != 8; "
                "cannot infer cube corners."
            )
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )

    return graph_dist, coords


def build_4x4x4_grid_from_bbox(coords_8: np.ndarray) -> np.ndarray:
    """
    Given 8 base coords, build a 4x4x4 grid of coords inside the same bounding box.
    Returns:
      coords_64: (64, 3)
    """
    mins = coords_8.min(axis=0)
    maxs = coords_8.max(axis=0)

    # 4 points along each axis
    xs = np.linspace(mins[0], maxs[0], 4)
    ys = np.linspace(mins[1], maxs[1], 4)
    zs = np.linspace(mins[2], maxs[2], 4)

    pts = []
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                pts.append([xs[ix], ys[iy], zs[iz]])

    coords_64 = np.array(pts, dtype=float)
    return coords_64


def build_adjacency_from_grid(coords: np.ndarray) -> np.ndarray:
    """
    Given coords for a regular 4x4x4 grid, build adjacency matrix by connecting
    nearest neighbors (within ~1 grid spacing).
    """
    n = coords.shape[0]
    # Estimate lattice spacing from unique coordinate differences
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if d > 0:
                diffs.append(d)
    diffs = np.array(diffs, dtype=float)
    # Characteristic neighbor distance is the smallest nonzero distance
    d_min = float(np.min(diffs))
    # Threshold a bit above that to catch grid neighbors
    thresh = 1.01 * d_min

    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if d <= thresh + 1e-12:
                adj[i, j] = 1
                adj[j, i] = 1

    # Make sure the graph is connected (it should be for a 4x4x4 grid)
    return adj


def bfs_distances(adj: np.ndarray, start: int) -> np.ndarray:
    """
    BFS to compute shortest-path distances from 'start' to all other nodes.
    """
    n = adj.shape[0]
    dist = np.full(n, np.inf, dtype=float)
    dist[start] = 0.0
    queue: List[int] = [start]
    head = 0
    while head < len(queue):
        i = queue[head]
        head += 1
        for j in range(n):
            if adj[i, j] and dist[j] == np.inf:
                dist[j] = dist[i] + 1.0
                queue.append(j)
    return dist


def compute_graph_dist_from_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    Compute all-pairs shortest path distances by BFS from each node.
    """
    n = adj.shape[0]
    gd = np.zeros((n, n), dtype=float)
    for i in range(n):
        gd[i, :] = bfs_distances(adj, i)
    return gd


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Upscale an 8-site emergent geometry to a 64-site 3D grid geometry."
        )
    )
    p.add_argument(
        "--base-geometry",
        type=str,
        default="lr_embedding_3d.npz",
        help="Path to base 8-site geometry npz (must contain graph_dist).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="lr_embedding_3d_64.npz",
        help="Output path for 64-site geometry npz.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_path = args.base_geometry
    out_path = args.output

    print("============================================")
    print("  Geometry Upscale: 8-site -> 64-site grid")
    print("============================================")
    print(f"Base geometry: {base_path}")
    print(f"Output:        {out_path}")
    print("--------------------------------------------")

    graph_dist_8, coords_8 = load_base_geometry(base_path)
    n_base = graph_dist_8.shape[0]
    if n_base != 8:
        raise ValueError(
            f"Base geometry has n_sites={n_base}, but this script assumes 8."
        )

    print(f"Base n_sites: {n_base}")
    print("Building 4x4x4 grid...")
    coords_64 = build_4x4x4_grid_from_bbox(coords_8)

    print("Building adjacency...")
    adj_64 = build_adjacency_from_grid(coords_64)
    avg_deg = float(adj_64.sum(axis=1).mean())
    print(f"64-site graph average degree: {avg_deg:.3f}")

    print("Computing graph distances (BFS from each node)...")
    graph_dist_64 = compute_graph_dist_from_adjacency(adj_64)

    # Sanity check: connectivity (no inf distances)
    if not np.isfinite(graph_dist_64).all():
        raise RuntimeError("Upscaled graph is not fully connected.")

    print("Saving new geometry npz...")
    # You can add more fields here if desired
    np.savez_compressed(
        out_path,
        graph_dist=graph_dist_64,
        coords=coords_64,
    )

    print("Done.")
    print("You can now use this as --geometry for precipitating_event.py.")


if __name__ == "__main__":
    main()
