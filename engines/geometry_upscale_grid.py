"""
geometry_upscale_grid.py

General upscaler from a small emergent geometry to a regular 3D grid.

Inputs:
  - Base geometry npz (e.g. lr_embedding_3d.npz) with:
      * graph_dist : (N, N) shortest-path distances
      * coords     : (N, 3) optional, node positions in 3D

Idea:
  - Interpret the base coords as sampling a 3D region (usually a cube-ish shape).
  - Build an Nx x Ny x Nz regular grid inside the same bounding box.
  - Connect grid points as a regular Manhattan grid:
        |Δix| + |Δiy| + |Δiz| == 1
    where (ix,iy,iz) are integer grid indices.
  - Compute new graph_dist (M x M) for M = Nx*Ny*Nz by BFS.
  - Save new npz with:
      * graph_dist : (M, M)
      * coords     : (M, 3)

Note:
  This script only constructs geometry. The quantum engine (precipitating_event.py)
  still scales as 2^M in Hilbert dimension, so for that engine you probably want
  M <= ~12 (e.g. 3x2x2, 2x2x3, etc.).
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Base geometry loading
# ---------------------------------------------------------------------


def load_base_geometry(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load base geometry npz.
    Returns:
      graph_dist: (N, N)
      coords: (N, 3)

    If coords are missing, assumes an 8-node cube and places them
    at the cube corners.
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


# ---------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------


def build_grid_from_bbox(
    coords_base: np.ndarray, Nx: int, Ny: int, Nz: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given base coords, build an Nx x Ny x Nz regular grid of coords
    inside the same bounding box.

    Returns:
      coords_grid: (M, 3) with M = Nx * Ny * Nz
      indices:     (M, 3) integer grid indices (ix, iy, iz) per site

    The ordering of points is:
      for ix in [0..Nx-1]:
        for iy in [0..Ny-1]:
          for iz in [0..Nz-1]:
              append (ix,iy,iz)
    """
    mins = coords_base.min(axis=0)
    maxs = coords_base.max(axis=0)

    # Avoid zero-length axes: if max == min, thicken by a tiny epsilon
    span = maxs - mins
    eps = 1e-6
    span = np.where(span < eps, eps, span)
    maxs = mins + span

    xs = np.linspace(mins[0], maxs[0], Nx)
    ys = np.linspace(mins[1], maxs[1], Ny)
    zs = np.linspace(mins[2], maxs[2], Nz)

    pts = []
    idxs = []
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                pts.append([xs[ix], ys[iy], zs[iz]])
                idxs.append([ix, iy, iz])

    coords_grid = np.array(pts, dtype=float)
    indices = np.array(idxs, dtype=int)
    return coords_grid, indices


def build_adjacency_manhattan(
    indices: np.ndarray, Nx: int, Ny: int, Nz: int
) -> np.ndarray:
    """
    Build adjacency for a regular Nx x Ny x Nz grid, connecting nearest
    neighbors in the Manhattan sense:
        |Δix| + |Δiy| + |Δiz| == 1
    """
    M = indices.shape[0]
    # Map (ix,iy,iz) -> linear index
    index_map = {}
    for k in range(M):
        ix, iy, iz = indices[k]
        index_map[(ix, iy, iz)] = k

    adj = np.zeros((M, M), dtype=int)

    # For each site, connect to neighbors along +x, +y, +z if they exist
    for k in range(M):
        ix, iy, iz = indices[k]
        # +x neighbor
        if ix + 1 < Nx:
            j = index_map[(ix + 1, iy, iz)]
            adj[k, j] = 1
            adj[j, k] = 1
        # +y neighbor
        if iy + 1 < Ny:
            j = index_map[(ix, iy + 1, iz)]
            adj[k, j] = 1
            adj[j, k] = 1
        # +z neighbor
        if iz + 1 < Nz:
            j = index_map[(ix, iy, iz + 1)]
            adj[k, j] = 1
            adj[j, k] = 1

    return adj


# ---------------------------------------------------------------------
# Graph distances via BFS
# ---------------------------------------------------------------------


def bfs_distances(adj: np.ndarray, start: int) -> np.ndarray:
    """
    BFS to compute shortest-path distances from 'start' to all other nodes
    in an unweighted graph.
    """
    n = adj.shape[0]
    dist = np.full(n, np.inf, dtype=float)
    dist[start] = 0.0
    queue: List[int] = [start]
    head = 0

    while head < len(queue):
        i = queue[head]
        head += 1
        neighbors = np.nonzero(adj[i])[0]
        for j in neighbors:
            if dist[j] == np.inf:
                dist[j] = dist[i] + 1.0
                queue.append(int(j))

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
            "Upscale a small emergent geometry to an Nx x Ny x Nz 3D grid geometry."
        )
    )
    p.add_argument(
        "--base-geometry",
        type=str,
        default="lr_embedding_3d.npz",
        help="Path to base geometry npz (must contain 'graph_dist').",
    )
    p.add_argument(
        "--Nx",
        type=int,
        default=3,
        help="Number of grid points along x (default: 3).",
    )
    p.add_argument(
        "--Ny",
        type=int,
        default=2,
        help="Number of grid points along y (default: 2).",
    )
    p.add_argument(
        "--Nz",
        type=int,
        default=2,
        help="Number of grid points along z (default: 2).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="lr_embedding_3d_grid.npz",
        help="Output path for upscaled geometry npz.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    base_path = args.base_geometry
    out_path = args.output
    Nx = int(args.Nx)
    Ny = int(args.Ny)
    Nz = int(args.Nz)

    if Nx <= 0 or Ny <= 0 or Nz <= 0:
        raise ValueError("Nx, Ny, Nz must all be positive integers.")

    M = Nx * Ny * Nz

    print("============================================")
    print("  Geometry Upscale: small -> 3D grid")
    print("============================================")
    print(f"Base geometry: {base_path}")
    print(f"Output:        {out_path}")
    print(f"Grid:          {Nx} x {Ny} x {Nz} = {M} sites")
    print("--------------------------------------------")

    base_gd, base_coords = load_base_geometry(base_path)
    n_base = base_gd.shape[0]
    print(f"Base n_sites:  {n_base}")

    print("Building grid coordinates from bounding box...")
    coords_grid, indices = build_grid_from_bbox(base_coords, Nx=Nx, Ny=Ny, Nz=Nz)

    print("Building Manhattan adjacency...")
    adj = build_adjacency_manhattan(indices, Nx=Nx, Ny=Ny, Nz=Nz)
    degrees = adj.sum(axis=1)
    avg_deg = float(degrees.mean())
    print(f"  Average degree: {avg_deg:.3f}")
    print(f"  Min degree:     {int(degrees.min())}")
    print(f"  Max degree:     {int(degrees.max())}")

    print("Computing graph distances via BFS...")
    graph_dist = compute_graph_dist_from_adjacency(adj)

    # Sanity: check connectivity
    if not np.isfinite(graph_dist).all():
        raise RuntimeError(
            "Upscaled graph is not fully connected (infinite distances detected)."
        )

    print("Saving new geometry npz...")
    np.savez_compressed(
        out_path,
        graph_dist=graph_dist,
        coords=coords_grid,
    )

    print("Done.")
    print("You can now use this npz as --geometry for precipitating_event.py.")
    print("Reminder: precipitating_event.py Hilbert dimension scales as 2^(Nx*Ny*Nz),")
    print("so keep Nx*Ny*Nz modest (<= ~12) for that engine.")


if __name__ == "__main__":
    main()
