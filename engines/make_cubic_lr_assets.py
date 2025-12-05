#!/usr/bin/env python3
"""
make_cubic_lr_assets.py
=======================

Utility to create "fake LR assets" for a 3D cubic lattice universe,
so we can run precipitating_event.py on a bigger emergent geometry
without paying the full substrate Hilbert cost.

What it does:

  - Build a 3D grid of nodes with size Lx x Ly x Lz.
  - Use 6-neighbor adjacency (±x, ±y, ±z) with either open or periodic
    boundaries.
  - Compute graph_dist[i,j] = shortest-path distance on that graph.
  - Set D_lr = graph_dist (a simple v=1 lightcone metric stand-in).
  - Set D_prop = graph_dist as well (for compatibility).
  - Define coordinates X[i] = (x,y,z) as floats.

Outputs in asset_dir:

  - lr_metrics.npz with:
      D_lr, D_prop, graph_dist
  - lr_embedding_3d.npz with:
      X   (shape (N,3))

This lets precipitating_event.py and smooth_lr_geometry.py treat this
as if it were an LR-derived emergent spacetime patch.
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np


@dataclass
class CubicConfig:
    asset_dir: str = "cubic_universe_L3"
    Lx: int = 3
    Ly: int = 3
    Lz: int = 3
    periodic: bool = False  # if True, wrap in all directions


def grid_index(x: int, y: int, z: int, Lx: int, Ly: int, Lz: int) -> int:
    return (z * Ly + y) * Lx + x


def grid_coords(i: int, Lx: int, Ly: int, Lz: int) -> Tuple[int, int, int]:
    x = i % Lx
    y = (i // Lx) % Ly
    z = i // (Lx * Ly)
    return x, y, z


def neighbors_3d(
    x: int, y: int, z: int,
    Lx: int, Ly: int, Lz: int,
    periodic: bool,
):
    """Yield neighbor coordinates with 6-connectivity."""
    deltas = [
        (+1, 0, 0),
        (-1, 0, 0),
        (0, +1, 0),
        (0, -1, 0),
        (0, 0, +1),
        (0, 0, -1),
    ]
    for dx, dy, dz in deltas:
        nx, ny, nz = x + dx, y + dy, z + dz
        if periodic:
            nx %= Lx
            ny %= Ly
            nz %= Lz
            yield nx, ny, nz
        else:
            if 0 <= nx < Lx and 0 <= ny < Ly and 0 <= nz < Lz:
                yield nx, ny, nz


def build_graph_and_distances(cfg: CubicConfig):
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    N = Lx * Ly * Lz

    # adjacency list
    adjacency = [[] for _ in range(N)]

    for i in range(N):
        x, y, z = grid_coords(i, Lx, Ly, Lz)
        for nx, ny, nz in neighbors_3d(x, y, z, Lx, Ly, Lz, cfg.periodic):
            j = grid_index(nx, ny, nz, Lx, Ly, Lz)
            adjacency[i].append(j)

    # compute all-pairs shortest path distances via BFS from each node
    graph_dist = np.full((N, N), np.inf, dtype=float)
    for i in range(N):
        dist = np.full(N, np.inf, dtype=float)
        dist[i] = 0.0
        q = deque([i])
        while q:
            u = q.popleft()
            for v in adjacency[u]:
                if dist[v] == np.inf:
                    dist[v] = dist[u] + 1.0
                    q.append(v)
        graph_dist[i, :] = dist

    # symmetric sanity
    graph_dist = np.minimum(graph_dist, graph_dist.T)

    return adjacency, graph_dist


def run(cfg: CubicConfig) -> None:
    print("Cubic LR asset config:")
    print(asdict(cfg))
    print()

    adjacency, graph_dist = build_graph_and_distances(cfg)
    N = graph_dist.shape[0]
    print(f"Built 3D grid with N={N} nodes.")
    print("Example adjacency (first few nodes):")
    max_print = min(N, 8)
    for i in range(max_print):
        print(f"  {i}: {adjacency[i]}")
    if N > max_print:
        print(f"  ... ({N - max_print} more nodes)")
    print()

    # Define D_lr and D_prop as just graph_dist (toy v=1 metric)
    D_lr = graph_dist.copy()
    D_prop = graph_dist.copy()

    # Define coordinates X
    X = np.zeros((N, 3), dtype=float)
    for i in range(N):
        x, y, z = grid_coords(i, cfg.Lx, cfg.Ly, cfg.Lz)
        X[i, :] = (float(x), float(y), float(z))

    # Save to asset_dir
    os.makedirs(cfg.asset_dir, exist_ok=True)

    metrics_path = os.path.join(cfg.asset_dir, "lr_metrics.npz")
    np.savez(metrics_path, D_lr=D_lr, D_prop=D_prop, graph_dist=graph_dist)
    print(f"Saved lr_metrics.npz to {metrics_path}")

    embed_path = os.path.join(cfg.asset_dir, "lr_embedding_3d.npz")
    np.savez(embed_path, X=X)
    print(f"Saved lr_embedding_3d.npz to {embed_path}")

    print("\nDone. You can now run smooth_lr_geometry.py and precipitating_event.py")
    print("on this asset_dir as if it came from substrate.py.")


def parse_args(argv: Optional[list[str]] = None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Create cubic 3D LR-like assets for precipitating_event."
    )
    parser.add_argument(
        "--asset-dir",
        type=str,
        default="cubic_universe_L3",
        help="Output directory for lr_metrics.npz and lr_embedding_3d.npz",
    )
    parser.add_argument(
        "--Lx", type=int, default=3, help="Grid size in x"
    )
    parser.add_argument(
        "--Ly", type=int, default=3, help="Grid size in y"
    )
    parser.add_argument(
        "--Lz", type=int, default=3, help="Grid size in z"
    )
    parser.add_argument(
        "--periodic",
        action="store_true",
        help="Use periodic boundaries in all directions.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    cfg = CubicConfig(
        asset_dir=args.asset_dir,
        Lx=args.Lx,
        Ly=args.Ly,
        Lz=args.Lz,
        periodic=args.periodic,
    )
    run(cfg)


if __name__ == "__main__":
    main()
