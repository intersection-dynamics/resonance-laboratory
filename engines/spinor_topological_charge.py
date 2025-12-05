#!/usr/bin/env python3
"""
spinor_topological_charge.py
============================

Compute a discrete skyrmion-like topological charge for a spinor
precipitating_event run.

This script *does not* re-run the simulation. It analyzes the
spinor_particle_features.npz produced by spinor_particle_analysis.py.

Geometry assumption
-------------------

We assume the LR embedding X forms a *regular 3D grid*, e.g. a 3x3x3 cube:

    X[i] ≈ (ix, iy, iz)  with ix,iy,iz ∈ {0,1,...,L-1}

We reconstruct integer grid indices by rounding the coordinates:

    ix = round(X[i,0]), iy = round(X[i,1]), iz = round(X[i,2])

Then we organize the local spin vectors s_vec[i] into a lattice:

    s[ix, iy, iz] ∈ R^3, |s|≈1

Topological charge (2D skyrmion proxy)
--------------------------------------

For each fixed z = const plane (xy-slice), we treat s(ix,iy,iz) as a 2D spin
texture n(x,y) on S^2 and compute a *discrete skyrmion number*:

    Q_z ≈ (1 / 4π) * Σ_plaquettes Ω_plaquette

where each plaquette (x,y) → (x+1,y) → (x+1,y+1) → (x,y+1) is split into
two triangles:

    (a,b,c) and (a,c,d)

with a,b,c,d unit vectors in R^3, and each triangle contributes a solid angle

    Ω(a,b,c) = 2 * atan2( a · (b × c),
                          1 + a·b + b·c + c·a )

We sum Ω over all triangles in the slice, then divide by 4π.

We repeat this for all z-slices (and optionally also for x-slices / y-slices
if you want to extend it later). For this prototype we focus on z-slices as
a simple, interpretable "matter vs antimatter" proxy:

  - Q_z > 0  → "matter-like chirality" in that slice
  - Q_z < 0  → "antimatter-like chirality" in that slice

In a bigger universe or with more structure, you can sum over slices or
look for clumps in Q_z(x,y) to identify localized particle-like objects.

Outputs
-------

Given a run directory containing spinor_particle_features.npz, we produce:

  - spinor_topology_summary.txt
      * grid shape
      * per-slice Q_z values
      * total Q_xy = Σ_z Q_z
  - spinor_topology_features.npz
      * Q_z      : (Lz,) array of per-slice charges
      * Q_total  : scalar total Q_xy
      * Lx, Ly, Lz
      * grid_ok  : bool indicating whether grid reconstruction was clean

Usage
-----

  python spinor_topological_charge.py --run-dir PATH\TO\RUN

where PATH\TO\RUN contains spinor_particle_features.npz.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Tuple

import numpy as np


# ======================================================================
# Utilities
# ======================================================================


def load_features(run_dir: str) -> Dict[str, Any]:
    """
    Load spinor_particle_features.npz from the run directory.
    """
    path = os.path.join(run_dir, "spinor_particle_features.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")

    data = np.load(path, allow_pickle=True)
    required = ["X", "s_vec"]
    for key in required:
        if key not in data:
            raise KeyError(f"{path} missing required key '{key}'")

    X = data["X"]          # (N,3)
    s_vec = data["s_vec"]  # (N,3)

    s_norm = data["s_norm"] if "s_norm" in data else None
    times = data["times"] if "times" in data else None
    Sx_hist = data["Sx_hist"] if "Sx_hist" in data else None
    Sy_hist = data["Sy_hist"] if "Sy_hist" in data else None
    Sz_hist = data["Sz_hist"] if "Sz_hist" in data else None
    config = data["config"].item() if "config" in data else {}

    return {
        "X": X,
        "s_vec": s_vec,
        "s_norm": s_norm,
        "times": times,
        "Sx_hist": Sx_hist,
        "Sy_hist": Sy_hist,
        "Sz_hist": Sz_hist,
        "config": config,
    }


def reconstruct_grid(
    X: np.ndarray,
    s_vec: np.ndarray,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, Tuple[int, int, int], bool]:
    """
    Reconstruct a 3D regular grid from coordinates X and attach spins.

    We assume X[i] ≈ (ix,iy,iz) with integer coords, up to small noise.

    Steps:
      - round coords to nearest integers,
      - determine (Lx,Ly,Lz) from max indices + 1,
      - verify that all sites are unique and the grid is full,
      - create s_grid[Lx,Ly,Lz,3] with s_grid[ix,iy,iz] = s_vec[i].

    Returns:
      s_grid : (Lx,Ly,Lz,3) float array
      (Lx,Ly,Lz) : grid shape
      grid_ok : bool indicating whether the mapping was clean.
    """
    if X.shape[0] != s_vec.shape[0]:
        raise ValueError("X and s_vec must have same first dimension")

    N = X.shape[0]

    # Round to nearest integers
    X_int = np.rint(X).astype(int)  # (N,3)
    xs, ys, zs = X_int[:, 0], X_int[:, 1], X_int[:, 2]

    Lx = xs.max() + 1
    Ly = ys.max() + 1
    Lz = zs.max() + 1

    s_grid = np.zeros((Lx, Ly, Lz, 3), dtype=float)
    filled = np.zeros((Lx, Ly, Lz), dtype=bool)

    grid_ok = True

    for i in range(N):
        ix, iy, iz = xs[i], ys[i], zs[i]
        if filled[ix, iy, iz]:
            # Duplicate mapping; this indicates embedding isn't a clean grid.
            grid_ok = False
        s_grid[ix, iy, iz, :] = s_vec[i]
        filled[ix, iy, iz] = True

    # Check if the grid is fully populated
    if not np.all(filled):
        grid_ok = False

    return s_grid, (Lx, Ly, Lz), grid_ok


# ======================================================================
# Skyrmion charge on 2D slices
# ======================================================================


def solid_angle_triangle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the signed solid angle Ω(a,b,c) of the spherical triangle
    formed by three unit vectors a,b,c ∈ R^3.

    Formula (Berg & Lüscher 1981):

        Ω = 2 * atan2( a · (b × c),
                       1 + a·b + b·c + c·a )

    Returns:
      Ω ∈ (-2π, 2π). Units: radians.
    """
    # Ensure unit vectors
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    c = c / max(np.linalg.norm(c), 1e-12)

    num = np.dot(a, np.cross(b, c))
    den = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)

    return 2.0 * np.arctan2(num, den)


def skyrmion_number_2d_slice(n: np.ndarray) -> float:
    """
    Compute a discrete skyrmion number for a 2D spin field n[x,y,:].

    n has shape (Lx, Ly, 3) with |n[x,y]| ≈ 1.

    We sum solid angles over all plaquettes:

      For plaquette with corners:
        (x,y)   -> a
        (x+1,y) -> b
        (x+1,y+1) -> c
        (x,y+1)   -> d

      We split into two triangles: (a,b,c) and (a,c,d)

      Ω_plaq = Ω(a,b,c) + Ω(a,c,d)

    Then:

      Q ≈ (1 / 4π) * Σ_plaquettes Ω_plaq
    """
    Lx, Ly, _ = n.shape
    Q = 0.0

    for x in range(Lx - 1):
        for y in range(Ly - 1):
            a = n[x, y]
            b = n[x + 1, y]
            c = n[x + 1, y + 1]
            d = n[x, y + 1]

            omega1 = solid_angle_triangle(a, b, c)
            omega2 = solid_angle_triangle(a, c, d)
            Q += omega1 + omega2

    Q /= (4.0 * np.pi)
    return float(Q)


def compute_slice_charges(s_grid: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Compute skyrmion charges Q_z for each z-plane of the 3D grid.

    s_grid: (Lx,Ly,Lz,3)
    shape: (Lx,Ly,Lz)

    Returns:
      Q_z : (Lz,) array of skyrmion numbers for each z.
    """
    Lx, Ly, Lz = shape
    Q_z = np.zeros(Lz, dtype=float)

    for iz in range(Lz):
        n_slice = s_grid[:, :, iz, :]  # (Lx,Ly,3)
        Q_z[iz] = skyrmion_number_2d_slice(n_slice)

    return Q_z


# ======================================================================
# Output
# ======================================================================


def save_topology_features(
    run_dir: str,
    Q_z: np.ndarray,
    shape: Tuple[int, int, int],
    grid_ok: bool,
) -> str:
    """
    Save topological data to spinor_topology_features.npz in run_dir.
    """
    Lx, Ly, Lz = shape
    Q_total = float(np.sum(Q_z))
    path = os.path.join(run_dir, "spinor_topology_features.npz")
    np.savez(
        path,
        Q_z=Q_z,
        Q_total=Q_total,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        grid_ok=grid_ok,
    )
    return path


def write_topology_summary(
    run_dir: str,
    Q_z: np.ndarray,
    shape: Tuple[int, int, int],
    grid_ok: bool,
    config: Dict[str, Any],
) -> str:
    """
    Write a human-readable summary of the topological charges.
    """
    Lx, Ly, Lz = shape
    Q_total = float(np.sum(Q_z))

    path = os.path.join(run_dir, "spinor_topology_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Spinor Topological Charge Analysis\n")
        f.write("==================================\n\n")
        f.write("Config (from spinor_particle_features):\n")
        f.write(repr(config) + "\n\n")
        f.write(f"Grid shape: Lx={Lx}, Ly={Ly}, Lz={Lz}\n")
        f.write(f"Grid reconstruction OK? {grid_ok}\n\n")

        f.write("Per-slice skyrmion numbers Q_z (z-slices, xy planes):\n")
        for iz, Q in enumerate(Q_z):
            f.write(f"  z={iz}: Q_z = {Q:+.6f}\n")
        f.write("\n")

        f.write(f"Total Q_xy (sum over z slices) = {Q_total:+.6f}\n")
        f.write("\n")

        if not grid_ok:
            f.write(
                "WARNING: grid reconstruction was not fully clean "
                "(missing or duplicate sites). Interpret Q with caution.\n"
            )

    return path


# ======================================================================
# Main
# ======================================================================


def analyze_topology(run_dir: str) -> None:
    print(f"Computing spinor topological charge in: {run_dir}")
    data = load_features(run_dir)
    X = data["X"]
    s_vec = data["s_vec"]
    config = data["config"]

    # Reconstruct regular grid
    s_grid, shape, grid_ok = reconstruct_grid(X, s_vec)
    Lx, Ly, Lz = shape
    print(f"  Reconstructed grid shape: Lx={Lx}, Ly={Ly}, Lz={Lz}")
    print(f"  Grid reconstruction OK? {grid_ok}")

    # Normalize spins to unit length (just to be safe)
    norm = np.linalg.norm(s_grid, axis=3, keepdims=True)
    norm[norm == 0.0] = 1.0
    n_grid = s_grid / norm

    # Compute Q_z for each z-slice
    Q_z = compute_slice_charges(n_grid, shape)
    Q_total = float(np.sum(Q_z))
    print("  Per-slice Q_z (xy slices at fixed z):")
    for iz, Q in enumerate(Q_z):
        print(f"    z={iz}: Q_z = {Q:+.6f}")
    print(f"  Total Q_xy (sum over z slices) = {Q_total:+.6f}")

    # Save outputs
    features_path = save_topology_features(run_dir, Q_z, shape, grid_ok)
    summary_path = write_topology_summary(run_dir, Q_z, shape, grid_ok, config)

    print("Topological analysis complete.")
    print(f"  Saved features: {features_path}")
    print(f"  Saved summary : {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a discrete skyrmion-like topological charge for a "
            "spinor precipitating_event run using 2D skyrmion numbers on "
            "xy-slices of the 3D grid."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory with spinor_particle_features.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    analyze_topology(run_dir)


if __name__ == "__main__":
    main()
