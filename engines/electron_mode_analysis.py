#!/usr/bin/env python3
"""
===============================================================================
Electron Mode Analysis (timeseries-only version)
===============================================================================

Given a single run directory that contains:

    <run_dir>/data/timeseries.npz

as produced by emergent_hydrogen_pointer_substrate.py, this script:

  1. Loads:
       - data/timeseries.npz

  2. Extracts:
       - positions (N, 3)
       - electron_pop_final (N,)

     and infers:
       - Lx, Ly, Lz (assuming cubic lattice)
       - lattice spacing a

  3. Performs a weighted PCA on the electron density:
       - center of mass R_cm
       - covariance matrix Σ
       - eigenvalues/eigenvectors of Σ
       - axis lengths and anisotropy ratios

  4. (Optional) Builds the lattice Laplacian L from inferred Lx, Ly, Lz, a,
     computes a small number of eigenmodes, and compares their densities
     to the electron density using a Bhattacharyya coefficient.

All of this is derived directly from the stored data; we do not require
params.json or summary.json, and we do not hard-code any orbital shapes.
===============================================================================
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Tuple

import numpy as np

# Optional SciPy for sparse eigenmodes
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_timeseries(run_dir: str) -> Dict[str, Any]:
    data_path = os.path.join(run_dir, "data", "timeseries.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find timeseries.npz at {data_path}")
    data = np.load(data_path)
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Lattice reconstruction from positions
# ---------------------------------------------------------------------------


def infer_lattice_from_positions(positions: np.ndarray) -> Tuple[int, int, int, float]:
    """
    Given positions of shape (N, 3) that were built as a regular Lx x Ly x Lz
    grid and then centered, infer Lx, Ly, Lz (assuming cubic) and spacing a.
    """
    N = positions.shape[0]
    # Assume cubic
    L = round(N ** (1.0 / 3.0))
    if L * L * L != N:
        raise ValueError(
            f"Cannot infer cubic lattice: N={N} is not a perfect cube."
        )
    Lx = Ly = Lz = L

    # Undo centering: get unique x, y, z coords and infer spacing from gaps
    xs = np.unique(positions[:, 0])
    xs.sort()
    if len(xs) < 2:
        raise ValueError("Not enough distinct x-coordinates to infer spacing.")
    dxs = np.diff(xs)
    a = float(np.min(dxs[dxs > 0]))  # smallest positive step

    return Lx, Ly, Lz, a


def build_laplacian(Lx: int, Ly: int, Lz: int, a: float) -> np.ndarray:
    """
    Build Laplacian L (N x N) in the same node ordering as the engine:
      i = (ix * Ly + iy) * Lz + iz
    """
    N = Lx * Ly * Lz
    a2 = a ** 2
    Lmat = np.zeros((N, N), dtype=float)

    def idx_fun(ix_: int, iy_: int, iz_: int) -> int:
        return (ix_ * Ly + iy_) * Lz + iz_

    for ix in range(Lx):
        for iy in range(Ly):
            for iz in range(Lz):
                i = idx_fun(ix, iy, iz)
                diag = 0.0
                for dx, dy, dz in [
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                ]:
                    jx, jy, jz = ix + dx, iy + dy, iz + dz
                    if 0 <= jx < Lx and 0 <= jy < Ly and 0 <= jz < Lz:
                        j = idx_fun(jx, jy, jz)
                        Lmat[i, j] = 1.0 / a2
                        diag -= 1.0 / a2
                Lmat[i, i] = diag

    return Lmat


# ---------------------------------------------------------------------------
# Weighted PCA of electron density
# ---------------------------------------------------------------------------


def electron_pca(positions: np.ndarray, electron_pop: np.ndarray) -> Dict[str, Any]:
    """
    Perform weighted PCA of the electron density.

    positions: (N, 3)
    electron_pop: (N,)
    """
    w = np.asarray(electron_pop, dtype=float)
    total_w = w.sum()
    if total_w <= 0.0:
        raise ValueError("Electron population is zero; cannot do PCA.")

    w_norm = w / total_w
    R_cm = np.sum(positions * w_norm[:, None], axis=0)

    X = positions - R_cm[None, :]
    cov = (X.T * w_norm) @ X  # (3, 3)

    evals, evecs = np.linalg.eigh(cov)  # ascending
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    axis_lengths = np.sqrt(np.clip(evals, 0.0, np.inf))

    if axis_lengths[0] > 0:
        ratio_mid = axis_lengths[1] / axis_lengths[0]
        ratio_min = axis_lengths[2] / axis_lengths[0]
    else:
        ratio_mid = 0.0
        ratio_min = 0.0

    return {
        "R_cm": R_cm,
        "cov": cov,
        "evals": evals,
        "evecs": evecs,
        "axis_lengths": axis_lengths,
        "ratio_mid": ratio_mid,
        "ratio_min": ratio_min,
    }


# ---------------------------------------------------------------------------
# Laplacian eigenmodes and density overlaps
# ---------------------------------------------------------------------------


def compute_laplacian_modes(
    L: np.ndarray,
    electron_pop: np.ndarray,
    n_modes: int,
    max_nodes: int,
) -> Dict[str, Any]:
    """
    Compute lowest n_modes eigenmodes of L and compare their squared
    amplitudes (densities) to the electron density.

    If SciPy is not available or the problem is too large, returns a dict
    with "available" = False.
    """
    N = L.shape[0]
    if N > max_nodes:
        return {
            "available": False,
            "reason": f"Matrix too large (N={N} > max_nodes={max_nodes}); skipping mode analysis.",
        }
    if not HAVE_SCIPY:
        return {
            "available": False,
            "reason": "SciPy not installed; skipping mode analysis.",
        }

    print(f"[info] Building sparse Laplacian (N={N})...")
    L_sparse = sp.csr_matrix(L)

    k = min(n_modes, N - 2)
    print(f"[info] Computing {k} lowest Laplacian eigenmodes...")
    evals, evecs = spla.eigsh(L_sparse, k=k, which="SA")  # smallest algebraic

    p = np.asarray(electron_pop, dtype=float)
    total_p = p.sum()
    if total_p <= 0:
        raise ValueError("Electron population is zero; cannot compare to modes.")
    p /= total_p

    overlaps = []
    for n in range(k):
        phi = evecs[:, n]
        rho = np.abs(phi) ** 2
        rho_sum = rho.sum()
        if rho_sum <= 0:
            overlaps.append(0.0)
            continue
        rho /= rho_sum
        bc = float(np.sum(np.sqrt(p * rho)))  # Bhattacharyya coefficient
        overlaps.append(bc)

    overlaps = np.array(overlaps, dtype=float)

    return {
        "available": True,
        "evals": evals,
        "evecs": evecs,
        "overlaps": overlaps,
    }


# ---------------------------------------------------------------------------
# Main analysis routine
# ---------------------------------------------------------------------------


def analyze_run(
    run_dir: str,
    n_modes: int = 10,
    max_nodes: int = 6000,
) -> None:
    ts = load_timeseries(run_dir)

    print("================================================================")
    print("Electron Mode Analysis (timeseries-only)")
    print("================================================================")
    print(f"Run dir : {run_dir}")
    print("================================================================")
    print()

    # Core data
    positions = ts.get("positions", None)
    if positions is None:
        raise KeyError(
            "positions not found in timeseries.npz; "
            "rerun the simulation with the engine that stores positions."
        )

    electron_pop = ts.get("electron_pop_final", None)
    if electron_pop is None:
        raise KeyError(
            "electron_pop_final not found in timeseries.npz; "
            "rerun the simulation with the updated engine that saves it."
        )

    positions = np.asarray(positions, dtype=float)
    electron_pop = np.asarray(electron_pop, dtype=float)

    N = positions.shape[0]
    print(f"N (number of lattice sites): {N}")
    print()

    # Infer lattice
    Lx, Ly, Lz, a = infer_lattice_from_positions(positions)
    print("Inferred lattice:")
    print(f"  Lx, Ly, Lz    : {Lx}, {Ly}, {Lz}")
    print(f"  spacing a     : {a}")
    print()

    # Weighted PCA
    print("---- Electron density PCA ----")
    pca_res = electron_pca(positions, electron_pop)
    R_cm = pca_res["R_cm"]
    axis_lengths = pca_res["axis_lengths"]
    evals = pca_res["evals"]
    evecs = pca_res["evecs"]
    ratio_mid = pca_res["ratio_mid"]
    ratio_min = pca_res["ratio_min"]

    print(f"Center of mass (x, y, z): {R_cm}")
    print("Eigenvalues of covariance (variances along principal axes):")
    for i, lam in enumerate(evals):
        print(f"  λ[{i}] = {lam:.6e}")
    print("Axis lengths (sqrt(λ)) along principal directions:")
    for i, a_len in enumerate(axis_lengths):
        print(f"  a[{i}] = {a_len:.6e}")
    print()
    print("Principal axis directions (columns are eigenvectors):")
    for i in range(3):
        v = evecs[:, i]
        print(f"  v[{i}] = ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})")
    print()
    print("Anisotropy ratios (relative to largest axis):")
    print(f"  mid / max = {ratio_mid:.4f}")
    print(f"  min / max = {ratio_min:.4f}")
    print("------------------------------")
    print()

    # Optional Laplacian eigenmodes
    print("---- Lattice mode analysis (substrate-derived 'orbitals') ----")
    if not HAVE_SCIPY:
        print("SciPy not available; skipping Laplacian eigenmode analysis.")
        mode_res = {"available": False, "reason": "SciPy not installed."}
    else:
        print(f"Attempting eigenmode analysis with max_nodes={max_nodes}, N={N}...")
        L = build_laplacian(Lx, Ly, Lz, a)
        mode_res = compute_laplacian_modes(
            L=L,
            electron_pop=electron_pop,
            n_modes=n_modes,
            max_nodes=max_nodes,
        )

    if not mode_res["available"]:
        print(f"Mode analysis unavailable: {mode_res['reason']}")
    else:
        evals_L = mode_res["evals"]
        overlaps = mode_res["overlaps"]
        print(f"Computed {len(evals_L)} Laplacian eigenmodes.")
        print("Lowest eigenvalues of L (units ~ 1/a^2):")
        for i, lam in enumerate(evals_L):
            print(f"  μ[{i}] = {lam:.6e}")
        print()
        print("Electron density overlaps with mode densities (Bhattacharyya coefficients):")
        for i, bc in enumerate(overlaps):
            print(f"  mode {i:2d}: overlap = {bc:.4f}")
        print("Values closer to 1.0 mean stronger similarity in density shape.")
    print("--------------------------------------------")
    print()
    print("Analysis complete.")
    print("================================================================")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze electron pointer density and substrate modes for a single run (timeseries-only)."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the run directory (must contain data/timeseries.npz).",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=10,
        help="Number of lowest Laplacian modes to compute (if feasible).",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=6000,
        help="Maximum N=Lx*Ly*Lz for which to attempt eigenmode analysis.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_dir = os.path.abspath(args.run_dir)
    analyze_run(run_dir, n_modes=args.n_modes, max_nodes=args.max_nodes)


if __name__ == "__main__":
    main()
