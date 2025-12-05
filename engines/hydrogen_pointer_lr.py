#!/usr/bin/env python3
"""
Hydrogen Pointer on Emergent 3D Spacetime
=========================================

Runtime engine that uses the emergent 3D geometry derived from the
Lieb–Robinson substrate (cube2 run) to host a hydrogen-like electron.

This script does NOT recompute LR. It assumes you have already run:

    python substrate.py --n-sites 8 --connectivity cube2 --lr-t-max 2.0 \
           --lr-n-steps 200 --lr-threshold 1e-2 --use-gpu --dtype complex64 \
           --output-dir substrate_cube2_gpu_v1

which produced an asset:

    substrate_cube2_gpu_v1/lr_embedding_3d.npz

We treat that as a tiny, derived 3D causal patch and then build a
larger 3D lattice (Nx x Ny x Nz) whose local connectivity matches that
cube and whose dynamics we calibrate via a simple tight-binding +
Coulomb-like potential.

This is a single-electron Schrödinger evolution on a 3D lattice:
  - Hilbert space H_e = span{|x,y,z>} with N = Nx*Ny*Nz sites.
  - Hamiltonian: H = H_kin + H_pot
      H_kin: nearest-neighbor hopping (discrete Laplacian)
      H_pot: central attractive potential ~ -Z / sqrt(r^2 + a^2)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict
from typing import Tuple, Optional

import numpy as np
from scipy.linalg import eigh as np_eigh


# Optional CuPy backend for larger grids
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:  # pragma: no cover
    cp = None
    HAS_CUPY = False


# =============================================================================
# Config
# =============================================================================


@dataclass
class HydroConfig:
    # Lattice size
    nx: int = 16
    ny: int = 16
    nz: int = 16

    # Tight-binding parameters
    t_hop: float = 1.0        # hopping strength (kinetic scale)
    mass_scale: float = 1.0   # not explicitly used; here for future tweaks

    # Coulomb-like potential
    Z: float = 1.0            # effective nuclear charge
    a_soft: float = 0.5       # softening length in sqrt(r^2 + a^2)

    # Time evolution
    t_max: float = 10.0
    n_times: int = 200

    # Initial electron wavepacket
    init_sigma: float = 1.5   # Gaussian width
    init_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Backend
    use_gpu: bool = False
    dtype: str = "complex64"  # or "complex128"

    # Geometry asset
    lr_asset_dir: str = "substrate_cube2_gpu_v1"

    # Output
    output_dir: str = "hydrogen_pointer_lr_out"


# =============================================================================
# Utilities
# =============================================================================


def choose_backend(use_gpu: bool, dtype_str: str):
    """Select numpy or cupy backend and dtype."""
    if use_gpu and not HAS_CUPY:
        print("WARNING: --use-gpu requested but CuPy not available. Falling back to CPU.")
        use_gpu = False

    if use_gpu:
        xp = cp
        dtype = cp.complex64 if dtype_str == "complex64" else cp.complex128
        print("Hydrogen engine: using CuPy GPU backend.")
    else:
        xp = np
        dtype = np.complex64 if dtype_str == "complex64" else np.complex128
        print("Hydrogen engine: using NumPy CPU backend.")

    return xp, dtype


def grid_coordinates(nx: int, ny: int, nz: int):
    """
    Build 3D grid coordinates centered at (0,0,0):

        x = (i - (nx-1)/2) * dx, etc., with dx=1 for now.
    """
    xs = np.arange(nx) - (nx - 1) / 2.0
    ys = np.arange(ny) - (ny - 1) / 2.0
    zs = np.arange(nz) - (nz - 1) / 2.0
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return X, Y, Z  # shape (nx, ny, nz)


def flatten_index(i: int, j: int, k: int, nx: int, ny: int, nz: int) -> int:
    """Map (i,j,k) to a single index in [0, nx*ny*nz)."""
    return (i * ny + j) * nz + k


# =============================================================================
# Hamiltonian construction
# =============================================================================


def build_hamiltonian(cfg: HydroConfig, xp, dtype):
    """
    Build single-electron Hamiltonian H for a 3D tight-binding lattice:

      H = H_kin + H_pot

    H_kin: nearest-neighbor hopping on a cubic lattice
           (-t) along +/-x, +/-y, +/-z.

    H_pot: diagonal Coulomb-like potential V(r) = -Z / sqrt(r^2 + a^2)
    """
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    N = nx * ny * nz
    t = cfg.t_hop

    print(f"Building Hamiltonian for lattice {nx}x{ny}x{nz} (N={N})...")

    # Coordinates for potential
    X, Y, Z = grid_coordinates(nx, ny, nz)
    R2 = X**2 + Y**2 + Z**2
    V = -cfg.Z / np.sqrt(R2 + cfg.a_soft**2)  # shape (nx,ny,nz)

    # Construct H as dense matrix (for now; N=4096 @ complex64 is ~128 MB)
    H = xp.zeros((N, N), dtype=dtype)

    # Potential term (diagonal)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx = flatten_index(i, j, k, nx, ny, nz)
                H[idx, idx] = V[i, j, k]

    # Kinetic term: nearest neighbors
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx = flatten_index(i, j, k, nx, ny, nz)

                # x±1
                if i + 1 < nx:
                    jdx = flatten_index(i + 1, j, k, nx, ny, nz)
                    H[idx, jdx] += -t
                    H[jdx, idx] += -t
                if i - 1 >= 0:
                    jdx = flatten_index(i - 1, j, k, nx, ny, nz)
                    H[idx, jdx] += -t
                    H[jdx, idx] += -t

                # y±1
                if j + 1 < ny:
                    jdx = flatten_index(i, j + 1, k, nx, ny, nz)
                    H[idx, jdx] += -t
                    H[jdx, idx] += -t
                if j - 1 >= 0:
                    jdx = flatten_index(i, j - 1, k, nx, ny, nz)
                    H[idx, jdx] += -t
                    H[jdx, idx] += -t

                # z±1
                if k + 1 < nz:
                    jdx = flatten_index(i, j, k + 1, nx, ny, nz)
                    H[idx, jdx] += -t
                    H[jdx, idx] += -t
                if k - 1 >= 0:
                    jdx = flatten_index(i, j, k - 1, nx, ny, nz)
                    H[idx, jdx] += -t
                    H[jdx, idx] += -t

    print("Hamiltonian build complete.")
    return H


# =============================================================================
# Initial state (Gaussian wavepacket)
# =============================================================================


def build_initial_state(cfg: HydroConfig, xp, dtype):
    """
    Build an initial Gaussian wavepacket centered near the origin (plus offset).

    ψ(x,y,z) ∝ exp(- (r - r0)^2 / (2 σ^2))
    """
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    X, Y, Z = grid_coordinates(nx, ny, nz)

    x0, y0, z0 = cfg.init_offset
    R2 = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2

    sigma2 = cfg.init_sigma**2
    psi = np.exp(-R2 / (2.0 * sigma2)).astype(np.complex128)

    # Flatten and normalize
    psi_flat = psi.reshape(-1)
    norm = np.linalg.norm(psi_flat)
    if norm > 0:
        psi_flat /= norm

    psi_xp = xp.asarray(psi_flat, dtype=dtype)
    print("Initial state: Gaussian wavepacket normalized.")
    return psi_xp


# =============================================================================
# Time evolution via eigendecomposition
# =============================================================================


def evolve_eigen(H, psi0, cfg: HydroConfig, xp, dtype):
    """
    Diagonalize H once, then evolve:

      H = V diag(E) V†
      ψ(t) = V exp(-i E t) V† ψ(0)

    This matches the trick in the optimized substrate engine.
    """
    print("Diagonalizing Hamiltonian (this is the heavy step)...")
    # Bring H to CPU for eigh if we're on GPU
    if xp is cp:
        H_cpu = cp.asnumpy(H)
        E, V = np_eigh(H_cpu)
        E_xp = xp.asarray(E)
        V_xp = xp.asarray(V, dtype=dtype)
    else:
        E, V = np_eigh(H)
        E_xp = xp.asarray(E)
        V_xp = xp.asarray(V, dtype=dtype)

    V_dag = V_xp.conj().T

    # Transform initial state to eigenbasis
    psi_eig0 = V_dag @ psi0

    times = np.linspace(0.0, cfg.t_max, cfg.n_times)
    N = H.shape[0]
    psi_t = xp.zeros((cfg.n_times, N), dtype=dtype)

    print("Evolving in time...")
    for idx, t in enumerate(times):
        phases = xp.exp(-1j * E_xp * t)
        psi_eig_t = phases * psi_eig0
        psi_t[idx] = V_xp @ psi_eig_t

    # Bring back to CPU for analysis
    psi_t_np = psi_t.get() if xp is cp else psi_t
    return times, psi_t_np


# =============================================================================
# Analysis: radial profiles, mean radius
# =============================================================================


def analyze_trajectories(cfg: HydroConfig, times, psi_t: np.ndarray, out_dir: str):
    """
    Compute:
      - mean radius vs time
      - a few radial probability profiles

    and save to npz + simple text summary.
    """
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    N = nx * ny * nz
    assert psi_t.shape[1] == N

    X, Y, Z = grid_coordinates(nx, ny, nz)
    R = np.sqrt(X**2 + Y**2 + Z**2).reshape(-1)  # shape (N,)

    mean_r = np.zeros_like(times)
    for t_idx, psi in enumerate(psi_t):
        prob = np.abs(psi)**2
        mean_r[t_idx] = float(np.sum(prob * R))

    # Pick a few times for radial histograms
    sample_indices = np.linspace(0, len(times) - 1, 5, dtype=int)
    r_bins = np.linspace(0.0, np.max(R), 50)
    radial_profiles = []

    for t_idx in sample_indices:
        psi = psi_t[t_idx]
        prob = np.abs(psi)**2
        # Bin probabilities by radius
        hist = np.zeros_like(r_bins[:-1])
        for p, r in zip(prob, R):
            # find bin
            k = np.searchsorted(r_bins, r) - 1
            if 0 <= k < len(hist):
                hist[k] += p
        radial_profiles.append(hist)

    radial_profiles = np.array(radial_profiles)

    # Save everything
    np.savez(
        os.path.join(out_dir, "hydrogen_analysis.npz"),
        times=times,
        mean_r=mean_r,
        r_bins=r_bins,
        sample_indices=sample_indices,
        radial_profiles=radial_profiles,
    )

    # Quick text summary
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("Hydrogen pointer LR analysis\n")
        f.write("============================\n\n")
        f.write(f"Config: {asdict(cfg)}\n\n")
        f.write("Mean radius over time (first and last 5 samples):\n")
        for i in range(min(5, len(times))):
            f.write(f"  t={times[i]:8.3f}  <r>={mean_r[i]:.4f}\n")
        f.write("  ...\n")
        for i in range(len(times) - 5, len(times)):
            f.write(f"  t={times[i]:8.3f}  <r>={mean_r[i]:.4f}\n")

    print("Analysis complete. Saved hydrogen_analysis.npz and summary.txt.")


# =============================================================================
# Main
# =============================================================================


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Hydrogen pointer on emergent 3D spacetime (LR-derived)."
    )
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--t-hop", type=float, default=1.0)
    parser.add_argument("--Z", type=float, default=1.0)
    parser.add_argument("--a-soft", type=float, default=0.5)
    parser.add_argument("--t-max", type=float, default=10.0)
    parser.add_argument("--n-times", type=int, default=200)
    parser.add_argument("--init-sigma", type=float, default=1.5)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument(
        "--dtype", type=str, default="complex64",
        choices=["complex64", "complex128"]
    )
    parser.add_argument(
        "--lr-asset-dir",
        type=str,
        default="substrate_cube2_gpu_v1",
        help="Directory containing lr_embedding_3d.npz (currently not used numerically, but kept for future calibration).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hydrogen_pointer_lr_out",
    )

    args = parser.parse_args(argv)

    cfg = HydroConfig(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        t_hop=args.t_hop,
        Z=args.Z,
        a_soft=args.a_soft,
        t_max=args.t_max,
        n_times=args.n_times,
        init_sigma=args.init_sigma,
        use_gpu=args.use_gpu,
        dtype=args.dtype,
        lr_asset_dir=args.lr_asset_dir,
        output_dir=args.output_dir,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("Hydrogen pointer LR config:")
    print(asdict(cfg))
    print()

    # Backend
    xp, dtype = choose_backend(cfg.use_gpu, cfg.dtype)

    # (For now we only *record* the LR asset path; in a next iteration we can
    #  use its metrics to calibrate t_hop and time scaling.)
    lr_asset_path = os.path.join(cfg.lr_asset_dir, "lr_embedding_3d.npz")
    if os.path.exists(lr_asset_path):
        print(f"Found LR geometry asset at {lr_asset_path}.")
    else:
        print(f"WARNING: LR asset {lr_asset_path} not found. Proceeding anyway.")

    # Build H
    H = build_hamiltonian(cfg, xp, dtype)

    # Initial state
    psi0 = build_initial_state(cfg, xp, dtype)

    # Time evolution
    times, psi_t = evolve_eigen(H, psi0, cfg, xp, dtype)

    # Analysis
    analyze_trajectories(cfg, times, psi_t, cfg.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
