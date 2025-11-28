"""
engines/substrate_3d_fullspace_cu.py

Full-Hilbert-space 3D substrate engine (CuPy / GPU).

Philosophy
----------
- There are no a-priori "particles" or "defects".
- We define only:
    * a local tensor-product Hilbert space H = ⊗_sites H_site,
    * a local Hamiltonian Ĥ built from nearest-neighbor couplings,
    * a random initial state |Ψ(0)⟩ (noise),
    * unitary evolution |Ψ(t)⟩ = e^{-i Ĥ t} |Ψ(0)⟩.
- All "excitation sectors" and "configuration space" structure are
  to be derived from |Ψ(t)⟩, not assumed.

Model summary
-------------
- 3D cubic lattice with periodic boundary conditions:
      Lx × Ly × Lz sites, N = Lx * Ly * Lz.
- Local Hilbert at each site:
      H_site = span{|0>, |1>}  (qubit).
  We do NOT interpret |1> as "a particle" here; it's just a basis label.

- Global Hilbert space:
      dim = 2^N.
- Hamiltonian:
      XY-type nearest-neighbor model on 3D torus,
      optionally with a longitudinal field term.

    Ĥ = -J ∑_<i,j> (σ^x_i σ^x_j + σ^y_i σ^y_j) + h_z ∑_i σ^z_i

  where <i,j> runs over nearest-neighbor pairs in 3D with periodic BC.

- Evolution:
      |Ψ(t)⟩ = e^{-i Ĥ t} |Ψ(0)⟩
  implemented by eigendecomposition of Ĥ on the GPU.

- Analysis (first layer of "emergent configuration" study):
    For each time t_k we compute:
        P_k(t_k) = ∑_{bitstrings of Hamming weight k} |Ψ_α(t_k)|^2
    for k = 0..N.

    This reveals which Hamming-weight sectors acquire and maintain
    significant weight dynamically, as a first step toward identifying
    emergent "excitation-number" sectors.

Contract
--------
This is a physics engine:
- NO file I/O.
- Deterministic given params["seed"].
- Public entry point:

      run_experiment(params: dict) -> results: dict

Returned results dict includes:
- params: echoes of the used parameters,
- times: list of sampling times t_k,
- sector_occupations: P_k(t_k) as a 2D list [time_index, k],
- diagnostics: basic info about lattice, Hilbert-space dimension, etc.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import math
import time

import numpy as np
import cupy as cp
from cupy.typing import NDArray as CpArray


# ---------------------------------------------------------------------------
# RNG helper
# ---------------------------------------------------------------------------

def _set_seed(seed: int | None) -> None:
    """
    Initialize both NumPy and CuPy RNGs.
    """
    if seed is not None:
        np.random.seed(seed)
        cp.random.seed(seed)


# ---------------------------------------------------------------------------
# 3D lattice indexing helpers
# ---------------------------------------------------------------------------

def coord_to_index(x: int, y: int, z: int, Lx: int, Ly: int, Lz: int) -> int:
    """
    Map integer coordinates (x, y, z) to a single site index in [0, N-1].
    """
    return x + Lx * (y + Ly * z)


def index_to_coord(s: int, Lx: int, Ly: int, Lz: int) -> Tuple[int, int, int]:
    """
    Inverse of coord_to_index.
    """
    z = s // (Lx * Ly)
    rem = s % (Lx * Ly)
    y = rem // Lx
    x = rem % Lx
    return x, y, z


def build_neighbors_3d(Lx: int, Ly: int, Lz: int) -> List[List[int]]:
    """
    Build nearest-neighbor list for each site on a 3D cubic lattice
    with periodic boundary conditions.

    Returns:
        neighbors: list of length N; neighbors[s] is a list of site
                   indices that are nearest neighbors of site s.
    """
    N = Lx * Ly * Lz
    neighbors: List[List[int]] = [[] for _ in range(N)]

    for s in range(N):
        x, y, z = index_to_coord(s, Lx, Ly, Lz)

        # +x, -x
        xp = (x + 1) % Lx
        xm = (x - 1) % Lx

        # +y, -y
        yp = (y + 1) % Ly
        ym = (y - 1) % Ly

        # +z, -z
        zp = (z + 1) % Lz
        zm = (z - 1) % Lz

        neighbors[s].append(coord_to_index(xp, y, z, Lx, Ly, Lz))
        neighbors[s].append(coord_to_index(xm, y, z, Lx, Ly, Lz))
        neighbors[s].append(coord_to_index(x, yp, z, Lx, Ly, Lz))
        neighbors[s].append(coord_to_index(x, ym, z, Lx, Ly, Lz))
        neighbors[s].append(coord_to_index(x, y, zp, Lx, Ly, Lz))
        neighbors[s].append(coord_to_index(x, y, zm, Lx, Ly, Lz))

    return neighbors


# ---------------------------------------------------------------------------
# Hamiltonian construction (CPU, converted to GPU)
# ---------------------------------------------------------------------------

def _paulis() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return single-site Pauli matrices and identity as NumPy arrays.
    """
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    id2 = np.eye(2, dtype=np.complex128)
    return sx, sy, sz, id2


def _kron_N(ops: List[np.ndarray]) -> np.ndarray:
    """
    Kronecker product of a list of single-site operators, in order.

    Given ops = [O_0, O_1, ..., O_{N-1}],
    returns O_0 ⊗ O_1 ⊗ ... ⊗ O_{N-1}.
    """
    res = ops[0]
    for op in ops[1:]:
        res = np.kron(res, op)
    return res


def build_full_hamiltonian_xy(
    Lx: int,
    Ly: int,
    Lz: int,
    J: float,
    hz: float = 0.0,
) -> np.ndarray:
    """
    Build the full Hilbert-space Hamiltonian Ĥ for a 3D XY model
    with optional longitudinal field, on the CPU as a dense NumPy array.

        Ĥ = -J ∑_<i,j> (σ^x_i σ^x_j + σ^y_i σ^y_j) + h_z ∑_i σ^z_i

    where <i,j> runs over nearest neighbors on a 3D torus.

    For small 3D lattices (e.g., 2x2x2 or 3x3x1), this is feasible:
      dim = 2^N, with N = Lx * Ly * Lz.
    """
    sx, sy, sz, id2 = _paulis()

    N = Lx * Ly * Lz
    dim = 2 ** N

    H = np.zeros((dim, dim), dtype=np.complex128)

    neighbors = build_neighbors_3d(Lx, Ly, Lz)

    # Build XY couplings
    seen_bonds = set()
    for i in range(N):
        for j in neighbors[i]:
            if j <= i:
                continue  # avoid double counting and self-bonds
            bond = (i, j)
            if bond in seen_bonds:
                continue
            seen_bonds.add(bond)

            # σ^x_i σ^x_j term
            ops_xx: List[np.ndarray] = []
            for site in range(N):
                if site == i or site == j:
                    ops_xx.append(sx)
                else:
                    ops_xx.append(id2)
            H += -J * _kron_N(ops_xx)

            # σ^y_i σ^y_j term
            ops_yy: List[np.ndarray] = []
            for site in range(N):
                if site == i or site == j:
                    ops_yy.append(sy)
                else:
                    ops_yy.append(id2)
            H += -J * _kron_N(ops_yy)

    # Longitudinal field term h_z ∑ σ^z_i
    if abs(hz) > 0.0:
        for i in range(N):
            ops_z: List[np.ndarray] = []
            for site in range(N):
                if site == i:
                    ops_z.append(sz)
                else:
                    ops_z.append(id2)
            H += hz * _kron_N(ops_z)

    # Hermitize explicitly to remove any numerical asymmetries
    H = 0.5 * (H + H.conj().T)
    return H


# ---------------------------------------------------------------------------
# Initial state and time evolution
# ---------------------------------------------------------------------------

def build_random_initial_state(dim: int) -> CpArray[cp.complex128]:
    """
    Build a random complex normalized state |Ψ(0)⟩ in the full Hilbert space.
    """
    real = cp.random.normal(size=dim)
    imag = cp.random.normal(size=dim)
    psi = real + 1j * imag
    norm = cp.linalg.norm(psi)
    if norm == 0:
        raise ValueError("Generated zero-norm initial state.")
    psi /= norm
    return psi


def compute_eigendecomposition(
    H_cp: CpArray[cp.complex128],
) -> Tuple[CpArray[cp.float64], CpArray[cp.complex128]]:
    """
    Diagonalize Ĥ on the GPU:

        Ĥ = V diag(E) V^†

    Returns:
        evals: (dim,) eigenvalues (real)
        evecs: (dim, dim) eigenvectors as columns
    """
    evals, evecs = cp.linalg.eigh(H_cp)
    return evals, evecs


def build_hamming_weights(dim: int, N_sites: int) -> np.ndarray:
    """
    Precompute Hamming weight (number of '1' bits) for each basis index
    in the computational basis of N_sites qubits.

    Basis convention:
        index b in [0, 2^N - 1] represents the bitstring of length N
        in binary, with site 0 as the least significant bit.
    """
    weights = np.zeros(dim, dtype=np.int32)
    for b in range(dim):
        # Python 3.8+: int.bit_count()
        weights[b] = b.bit_count()
    return weights


def time_evolve_sector_occupations(
    evals: CpArray[cp.float64],
    evecs: CpArray[cp.complex128],
    psi0: CpArray[cp.complex128],
    T: float,
    dt: float,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evolve |Ψ(0)⟩ under Ĥ and compute sector occupations:

        P_k(t) = ∑_{bitstrings of Hamming weight k} |Ψ_α(t)|^2

    for k = 0..N_sites at each sampled time t in [0, T].

    Implementation:
      - Work in the eigenbasis of Ĥ:
            psi0 = V c
            psi(t) = V diag(e^{-i E t}) c
      - At each t, transform back to computational basis, compute |Ψ_α(t)|^2,
        and bin by Hamming weight using the precomputed weights array.

    Returns:
        times: shape (n_steps,)
        Pk:    shape (n_steps, N_sites+1)
    """
    dim = psi0.shape[0]
    max_k = int(weights.max())
    N_sites = max_k if max_k > 0 else int(round(math.log2(dim)))

    n_steps = int(math.floor(T / dt)) + 1
    times = np.linspace(0.0, T, n_steps, dtype=np.float64)
    Pk = np.zeros((n_steps, N_sites + 1), dtype=np.float64)

    # Represent psi0 in eigenbasis: c = V^† psi0
    coeffs = evecs.conj().T @ psi0  # (dim,)
    coeffs_conj = coeffs.conj()

    weights_np = weights  # alias
    # We will repeatedly move psi(t) to CPU for probability binning.
    # For small dims (e.g. 2^8 = 256), this is fine.

    for idx, t in enumerate(times):
        t_val = float(t)
        phase = cp.exp(-1j * evals * t_val)
        psi_t_coeffs = coeffs * phase
        # psi(t) in computational basis
        psi_t = evecs @ psi_t_coeffs
        probs_cp = cp.abs(psi_t) ** 2
        probs = cp.asnumpy(probs_cp)  # (dim,)

        # Bin probabilities by Hamming weight
        for k in range(N_sites + 1):
            mask = (weights_np == k)
            if not np.any(mask):
                continue
            Pk[idx, k] = float(probs[mask].sum())

        # (Optional: sanity check normalization)
        # total_prob = Pk[idx, :].sum()
        # if not np.isclose(total_prob, 1.0, atol=1e-6):
        #     print(f"Warning: total prob at t={t_val} is {total_prob}")

    return times, Pk


# ---------------------------------------------------------------------------
# Public engine entry point
# ---------------------------------------------------------------------------

def run_experiment(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core entry point for the full-Hilbert-space 3D substrate engine.

    Expected params keys (all optional, with defaults):
        - Lx (int): lattice size in x (default: 2)
        - Ly (int): lattice size in y (default: 2)
        - Lz (int): lattice size in z (default: 2)
        - J (float): XY coupling strength (default: 1.0)
        - hz (float): longitudinal field strength (default: 0.0)
        - T (float): total evolution time (default: 10.0)
        - dt (float): time step for sampling (default: 0.1)
        - seed (int or None): RNG seed (default: None)

    Returns:
        results dict with keys:
            - "params": effective parameters used
            - "times": list of sample times
            - "sector_occupations": list-of-lists, P_k(t) matrix
            - "diagnostics": basic numerical info
    """
    t_start = time.time()

    # Fill defaults
    Lx = int(params.get("Lx", 2))
    Ly = int(params.get("Ly", 2))
    Lz = int(params.get("Lz", 2))
    J = float(params.get("J", 1.0))
    hz = float(params.get("hz", 0.0))
    T = float(params.get("T", 10.0))
    dt = float(params.get("dt", 0.1))
    seed = params.get("seed", None)

    _set_seed(seed)

    N_sites = Lx * Ly * Lz
    dim = 2 ** N_sites

    # Build Ĥ on CPU
    H_np = build_full_hamiltonian_xy(Lx=Lx, Ly=Ly, Lz=Lz, J=J, hz=hz)

    # Move to GPU
    H_cp = cp.asarray(H_np)

    # Eigendecomposition on GPU
    evals_cp, evecs_cp = compute_eigendecomposition(H_cp)

    # Build random initial state |Ψ(0)⟩ in full Hilbert space
    psi0_cp = build_random_initial_state(dim)

    # Hamming weights in computational basis
    weights_np = build_hamming_weights(dim=dim, N_sites=N_sites)

    # Evolve and compute sector occupations
    times_np, Pk_np = time_evolve_sector_occupations(
        evals=evals_cp,
        evecs=evecs_cp,
        psi0=psi0_cp,
        T=T,
        dt=dt,
        weights=weights_np,
    )

    t_end = time.time()

    diagnostics: Dict[str, Any] = {
        "Lx": Lx,
        "Ly": Ly,
        "Lz": Lz,
        "N_sites": N_sites,
        "dim": dim,
        "J": J,
        "hz": hz,
        "T": T,
        "dt": dt,
        "runtime_sec": t_end - t_start,
        "eig_min": float(cp.asnumpy(evals_cp).min()),
        "eig_max": float(cp.asnumpy(evals_cp).max()),
        "n_steps": len(times_np),
    }

    results: Dict[str, Any] = {
        "params": {
            "Lx": Lx,
            "Ly": Ly,
            "Lz": Lz,
            "J": J,
            "hz": hz,
            "T": T,
            "dt": dt,
            "seed": seed,
        },
        "times": times_np.tolist(),
        "sector_occupations": Pk_np.tolist(),  # shape (n_steps, N_sites+1)
        "diagnostics": diagnostics,
    }

    return results
