"""
geom_precipitation_engine.py

Geometry-driven Hilbert-substrate engine for the Precipitating Event.

This engine:
- Loads an emergent geometry (adjacency) supplied by the experiment.
- Builds a local qubit Hamiltonian on that graph.
- Runs a global quench (hot -> cold) in a constrained local Hilbert space.
- Tracks local observables and "lump" formation as proto-particles.

No file I/O happens here; all saving is done by the experiment script.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class PrecipitationParams:
    """
    Parameters controlling the Precipitating Event dynamics.

    n_sites        : number of sites (must match geometry adjacency size)
    local_dim      : local Hilbert dimension (currently only 2 supported)
    J_coupling     : strength of Heisenberg-type coupling on edges
    h_field        : on-site Z field strength
    defrag_hot     : defrag (clustering) strength before quench
    defrag_cold    : defrag strength after quench
    t_total        : total evolution time
    t_quench       : quench time (0 < t_quench < t_total)
    n_steps        : number of snapshots between 0 and t_total
    """
    n_sites: int = 8
    local_dim: int = 2
    J_coupling: float = 1.0
    h_field: float = 0.2
    defrag_hot: float = 0.3
    defrag_cold: float = 1.0
    t_total: float = 10.0
    t_quench: float = 4.0
    n_steps: int = 101


def pauli_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z


def kron_on_sites(op_single: np.ndarray, site: int, n_sites: int) -> np.ndarray:
    """
    Build an operator that applies op_single on a given site and identity elsewhere.
    """
    assert 0 <= site < n_sites
    I2 = np.eye(2, dtype=complex)
    ops: List[np.ndarray] = []
    for j in range(n_sites):
        if j == site:
            ops.append(op_single)
        else:
            ops.append(I2)
    out = ops[0]
    for a in ops[1:]:
        out = np.kron(out, a)
    return out


def build_local_ops(n_sites: int) -> Dict[str, List[np.ndarray]]:
    """
    Precompute single-site Pauli operators on the full Hilbert space.
    """
    I, X, Y, Z = pauli_matrices()
    X_ops: List[np.ndarray] = []
    Y_ops: List[np.ndarray] = []
    Z_ops: List[np.ndarray] = []
    for i in range(n_sites):
        X_ops.append(kron_on_sites(X, i, n_sites))
        Y_ops.append(kron_on_sites(Y, i, n_sites))
        Z_ops.append(kron_on_sites(Z, i, n_sites))
    return {"X": X_ops, "Y": Y_ops, "Z": Z_ops}


def build_heisenberg_defrag_hamiltonian(
    adjacency: np.ndarray,
    params: PrecipitationParams,
    defrag_strength: float,
    local_ops: Dict[str, List[np.ndarray]],
) -> np.ndarray:
    """
    Build a geometry-aware Hamiltonian:

    H = J * sum_{(i,j) in edges} (X_i X_j + Y_i Y_j + Z_i Z_j)
        - defrag_strength * sum_{(i,j)} Z_i Z_j
        - h_field * sum_i Z_i

    This is a generic local Hamiltonian on the emergent geometry. The Pauli axis
    is just an internal label in Hilbert space; no literal "spin" or particle field
    is imposed.
    """
    n_sites = params.n_sites
    J = params.J_coupling
    h = params.h_field

    X_ops = local_ops["X"]
    Y_ops = local_ops["Y"]
    Z_ops = local_ops["Z"]

    dim = 2 ** n_sites
    H = np.zeros((dim, dim), dtype=complex)

    # Edge terms
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            if adjacency[i, j] != 0:
                # Heisenberg-like coupling
                H += J * (
                    X_ops[i] @ X_ops[j]
                    + Y_ops[i] @ Y_ops[j]
                    + Z_ops[i] @ Z_ops[j]
                )
                # "Defrag" clustering term: encourage aligned Z on adjacent sites
                H += -defrag_strength * (Z_ops[i] @ Z_ops[j])

    # On-site field
    for i in range(n_sites):
        H += -h * Z_ops[i]

    # Make Hermitian explicitly
    H = 0.5 * (H + H.conj().T)
    return H


def diagonalize(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize Hermitian H = V diag(E) V^\dagger.
    """
    E, V = np.linalg.eigh(H)
    return E, V


def evolve_state_spectral(
    evals: np.ndarray,
    evecs: np.ndarray,
    psi0: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Evolve psi0 under H with spectral decomposition:

    psi(t) = V * exp(-i * E * t) * (V^\dagger psi0)
    """
    # Transform to eigenbasis
    alpha0 = evecs.conj().T @ psi0
    phase = np.exp(-1j * evals * t)
    alpha_t = phase * alpha0
    psi_t = evecs @ alpha_t
    return psi_t


def random_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw a Haar-random pure state via complex Gaussian components.
    """
    re = rng.normal(size=dim)
    im = rng.normal(size=dim)
    v = re + 1j * im
    v /= np.linalg.norm(v)
    return v


def compute_local_expectations(
    psi: np.ndarray,
    Z_ops: List[np.ndarray],
) -> np.ndarray:
    """
    Compute <Z_i> for each site in a state psi.
    """
    vals = []
    bra = psi.conj().T
    for Zi in Z_ops:
        v = bra @ (Zi @ psi)
        vals.append(np.real_if_close(v))
    return np.array(vals, dtype=float)


def find_lumps_from_profile(
    z_profile: np.ndarray,
    adjacency: np.ndarray,
    z_threshold: float = 0.5,
) -> Tuple[int, List[List[int]]]:
    """
    Identify "lumps" as connected components (w.r.t adjacency) of sites whose
    |Z_i - mean(Z)| >= z_threshold.

    Returns:
        n_lumps, list of lumps (each lump is a list of site indices).
    """
    n_sites = len(z_profile)
    mean_z = float(np.mean(z_profile))
    mask = np.abs(z_profile - mean_z) >= z_threshold
    visited = np.zeros(n_sites, dtype=bool)
    lumps: List[List[int]] = []

    for i in range(n_sites):
        if not mask[i] or visited[i]:
            continue
        # BFS / flood-fill
        stack = [i]
        visited[i] = True
        comp = [i]
        while stack:
            u = stack.pop()
            for v in range(n_sites):
                if adjacency[u, v] != 0 and mask[v] and not visited[v]:
                    visited[v] = True
                    stack.append(v)
                    comp.append(v)
        lumps.append(sorted(comp))

    return len(lumps), lumps


def compute_exchange_signature_placeholder(
    # This is a placeholder; we will wire in the real exchange protocol later.
    # For now it just returns 0.0 and marks that it's not yet implemented.
    lump_history: List[List[List[int]]],
    times: np.ndarray,
) -> Dict[str, Any]:
    """
    Placeholder for a future exchange-signature computation.

    Once we can reliably form two well-separated lumps, this function should:
        - identify time windows with exactly two lumps,
        - define a graph-based exchange path between them,
        - construct an operational exchange phase (as in the exchange-statistics paper),
        - measure its long-time stability.

    For now it returns a sentinel structure.
    """
    return {
        "implemented": False,
        "exchange_phase_estimate": None,
        "notes": "Exchange signature not yet implemented in this engine.",
    }


def run_experiment(
    geometry: Dict[str, np.ndarray],
    seed: int,
    params_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Core entry point for the engine.

    Args:
        geometry: dict with at least "graph_dist" (n_sites x n_sites).
        seed: random seed for the initial state.
        params_dict: dict used to construct PrecipitationParams.

    Returns:
        results dict with:
            - "params": serialized parameters,
            - "times": array of times,
            - "local_z_t": (n_steps, n_sites) array,
            - "lump_counts": (n_steps,) array of ints,
            - "lump_hist": list of lump lists per time step,
            - "metrics": summary metrics,
            - "exchange": placeholder exchange-signature info.
    """
    rng = np.random.default_rng(seed)

    graph_dist = geometry["graph_dist"]
    n_sites = graph_dist.shape[0]

    params = PrecipitationParams(**params_dict)
    if params.n_sites != n_sites:
        raise ValueError(
            f"params.n_sites={params.n_sites} but geometry has {n_sites} sites."
        )
    if params.local_dim != 2:
        raise ValueError("Only local_dim=2 is supported in this engine (qubits).")

    # Adjacency: treat graph_dist == 1 as edges
    adjacency = (graph_dist == 1).astype(int)

    # Precompute local operators
    local_ops = build_local_ops(n_sites)
    Z_ops = local_ops["Z"]

    # Build hot and cold Hamiltonians
    H_hot = build_heisenberg_defrag_hamiltonian(
        adjacency=adjacency,
        params=params,
        defrag_strength=params.defrag_hot,
        local_ops=local_ops,
    )
    H_cold = build_heisenberg_defrag_hamiltonian(
        adjacency=adjacency,
        params=params,
        defrag_strength=params.defrag_cold,
        local_ops=local_ops,
    )

    evals_hot, evecs_hot = diagonalize(H_hot)
    evals_cold, evecs_cold = diagonalize(H_cold)

    # Time grid and quench index
    times = np.linspace(0.0, params.t_total, params.n_steps)
    if not (0.0 < params.t_quench < params.t_total):
        raise ValueError("Require 0 < t_quench < t_total.")
    k_quench = int(np.argmin(np.abs(times - params.t_quench)))
    t_quench_exact = float(times[k_quench])

    # Initial state: Haar-random, high-entropy
    dim = 2 ** n_sites
    psi0 = random_state(dim, rng)

    # Precompute psi at quench from psi0 under H_hot
    psi_quench = evolve_state_spectral(
        evals_hot, evecs_hot, psi0, t_quench_exact
    )

    # Time evolution and diagnostics
    local_z_t = np.zeros((params.n_steps, n_sites), dtype=float)
    lump_counts = np.zeros(params.n_steps, dtype=int)
    lump_hist: List[List[List[int]]] = []

    for idx, t in enumerate(times):
        if t <= t_quench_exact + 1e-12:
            psi_t = evolve_state_spectral(evals_hot, evecs_hot, psi0, t)
        else:
            dt = float(t - t_quench_exact)
            psi_t = evolve_state_spectral(
                evals_cold, evecs_cold, psi_quench, dt
            )

        # Local expectations
        z_profile = compute_local_expectations(psi_t, Z_ops)
        local_z_t[idx, :] = z_profile

        # Lump finding
        n_lumps, lumps = find_lumps_from_profile(
            z_profile,
            adjacency=adjacency,
            z_threshold=0.5,
        )
        lump_counts[idx] = n_lumps
        lump_hist.append(lumps)

    # Simple summary metrics at final time
    final_z = local_z_t[-1, :]
    final_n_lumps, final_lumps = find_lumps_from_profile(
        final_z, adjacency=adjacency, z_threshold=0.5
    )
    final_lump_sizes = [len(L) for L in final_lumps]

    has_particle_candidates = final_n_lumps > 0

    metrics: Dict[str, Any] = {
        "t_quench_index": int(k_quench),
        "t_quench_effective": t_quench_exact,
        "final_n_lumps": int(final_n_lumps),
        "final_lump_sizes": final_lump_sizes,
        "has_particle_candidates": bool(has_particle_candidates),
        "mean_lump_count": float(np.mean(lump_counts)),
    }

    exchange_info = compute_exchange_signature_placeholder(
        lump_history=lump_hist,
        times=times,
    )

    results: Dict[str, Any] = {
        "params": asdict(params),
        "times": times,
        "local_z_t": local_z_t,
        "lump_counts": lump_counts,
        "lump_hist": lump_hist,
        "metrics": metrics,
        "exchange": exchange_info,
    }

    return results
