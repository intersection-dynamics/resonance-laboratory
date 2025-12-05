#!/usr/bin/env python3
"""
spinor_precipitating_event.py
=============================

Spinor + gauge-link version of the precipitating event quench lab.

Goal
----

Upgrade the scalar Hilbert-space toy into a minimal spinor + U(1) gauge 
structure on the emergent LR geometry, while keeping:

  - No particles, only Hilbert space.
  - 3D geometry from LR metrics + embedding.
  - Quench + cooling schedule (lambda(t), mu(t)) from the scalar lab.

Structure
---------

Hilbert space:

  - Sites i = 0..N-1 form the emergent LR graph.
  - At each site, we have a 2-component spinor:

        Psi(i) = (psi_up(i), psi_dn(i))^T ∈ C^2

  - Global state |Psi> is a vector of length 2N (flattened).

Geometry:

  - Loaded from asset_dir / lr_metrics.npz and 
               asset_dir / lr_embedding_3d_smooth.npz (or fallback).
  - Adjacency from graph_dist == 1.

Hamiltonian pieces (time-independent matrices in 2N x 2N space):

  1) H_hot:
       - random on-site energies (different for each spin component),
       - random spin-mixing terms at each site.

  2) H_order:
       - gauge-neutral hopping on edges with strength J_order,
         same for both spin components:
           H_order ⊃ -J_order * sum_{(i,j)} Psi(i)† Psi(j) + h.c.

  3) H_spin:
       - spin coupling to a background "magnetic" field B_i derived from
         the rotation vector Omega and position X_i:

           B_i = spin_coupling * (Omega × X_i)

         and locally:

           H_spin_i = B_i · sigma  (Pauli matrices)

  4) H_gauge:
       - U(1) gauge-link phases on hoppings, acting as 2x2 identity in spin space:

           U_ij = exp(i * A_ij) * I_2

         with A_ij ∝ gauge_curl * (Omega · (m_ij × e_ij)),
         where m_ij is edge midpoint and e_ij = r_j - r_i.

Time-dependent Hamiltonian:

  - As in the scalar precipitating_event:

      H_base(t) = (1 - lambda(t)) H_hot
                  + lambda(t) (H_order + H_spin + H_gauge)

      H_eff(t)  = mu(t) * H_base(t)

  - lambda(t):
        ramp 0→1 over t_quench = quench_fraction * t_max using sin^2.

  - mu(t):
        1 on [0, t_quench], then cos^2 ramp 1→0 on [t_quench, t_max].

Evolution:

  - i d|Psi>/dt = H_eff(t) |Psi>
  - Integrated with RK4, re-normalizing after each step.

Outputs:

  - asset_dir / output_dir / spinor_precipitating_event_timeseries.npz:
        times      : (T,)
        psi_t      : (T, N, 2) complex array (spinor field over time)
        X          : (N,3) embedding
        config     : dict of config parameters

  - asset_dir / output_dir / summary.txt:
        config, some norms and spin diagnostics at sample times.

This is a first spinor + gauge-link quench lab, ready to be coupled
later to the topology/cluster machinery to extract emergent "particles".
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple, Any

import numpy as np

# We reuse the geometry + schedule tools from the scalar quench lab if present.
try:
    from precipitating_event import (
        load_metrics,
        load_embedding,
        lambda_quench,
        cooling_mu,
    )
except ImportError:
    # Fallback: local definitions in case precipitating_event is not importable.
    def load_metrics(asset_dir: str, metrics_file: str):
        path = os.path.join(asset_dir, metrics_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find metrics file at {path}")
        data = np.load(path)
        D_lr = data.get("D_lr")
        D_prop = data.get("D_prop")
        graph_dist = data.get("graph_dist")
        return D_lr, D_prop, graph_dist

    def load_embedding(asset_dir: str, smooth_file: str, fallback_file: str):
        path_smooth = os.path.join(asset_dir, smooth_file)
        if os.path.exists(path_smooth):
            data = np.load(path_smooth)
            if "X_smooth" in data:
                X = data["X_smooth"]
            elif "X" in data:
                X = data["X"]
            else:
                raise KeyError(f"{smooth_file} does not contain X_smooth or X.")
            return X, True
        path_orig = os.path.join(asset_dir, fallback_file)
        if not os.path.exists(path_orig):
            raise FileNotFoundError(
                f"Neither {smooth_file} nor {fallback_file} found in {asset_dir}"
            )
        data = np.load(path_orig)
        if "X" not in data:
            raise KeyError(f"{fallback_file} does not contain X.")
        return data["X"], False

    def lambda_quench(t: float, cfg) -> float:
        t_quench = cfg.quench_fraction * cfg.t_max
        if t <= 0.0:
            return 0.0
        if t >= t_quench:
            return 1.0
        x = t / t_quench
        return float(np.sin(0.5 * np.pi * x) ** 2)

    def cooling_mu(t: float, cfg) -> float:
        t_quench = cfg.quench_fraction * cfg.t_max
        if t <= t_quench:
            return 1.0
        if t >= cfg.t_max:
            return 0.0
        denom = max(cfg.t_max - t_quench, 1e-12)
        x = (t - t_quench) / denom
        x = min(max(x, 0.0), 1.0)
        return float(np.cos(0.5 * np.pi * x) ** 2)


# =============================================================================
# Config
# =============================================================================


@dataclass
class SpinorConfig:
    # Geometry / assets
    asset_dir: str = "cubic_universe_L3"
    metrics_file: str = "lr_metrics.npz"
    embedding_smooth_file: str = "lr_embedding_3d_smooth.npz"
    embedding_file: str = "lr_embedding_3d.npz"

    # Spinor structure
    spin_dim: int = 2  # two-component spinor per site

    # Hamiltonian strengths
    hot_strength: float = 1.0   # disordered H_hot
    J_order: float = 1.0        # gauge-neutral hopping
    spin_coupling: float = 0.5  # strength of B·sigma
    gauge_strength: float = 0.8 # magnitude of U(1) phases on edges
    gauge_curl: float = 0.3     # geometric factor for A_ij from Omega · (m×e)

    # Background rotation (defines geometry + B-field alignment)
    Omega_x: float = 0.0
    Omega_y: float = 0.0
    Omega_z: float = 1.0  # strong spin around z-axis

    # Quench + cooling schedule
    t_max: float = 10.0
    n_steps: int = 2000
    quench_fraction: float = 0.4

    # Initial state / randomness
    init_random_spinor: bool = True
    seed: int = 12345

    # Output
    output_dir: str = "spinor_precipitating_event_out"


# =============================================================================
# Geometry helpers
# =============================================================================


def build_adjacency_from_graph_dist(graph_dist: np.ndarray) -> Dict[int, List[int]]:
    """
    Build adjacency dict from graph distance matrix:
      edge (i,j) exists if graph_dist[i,j] == 1.
    """
    N = graph_dist.shape[0]
    adjacency: Dict[int, List[int]] = {i: [] for i in range(N)}
    for i in range(N):
        for j in range(N):
            if i != j and np.isclose(graph_dist[i, j], 1.0):
                adjacency[i].append(j)
    return adjacency


# =============================================================================
# Pauli matrices and spin/B-field coupling
# =============================================================================


def pauli_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return sx, sy, sz


def local_spin_hamiltonians(
    X: np.ndarray,
    Omega: np.ndarray,
    spin_coupling: float,
    spin_dim: int,
) -> List[np.ndarray]:
    """
    For each site i with position X[i], build a local 2x2 (or spin_dim x spin_dim)
    spin Hamiltonian:

        B_i = spin_coupling * (Omega × X_i)
        H_spin_i = B_i · sigma

    For spin_dim != 2, we simply embed B·sigma into the top-left 2x2 block
    and leave remaining components uncoupled (toy).
    """
    N = X.shape[0]
    sx, sy, sz = pauli_matrices()
    H_list: List[np.ndarray] = []

    for i in range(N):
        r = X[i]
        B = spin_coupling * np.cross(Omega, r)  # 3-vector
        H2 = B[0] * sx + B[1] * sy + B[2] * sz  # 2x2

        if spin_dim == 2:
            H_local = H2
        else:
            # simple embedding: H2 in top-left, zeros elsewhere
            H_local = np.zeros((spin_dim, spin_dim), dtype=np.complex128)
            H_local[:2, :2] = H2

        H_list.append(H_local)

    return H_list


# =============================================================================
# Gauge-link construction
# =============================================================================


def build_u1_links(
    X: np.ndarray,
    adjacency: Dict[int, List[int]],
    Omega: np.ndarray,
    gauge_strength: float,
    gauge_curl: float,
) -> Dict[Tuple[int, int], complex]:
    """
    U(1) gauge links on edges:

       U_ij = exp(i * A_ij)

    with A_ij derived from geometry:

       m_ij = (r_i + r_j)/2
       e_ij = r_j - r_i
       a_ij = Omega · (m_ij × e_ij)

       A_ij = gauge_strength * gauge_curl * a_ij

    Returns a dict mapping (i,j) -> U_ij for i!=j, symmetric with U_ji = conj(U_ij).
    """
    N = X.shape[0]
    links: Dict[Tuple[int, int], complex] = {}
    for i in range(N):
        ri = X[i]
        for j in adjacency[i]:
            if j <= i:
                continue
            rj = X[j]
            m = 0.5 * (ri + rj)
            e = rj - ri
            a = np.dot(Omega, np.cross(m, e))
            A = gauge_strength * gauge_curl * a
            U = np.exp(1j * A)
            links[(i, j)] = U
            links[(j, i)] = np.conj(U)
    return links


# =============================================================================
# Hamiltonians in spinor space (2N x 2N)
# =============================================================================


def build_spinor_hot_hamiltonian(
    N: int,
    spin_dim: int,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Random "hot" Hamiltonian on spinors:

      - random on-site energies per spin component,
      - random spin-mixing at each site.

    No inter-site terms here; those are in H_order/H_gauge.
    """
    D = N * spin_dim
    H = np.zeros((D, D), dtype=np.complex128)

    for i in range(N):
        # site block indices
        start = i * spin_dim
        end = (i + 1) * spin_dim

        # random Hermitian matrix: A + A†
        A = strength * (rng.normal(size=(spin_dim, spin_dim)) +
                        1j * rng.normal(size=(spin_dim, spin_dim)))
        H_loc = 0.5 * (A + A.conj().T)
        H[start:end, start:end] += H_loc

    return H


def build_spinor_order_hamiltonian(
    N: int,
    spin_dim: int,
    adjacency: Dict[int, List[int]],
    J_order: float,
) -> np.ndarray:
    """
    Gauge-neutral hopping that tends to align phases between neighboring sites:

          H_order ⊃ -J_order * sum_{(i,j)} Psi(i)† Psi(j) + h.c.

    Implemented as -J_order * I_spin in each off-diagonal block for edges.
    """
    D = N * spin_dim
    H = np.zeros((D, D), dtype=np.complex128)

    I_s = np.eye(spin_dim, dtype=np.complex128)

    for i in range(N):
        for j in adjacency[i]:
            if j <= i:
                continue
            si = i * spin_dim
            sj = j * spin_dim
            H[si:si+spin_dim, sj:sj+spin_dim] += -J_order * I_s
            H[sj:sj+spin_dim, si:si+spin_dim] += -J_order * I_s

    # Hermitian symmetrization not strictly necessary but safe.
    H = 0.5 * (H + H.conj().T)
    return H


def build_spinor_spin_hamiltonian(
    N: int,
    spin_dim: int,
    H_loc_list: List[np.ndarray],
) -> np.ndarray:
    """
    Assemble block-diagonal spin Hamiltonian from local H_spin_i matrices.
    """
    D = N * spin_dim
    H = np.zeros((D, D), dtype=np.complex128)

    for i in range(N):
        si = i * spin_dim
        H[si:si+spin_dim, si:si+spin_dim] += H_loc_list[i]

    return H


def build_spinor_gauge_hamiltonian(
    N: int,
    spin_dim: int,
    adjacency: Dict[int, List[int]],
    links_u1: Dict[Tuple[int, int], complex],
    J_gauge: float,
) -> np.ndarray:
    """
    Gauge-covariant hopping:

        H_gauge ⊃ -J_gauge * sum_{(i,j)} Psi(i)† U_ij Psi(j) + h.c.
    
    For U(1), U_ij is a complex phase times I_spin.
    """
    D = N * spin_dim
    H = np.zeros((D, D), dtype=np.complex128)

    I_s = np.eye(spin_dim, dtype=np.complex128)

    for i in range(N):
        for j in adjacency[i]:
            if j <= i:
                continue
            U_ij = links_u1[(i, j)]
            U_ji = links_u1[(j, i)]
            si = i * spin_dim
            sj = j * spin_dim

            # H_ij = -J_gauge * U_ij * I_spin
            H[si:si+spin_dim, sj:sj+spin_dim] += -J_gauge * U_ij * I_s
            H[sj:sj+spin_dim, si:si+spin_dim] += -J_gauge * U_ji * I_s

    # Hermitian symmetrization (should be already Hermitian)
    H = 0.5 * (H + H.conj().T)
    return H


# =============================================================================
# Time evolution
# =============================================================================


def schrodinger_rhs_spinor(
    t: float,
    psi: np.ndarray,
    H_hot: np.ndarray,
    H_order: np.ndarray,
    H_spin: np.ndarray,
    H_gauge: np.ndarray,
    cfg: SpinorConfig,
) -> np.ndarray:
    lam = lambda_quench(t, cfg)
    mu = cooling_mu(t, cfg)
    H_base = (1.0 - lam) * H_hot + lam * (H_order + H_spin + H_gauge)
    H_eff = mu * H_base
    return -1j * (H_eff @ psi)


def rk4_step_spinor(
    t: float,
    psi: np.ndarray,
    dt: float,
    H_hot: np.ndarray,
    H_order: np.ndarray,
    H_spin: np.ndarray,
    H_gauge: np.ndarray,
    cfg: SpinorConfig,
) -> np.ndarray:
    k1 = schrodinger_rhs_spinor(t, psi, H_hot, H_order, H_spin, H_gauge, cfg)
    k2 = schrodinger_rhs_spinor(t + 0.5*dt, psi + 0.5*dt*k1, H_hot, H_order, H_spin, H_gauge, cfg)
    k3 = schrodinger_rhs_spinor(t + 0.5*dt, psi + 0.5*dt*k2, H_hot, H_order, H_spin, H_gauge, cfg)
    k4 = schrodinger_rhs_spinor(t + dt, psi + dt*k3, H_hot, H_order, H_spin, H_gauge, cfg)

    psi_next = psi + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # normalize
    norm = np.linalg.norm(psi_next)
    if norm > 0:
        psi_next /= norm

    return psi_next


# =============================================================================
# Diagnostics
# =============================================================================


def spin_expectation(
    psi_spinor: np.ndarray,
    spin_dim: int,
) -> Tuple[float, float, float]:
    """
    Given psi_spinor of shape (N, spin_dim), compute global spin expectations
    Sx, Sy, Sz as expectation values of total spin operator (sum over sites).

    For spin_dim=2, use Pauli matrices. For larger spin_dim, we only consider
    the first two components.
    """
    N = psi_spinor.shape[0]
    if spin_dim < 2:
        return 0.0, 0.0, 0.0

    sx, sy, sz = pauli_matrices()
    psi2 = psi_spinor[:, :2]  # (N,2)

    # local expectation at each site, then sum
    Sx = 0.0 + 0.0j
    Sy = 0.0 + 0.0j
    Sz = 0.0 + 0.0j
    for i in range(N):
        v = psi2[i:i+1, :].T  # (2,1)
        vdag = np.conj(v).T
        Sx += (vdag @ (sx @ v))[0, 0]
        Sy += (vdag @ (sy @ v))[0, 0]
        Sz += (vdag @ (sz @ v))[0, 0]

    return float(Sx.real), float(Sy.real), float(Sz.real)


# =============================================================================
# Main simulation driver
# =============================================================================


def run_spinor_precipitating_event(cfg: SpinorConfig) -> Dict[str, Any]:
    """
    Run the spinor + gauge-link precipitating event and return data dict.
    Also writes npz + summary.txt to disk.
    """
    print("Spinor precipitating event config:")
    print(asdict(cfg))
    print()

    # Geometry
    D_lr, D_prop, graph_dist = load_metrics(cfg.asset_dir, cfg.metrics_file)
    if graph_dist is None:
        raise ValueError("graph_dist not found in metrics; cannot build adjacency.")
    X, used_smooth = load_embedding(
        cfg.asset_dir, cfg.embedding_smooth_file, cfg.embedding_file
    )

    N = X.shape[0]
    spin_dim = cfg.spin_dim
    D = N * spin_dim

    print(f"N sites = {N}, spin_dim = {spin_dim}, global dimension D = {D}")
    print(f"Using smoothed embedding? {used_smooth}")
    print()

    adjacency = build_adjacency_from_graph_dist(graph_dist)
    print("Adjacency (first few nodes):")
    max_print = min(N, 8)
    for i in range(max_print):
        print(f"  {i}: {adjacency[i]}")
    if N > max_print:
        print(f"  ... ({N - max_print} more nodes)")
    print()

    # Random generator
    rng = np.random.default_rng(cfg.seed)

    # Spin Hamiltonian pieces
    Omega = np.array([cfg.Omega_x, cfg.Omega_y, cfg.Omega_z], dtype=float)
    H_loc_spin = local_spin_hamiltonians(X, Omega, cfg.spin_coupling, spin_dim)
    links_u1 = build_u1_links(
        X, adjacency, Omega, cfg.gauge_strength, cfg.gauge_curl
    )

    H_hot = build_spinor_hot_hamiltonian(N, spin_dim, cfg.hot_strength, rng)
    H_order = build_spinor_order_hamiltonian(N, spin_dim, adjacency, cfg.J_order)
    # reuse J_order as gauge hopping strength (can separate later if desired)
    H_gauge = build_spinor_gauge_hamiltonian(N, spin_dim, adjacency, links_u1, cfg.J_order)
    H_spin = build_spinor_spin_hamiltonian(N, spin_dim, H_loc_spin)

    print("Hamiltonian blocks:")
    print("  H_hot   shape:", H_hot.shape)
    print("  H_order shape:", H_order.shape)
    print("  H_spin  shape:", H_spin.shape)
    print("  H_gauge shape:", H_gauge.shape)
    print()

    # Initial state
    if cfg.init_random_spinor:
        # random complex spinors at each site, then normalize global state
        psi = (rng.normal(size=(N, spin_dim)) +
               1j * rng.normal(size=(N, spin_dim)))
        psi_flat = psi.reshape(D)
        psi_flat /= np.linalg.norm(psi_flat)
    else:
        psi_flat = np.zeros(D, dtype=np.complex128)
        # localized spin-up at site 0
        psi_flat[0] = 1.0

    print("Initial state prepared. Norm:", np.linalg.norm(psi_flat))
    print()

    # Time grid
    T = cfg.n_steps + 1
    times = np.linspace(0.0, cfg.t_max, T)
    dt = times[1] - times[0] if T > 1 else cfg.t_max

    psi_t = np.zeros((T, N, spin_dim), dtype=np.complex128)

    psi_t[0] = psi_flat.reshape(N, spin_dim)
    Sx_hist = np.zeros(T)
    Sy_hist = np.zeros(T)
    Sz_hist = np.zeros(T)

    Sx, Sy, Sz = spin_expectation(psi_t[0], spin_dim)
    Sx_hist[0], Sy_hist[0], Sz_hist[0] = Sx, Sy, Sz

    print("Starting time evolution...")
    psi = psi_flat.copy()
    for n in range(1, T):
        t = times[n - 1]
        psi = rk4_step_spinor(t, psi, dt, H_hot, H_order, H_spin, H_gauge, cfg)
        psi_t[n] = psi.reshape(N, spin_dim)

        Sx, Sy, Sz = spin_expectation(psi_t[n], spin_dim)
        Sx_hist[n], Sy_hist[n], Sz_hist[n] = Sx, Sy, Sz

        if n % max(1, T // 10) == 0:
            print(
                f"  step {n}/{T-1}, t={times[n]:.3f}, "
                f"||psi||={np.linalg.norm(psi):.6f}, "
                f"S=({Sx:+.3f},{Sy:+.3f},{Sz:+.3f})"
            )

    # Output
    out_dir = os.path.join(cfg.asset_dir, cfg.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, "spinor_precipitating_event_timeseries.npz")
    np.savez(
        npz_path,
        times=times,
        psi_t=psi_t,
        X=X,
        Sx_hist=Sx_hist,
        Sy_hist=Sy_hist,
        Sz_hist=Sz_hist,
        config=asdict(cfg),
    )

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Spinor Precipitating Event\n")
        f.write("==========================\n\n")
        f.write("Config:\n")
        f.write(repr(asdict(cfg)) + "\n\n")
        f.write(f"N sites = {N}, spin_dim = {spin_dim}, D = {D}\n")
        f.write(f"Using smoothed embedding: {used_smooth}\n\n")
        f.write("Hamiltonian pieces:\n")
        f.write(f"  hot_strength   = {cfg.hot_strength}\n")
        f.write(f"  J_order        = {cfg.J_order}\n")
        f.write(f"  spin_coupling  = {cfg.spin_coupling}\n")
        f.write(f"  gauge_strength = {cfg.gauge_strength}\n")
        f.write(f"  gauge_curl     = {cfg.gauge_curl}\n")
        f.write(f"  Omega          = ({cfg.Omega_x}, {cfg.Omega_y}, {cfg.Omega_z})\n\n")
        f.write("Schedule:\n")
        f.write(f"  t_max          = {cfg.t_max}\n")
        f.write(f"  n_steps        = {cfg.n_steps}\n")
        f.write(f"  quench_fraction= {cfg.quench_fraction}\n")
        f.write(f"  t_quench       = {cfg.quench_fraction * cfg.t_max}\n\n")
        f.write("Sample spin expectations:\n")
        sample_indices = np.linspace(0, T-1, min(12, T), dtype=int)
        for idx in sample_indices:
            f.write(
                f"  t={times[idx]:7.3f}  "
                f"Sx={Sx_hist[idx]:+7.3f}  "
                f"Sy={Sy_hist[idx]:+7.3f}  "
                f"Sz={Sz_hist[idx]:+7.3f}\n"
            )

    print("\nSimulation complete.")
    print("Saved:", npz_path)
    print("Saved:", summary_path)

    return {
        "times": times,
        "psi_t": psi_t,
        "X": X,
        "Sx_hist": Sx_hist,
        "Sy_hist": Sy_hist,
        "Sz_hist": Sz_hist,
        "config": asdict(cfg),
    }


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Spinor + U(1) gauge-link version of the precipitating event "
            "quench+cooling lab on an emergent LR geometry."
        )
    )

    parser.add_argument(
        "--asset-dir",
        type=str,
        default="cubic_universe_L3",
        help="Directory with lr_metrics.npz and LR embedding files.",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="lr_metrics.npz",
        help="Metrics file name inside asset-dir.",
    )
    parser.add_argument(
        "--embedding-smooth-file",
        type=str,
        default="lr_embedding_3d_smooth.npz",
        help="Preferred smoothed embedding file name.",
    )
    parser.add_argument(
        "--embedding-file",
        type=str,
        default="lr_embedding_3d.npz",
        help="Fallback embedding file name.",
    )

    parser.add_argument(
        "--hot-strength",
        type=float,
        default=1.0,
        help="Strength for disordered H_hot.",
    )
    parser.add_argument(
        "--J-order",
        type=float,
        default=1.0,
        help="Gauge-neutral hopping strength.",
    )
    parser.add_argument(
        "--spin-coupling",
        type=float,
        default=0.5,
        help="Coupling strength for B·sigma spin term.",
    )
    parser.add_argument(
        "--gauge-strength",
        type=float,
        default=0.8,
        help="Magnitude of U(1) link phases.",
    )
    parser.add_argument(
        "--gauge-curl",
        type=float,
        default=0.3,
        help="Geometric factor for A_ij from Omega·(m×e).",
    )

    parser.add_argument(
        "--Omega-x",
        type=float,
        default=0.0,
        help="Rotation vector Omega_x.",
    )
    parser.add_argument(
        "--Omega-y",
        type=float,
        default=0.0,
        help="Rotation vector Omega_y.",
    )
    parser.add_argument(
        "--Omega-z",
        type=float,
        default=1.0,
        help="Rotation vector Omega_z (strong spin about z).",
    )

    parser.add_argument(
        "--t-max",
        type=float,
        default=10.0,
        help="Total evolution time.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2000,
        help="Number of RK4 steps.",
    )
    parser.add_argument(
        "--quench-fraction",
        type=float,
        default=0.4,
        help="Fraction of t_max for the quench ramp 0→1.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for hot Hamiltonian and initial state.",
    )
    parser.add_argument(
        "--no-random-spinor",
        action="store_true",
        help="If set, start from localized spin-up at site 0.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="spinor_precipitating_event_out",
        help="Output subdirectory inside asset-dir.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    cfg = SpinorConfig(
        asset_dir=args.asset_dir,
        metrics_file=args.metrics_file,
        embedding_smooth_file=args.embedding_smooth_file,
        embedding_file=args.embedding_file,
        hot_strength=args.hot_strength,
        J_order=args.J_order,
        spin_coupling=args.spin_coupling,
        gauge_strength=args.gauge_strength,
        gauge_curl=args.gauge_curl,
        Omega_x=args.Omega_x,
        Omega_y=args.Omega_y,
        Omega_z=args.Omega_z,
        t_max=args.t_max,
        n_steps=args.n_steps,
        quench_fraction=args.quench_fraction,
        seed=args.seed,
        init_random_spinor=not args.no_random_spinor,
        output_dir=args.output_dir,
    )
    run_spinor_precipitating_event(cfg)


if __name__ == "__main__":
    main()
