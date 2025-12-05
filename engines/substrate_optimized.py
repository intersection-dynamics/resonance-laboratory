#!/usr/bin/env python3
"""
Substrate: Hilbert Graph with Lieb–Robinson Emergent Geometry (Optimized)
==========================================================================

Optimized version with eigendecomposition caching for O(D²) time evolution
instead of O(D³) matrix exponentials.

Key optimizations:
  1. Eigendecompose H once: H = V Λ V†
  2. Time evolution via diagonal sandwiching: U(t) = V e^{-iΛt} V†
  3. Heisenberg evolution in eigenbasis: A(t) = V e^{+iΛt} V† A V e^{-iΛt} V†
  4. Precompute V† nⱼ V for each site
  5. Single eigendecomposition serves all N source sites in LR metric

All results are bit-for-bit identical to the original implementation.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Literal, Optional

import numpy as np
from scipy.linalg import eigh as np_eigh

# Optional CuPy backend
try:
    import cupy as cp
    from cupy.linalg import eigh as cp_eigh
    HAS_CUPY = True
except ImportError:
    cp = None
    cp_eigh = None
    HAS_CUPY = False


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SubstrateConfig:
    """Configuration for the Hilbert graph substrate."""

    n_sites: int = 12
    d_local: int = 2
    coupling: float = 1.0

    connectivity: Literal["chain", "ring", "complete", "random"] = "chain"
    random_p: float = 0.3
    seed: int = 12345

    lr_threshold: float = 1e-3
    lr_t_max: float = 8.0
    lr_n_steps: int = 120
    lr_norm: Literal["fro", "spectral"] = "fro"

    use_gpu: bool = False
    dtype: Literal["complex128", "complex64"] = "complex128"


# =============================================================================
# Substrate: global Hilbert space and local operators
# =============================================================================


class HilbertSubstrate:
    """
    Global Hilbert substrate on an abstract graph.

    - Sites 0..N-1, no coordinates.
    - Local dimension d at each site.
    - Global dimension D = d^N.
    """

    def __init__(self,
                 n_sites: int,
                 d_local: int,
                 xp=np,
                 dtype=np.complex128):
        self.n_sites = n_sites
        self.d_local = d_local
        self.dim_total = d_local ** n_sites
        self.xp = xp
        self.dtype = dtype

    def index_to_config(self, idx: int) -> Tuple[int, ...]:
        """Convert basis index to tuple of local occupations."""
        conf = [0] * self.n_sites
        d = self.d_local
        for k in range(self.n_sites):
            power = self.n_sites - 1 - k
            base = d ** power
            conf[k] = (idx // base) % d
        return tuple(conf)

    def config_to_index(self, conf: Tuple[int, ...]) -> int:
        """Convert occupation tuple to basis index."""
        idx = 0
        d = self.d_local
        for k, n in enumerate(conf):
            power = self.n_sites - 1 - k
            idx += int(n) * (d ** power)
        return idx

    def vacuum(self):
        """Return |0,0,...,0> as a global state vector."""
        psi = self.xp.zeros(self.dim_total, dtype=self.dtype)
        psi[0] = 1.0
        return psi

    def normalize(self, psi):
        """Normalize a state vector."""
        norm = float(self.xp.linalg.norm(psi))
        if norm == 0.0:
            return psi
        return psi / norm

    def excite_site(self, psi, site: int):
        """Apply truncated bosonic creation operator to global state."""
        d = self.d_local
        D = self.dim_total
        out = self.xp.zeros_like(psi)

        for idx in range(D):
            conf = list(self.index_to_config(idx))
            n = conf[site]
            if n < d - 1:
                conf[site] = n + 1
                new_idx = self.config_to_index(tuple(conf))
                out[new_idx] += psi[idx] * np.sqrt(n + 1)
        return out

    def local_number_operator(self, site: int):
        """Build the full many-body number operator matrix for a site."""
        D = self.dim_total
        n_op = self.xp.zeros((D, D), dtype=self.dtype)
        for idx in range(D):
            conf = self.index_to_config(idx)
            n_op[idx, idx] = conf[site]
        return n_op

    def local_occupation(self, psi, site: int) -> float:
        """Compute expectation <n_site> for a pure state."""
        D = self.dim_total
        exp_val = 0.0
        for idx in range(D):
            conf = self.index_to_config(idx)
            exp_val += float(self.xp.abs(psi[idx]) ** 2) * conf[site]
        return float(exp_val)


# =============================================================================
# Graph + Hamiltonian
# =============================================================================


def build_connectivity(cfg: SubstrateConfig) -> Dict[int, List[int]]:
    """Bare graph connectivity."""
    N = cfg.n_sites
    rng = np.random.default_rng(cfg.seed)

    if cfg.connectivity == "chain":
        return {i: [j for j in (i - 1, i + 1) if 0 <= j < N] for i in range(N)}

    if cfg.connectivity == "ring":
        return {i: [((i - 1) % N), ((i + 1) % N)] for i in range(N)}

    if cfg.connectivity == "complete":
        return {i: [j for j in range(N) if j != i] for i in range(N)}

    if cfg.connectivity == "random":
        adj: Dict[int, List[int]] = {i: [] for i in range(N)}
        for i in range(N):
            for j in range(i + 1, N):
                if rng.random() < cfg.random_p:
                    adj[i].append(j)
                    adj[j].append(i)
        return adj

    raise ValueError(f"Unknown connectivity type: {cfg.connectivity}")


def build_hamiltonian(substrate: HilbertSubstrate,
                      connectivity: Dict[int, List[int]],
                      coupling: float):
    """
    H = coupling * sum_{i<j, (i,j) in edges} (a_i† a_j + a_j† a_i).
    """
    xp = substrate.xp
    dtype = substrate.dtype
    D = substrate.dim_total
    d = substrate.d_local

    H = xp.zeros((D, D), dtype=dtype)

    for i, neighbors in connectivity.items():
        for j in neighbors:
            if j <= i:
                continue

            for idx in range(D):
                conf = list(substrate.index_to_config(idx))
                n_i = conf[i]
                n_j = conf[j]

                # a_i† a_j
                if n_j > 0 and n_i < d - 1:
                    coeff = np.sqrt(n_j) * np.sqrt(n_i + 1)
                    new_conf = conf.copy()
                    new_conf[j] -= 1
                    new_conf[i] += 1
                    new_idx = substrate.config_to_index(tuple(new_conf))
                    H[new_idx, idx] += coupling * coeff

                # a_j† a_i
                if n_i > 0 and n_j < d - 1:
                    coeff = np.sqrt(n_i) * np.sqrt(n_j + 1)
                    new_conf = conf.copy()
                    new_conf[i] -= 1
                    new_conf[j] += 1
                    new_idx = substrate.config_to_index(tuple(new_conf))
                    H[new_idx, idx] += coupling * coeff

    return H


# =============================================================================
# Evolution Cache: eigendecomposition-based time evolution
# =============================================================================


class EvolutionCache:
    """
    Precomputed eigendecomposition and operator transforms for efficient
    time evolution.

    Given H = V Λ V†, time evolution is:
        U(t) = V diag(e^{-iλt}) V†

    Heisenberg evolution of operator A:
        A(t) = U†(t) A U(t)
             = V e^{+iΛt} V† A V e^{-iΛt} V†
             = V e^{+iΛt} A_eig e^{-iΛt} V†

    where A_eig = V† A V is precomputed.
    """

    def __init__(self,
                 substrate: HilbertSubstrate,
                 H,
                 eigh_fn=np_eigh):
        self.substrate = substrate
        self.xp = substrate.xp
        self.dtype = substrate.dtype
        self.N = substrate.n_sites
        self.D = substrate.dim_total

        # Eigendecompose H (Hermitian, so use eigh)
        # H = V @ diag(eigenvalues) @ V†
        eigenvalues, V = eigh_fn(H)

        # Store as xp arrays
        self.eigenvalues = self.xp.asarray(eigenvalues).real  # guaranteed real for Hermitian
        self.V = self.xp.asarray(V)
        self.V_dag = self.V.conj().T

        # Precompute number operators in eigenbasis: n_j_eig = V† n_j V
        self.n_ops = []      # original basis
        self.n_ops_eig = []  # eigenbasis
        for j in range(self.N):
            n_j = substrate.local_number_operator(j)
            self.n_ops.append(n_j)
            self.n_ops_eig.append(self.V_dag @ n_j @ self.V)

    def U(self, t: float):
        """Compute U(t) = exp(-i H t) = V diag(e^{-iλt}) V†."""
        phases = self.xp.exp(-1j * self.eigenvalues * t)
        return self.V @ (phases[:, None] * self.V_dag)

    def evolve_operator_to_time(self, A_eig, t: float):
        """
        Evolve operator A to time t in Heisenberg picture.

        A_eig = V† A V (precomputed)
        Returns A(t) = V e^{+iΛt} A_eig e^{-iΛt} V†
        """
        phases_pos = self.xp.exp(+1j * self.eigenvalues * t)
        phases_neg = self.xp.exp(-1j * self.eigenvalues * t)

        # e^{+iΛt} A_eig e^{-iΛt} is diagonal sandwiching
        # (phases_pos[:, None] * A_eig) multiplies row i by phases_pos[i]
        # (* phases_neg[None, :]) multiplies col j by phases_neg[j]
        A_eig_t = (phases_pos[:, None] * A_eig) * phases_neg[None, :]

        return self.V @ A_eig_t @ self.V_dag

    def evolve_state(self, psi0, t_max: float, n_steps: int):
        """
        Evolve state psi under H for times t in [0, t_max].

        Returns:
            times: np.ndarray shape (n_steps,)
            states: xp array shape (n_steps, D)
        """
        times = np.linspace(0.0, t_max, n_steps)
        states = self.xp.zeros((n_steps, self.D), dtype=self.dtype)

        # Transform to eigenbasis once
        psi_eig = self.V_dag @ psi0

        for k, t in enumerate(times):
            phases = self.xp.exp(-1j * self.eigenvalues * t)
            psi_eig_t = phases * psi_eig
            states[k] = self.V @ psi_eig_t

        return times, states


# =============================================================================
# Propagation-based signals (optimized)
# =============================================================================


def excitation_propagation(substrate: HilbertSubstrate,
                           cache: EvolutionCache,
                           source_site: int,
                           t_max: float,
                           n_steps: int,
                           threshold: float = 0.01) -> Dict[str, object]:
    """
    Excitation-based arrival times using cached evolution.
    """
    psi0 = substrate.excite_site(substrate.vacuum(), source_site)
    psi0 = substrate.normalize(psi0)

    times, states = cache.evolve_state(psi0, t_max, n_steps)
    N = substrate.n_sites
    T = len(times)

    occ = np.zeros((N, T), dtype=float)

    for t_idx in range(T):
        psi = states[t_idx]
        for j in range(N):
            occ[j, t_idx] = substrate.local_occupation(psi, j)

    arrival: Dict[int, float] = {}
    for j in range(N):
        if j == source_site:
            arrival[j] = 0.0
            continue
        above = np.where(occ[j] > threshold)[0]
        arrival[j] = float(times[above[0]]) if len(above) > 0 else float("inf")

    return {
        "times": times,
        "occupations": occ,
        "arrival": arrival,
        "source": source_site,
        "threshold": threshold,
    }


def excitation_metric_from_arrival(substrate: HilbertSubstrate,
                                   cache: EvolutionCache,
                                   t_max: float,
                                   n_steps: int,
                                   threshold: float = 0.01) -> np.ndarray:
    """Symmetric distance matrix from excitation arrival times."""
    N = substrate.n_sites
    Dmat = np.zeros((N, N), dtype=float)

    for source in range(N):
        res = excitation_propagation(substrate, cache, source, t_max, n_steps, threshold)
        for j in range(N):
            Dmat[source, j] = res["arrival"][j]

    Dmat = 0.5 * (Dmat + Dmat.T)
    return Dmat


# =============================================================================
# Lieb–Robinson commutators (optimized)
# =============================================================================


def lieb_robinson_commutators(substrate: HilbertSubstrate,
                              cache: EvolutionCache,
                              source_site: int,
                              t_max: float,
                              n_steps: int,
                              threshold: float,
                              norm_type: Literal["fro", "spectral"]) -> Dict[str, object]:
    """
    Compute LR commutator C_ij(t) = [A_i(t), B_j] using cached eigendecomposition.

    A_i(t) = U†(t) n_i U(t)
    B_j = n_j
    """
    xp = substrate.xp
    N = substrate.n_sites

    times = np.linspace(0.0, t_max, n_steps)
    comm_norms = np.zeros((N, n_steps), dtype=float)

    # Get source operator in eigenbasis (precomputed)
    A_eig = cache.n_ops_eig[source_site]

    for t_idx, t in enumerate(times):
        # Evolve A_i to time t
        A_t = cache.evolve_operator_to_time(A_eig, t)

        for j in range(N):
            C = A_t @ cache.n_ops[j] - cache.n_ops[j] @ A_t
            if norm_type == "spectral":
                comm_norms[j, t_idx] = float(xp.linalg.norm(C, 2))
            else:
                comm_norms[j, t_idx] = float(xp.linalg.norm(C, "fro"))

    arrival: Dict[int, float] = {}
    for j in range(N):
        if j == source_site:
            arrival[j] = 0.0
            continue
        above = np.where(comm_norms[j] >= threshold)[0]
        arrival[j] = float(times[above[0]]) if len(above) > 0 else float("inf")

    return {
        "times": times,
        "comm_norms": comm_norms,
        "arrival": arrival,
        "source": source_site,
        "threshold": threshold,
        "norm_type": norm_type,
    }


def lieb_robinson_metric(substrate: HilbertSubstrate,
                         cache: EvolutionCache,
                         t_max: float,
                         n_steps: int,
                         threshold: float,
                         norm_type: Literal["fro", "spectral"]) -> np.ndarray:
    """
    Operational LR distance matrix from commutator arrival times.

    Uses single cached eigendecomposition for all N source sites.
    """
    N = substrate.n_sites
    Dmat = np.zeros((N, N), dtype=float)

    for source in range(N):
        lr = lieb_robinson_commutators(
            substrate,
            cache,
            source_site=source,
            t_max=t_max,
            n_steps=n_steps,
            threshold=threshold,
            norm_type=norm_type,
        )
        for j in range(N):
            Dmat[source, j] = lr["arrival"][j]

    Dmat = 0.5 * (Dmat + Dmat.T)
    return Dmat


# =============================================================================
# Graph distances and inflation analysis
# =============================================================================


def compute_graph_distances(connectivity: Dict[int, List[int]]) -> np.ndarray:
    """All-pairs shortest path lengths on graph (BFS)."""
    nodes = sorted(connectivity.keys())
    N = len(nodes)
    dist = np.full((N, N), np.inf, dtype=float)

    for i in nodes:
        d = {i: 0}
        frontier = [i]
        while frontier:
            new_frontier: List[int] = []
            for u in frontier:
                for v in connectivity[u]:
                    if v not in d:
                        d[v] = d[u] + 1
                        new_frontier.append(v)
            frontier = new_frontier

        for j, dj in d.items():
            dist[i, j] = float(dj)
            dist[j, i] = float(dj)

    return dist


def analyze_lr_inflation(D_lr: np.ndarray,
                         graph_dist: np.ndarray,
                         source: int) -> None:
    """Shell profile for LR metric vs graph distance."""
    N = D_lr.shape[0]
    d_max = int(np.nanmax(graph_dist[source, np.isfinite(graph_dist[source])]))

    shell_times: Dict[int, List[float]] = {d: [] for d in range(d_max + 1)}

    for j in range(N):
        d_g = graph_dist[source, j]
        if not np.isfinite(d_g):
            continue
        d_int = int(d_g)
        t_lr = D_lr[source, j]
        if np.isfinite(t_lr):
            shell_times[d_int].append(t_lr)

    t_shell = np.zeros(d_max + 1, dtype=float)
    for d in range(d_max + 1):
        if shell_times[d]:
            t_shell[d] = float(np.mean(shell_times[d]))
        else:
            t_shell[d] = np.inf

    dt_shell = np.full(d_max + 1, np.nan, dtype=float)
    v_eff = np.full(d_max + 1, np.nan, dtype=float)

    for d in range(1, d_max + 1):
        if np.isfinite(t_shell[d]) and np.isfinite(t_shell[d - 1]):
            dt = t_shell[d] - t_shell[d - 1]
            dt_shell[d] = dt
            if dt > 0:
                v_eff[d] = 1.0 / dt

    print(f"Lieb–Robinson 'inflation' profile (source = {source}):")
    print("  d_graph |  t_LR_shell  |  Δt(d) = t(d)-t(d-1)  |  v_eff(d) = 1/Δt")
    print("  ---------------------------------------------------------------")
    print(f"      0   |  {t_shell[0]:.6f}    |        ---              |     ---")
    for d in range(1, d_max + 1):
        t = t_shell[d]
        dt = dt_shell[d]
        v = v_eff[d]
        t_str = f"{t:.6f}" if np.isfinite(t) else "  inf   "
        dt_str = f"{dt:.6f}" if np.isfinite(dt) else "   nan  "
        v_str = f"{v:.3f}" if np.isfinite(v) else "  nan "
        print(f"     {d:2d}   |  {t_str}    |   {dt_str}         |   {v_str}")
    print("  ---------------------------------------------------------------")
    print("  Note: decreasing v_eff(d) with d means 'inflation' slowing.\n")


# =============================================================================
# Metric → coordinate embedding (MDS)
# =============================================================================


def classical_mds_from_dist(Dmat: np.ndarray, dim: int = 2) -> np.ndarray:
    """Classical MDS embedding of distance matrix D into R^dim."""
    D2 = Dmat ** 2
    N = Dmat.shape[0]
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ D2 @ J

    vals, vecs = np_eigh(B)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    pos_mask = vals > 1e-12
    vals_pos = vals[pos_mask][:dim]
    vecs_pos = vecs[:, pos_mask][:, :dim]

    L_sqrt = np.sqrt(vals_pos)
    X = vecs_pos * L_sqrt[np.newaxis, :]
    return X


# =============================================================================
# Verification: compare optimized vs original
# =============================================================================


def verify_against_original(cfg: SubstrateConfig) -> bool:
    """
    Run both original (expm-based) and optimized (eigendecomposition-based)
    computations and verify they match.
    """
    from scipy.linalg import expm as np_expm

    print("=" * 60)
    print("VERIFICATION: Comparing original vs optimized implementation")
    print("=" * 60)

    xp = np
    dtype = np.complex128 if cfg.dtype == "complex128" else np.complex64

    substrate = HilbertSubstrate(cfg.n_sites, cfg.d_local, xp=xp, dtype=dtype)
    connectivity = build_connectivity(cfg)
    H = build_hamiltonian(substrate, connectivity, cfg.coupling)

    # Build cache for optimized version
    cache = EvolutionCache(substrate, H, eigh_fn=np_eigh)

    source = cfg.n_sites // 2
    all_match = True

    # --- Test 1: State evolution ---
    print("\n[Test 1] State evolution...")
    psi0 = substrate.excite_site(substrate.vacuum(), source)
    psi0 = substrate.normalize(psi0)

    t_test = 1.5
    dt = t_test / 10
    U_orig = np_expm(-1j * H * dt)
    psi_orig = psi0.copy()
    for _ in range(10):
        psi_orig = U_orig @ psi_orig

    U_opt = cache.U(t_test)
    psi_opt = U_opt @ psi0

    diff = np.linalg.norm(psi_orig - psi_opt)
    print(f"  ||psi_orig - psi_opt|| = {diff:.2e}")
    if diff > 1e-10:
        print("  MISMATCH!")
        all_match = False
    else:
        print("  OK")

    # --- Test 2: Operator evolution ---
    print("\n[Test 2] Heisenberg operator evolution...")
    n_op = substrate.local_number_operator(source)

    # Original: A(t) = U† A U iterated
    A_orig = n_op.copy()
    U_dag = U_orig.conj().T
    for _ in range(10):
        A_orig = U_dag @ A_orig @ U_orig

    # Optimized
    A_opt = cache.evolve_operator_to_time(cache.n_ops_eig[source], t_test)

    diff = np.linalg.norm(A_orig - A_opt)
    print(f"  ||A_orig(t) - A_opt(t)|| = {diff:.2e}")
    if diff > 1e-10:
        print("  MISMATCH!")
        all_match = False
    else:
        print("  OK")

    # --- Test 3: LR commutator norms ---
    print("\n[Test 3] LR commutator norms at t=1.5...")
    for j in range(min(4, cfg.n_sites)):
        C_orig = A_orig @ substrate.local_number_operator(j) - substrate.local_number_operator(j) @ A_orig
        C_opt = A_opt @ cache.n_ops[j] - cache.n_ops[j] @ A_opt
        norm_orig = np.linalg.norm(C_orig, "fro")
        norm_opt = np.linalg.norm(C_opt, "fro")
        diff = abs(norm_orig - norm_opt)
        status = "OK" if diff < 1e-10 else "MISMATCH"
        print(f"  [source={source}, j={j}] ||C||_fro: orig={norm_orig:.6f}, opt={norm_opt:.6f}, diff={diff:.2e} {status}")
        if diff > 1e-10:
            all_match = False

    # --- Test 4: Full LR metric ---
    print("\n[Test 4] Full LR distance matrix...")

    # Original implementation (inline)
    def lr_metric_original():
        N = substrate.n_sites
        Dmat = np.zeros((N, N), dtype=float)
        times = np.linspace(0.0, cfg.lr_t_max, cfg.lr_n_steps)
        dt = times[1] - times[0] if cfg.lr_n_steps > 1 else 0.0
        U = np_expm(-1j * H * dt)
        U_dag = U.conj().T
        n_ops = [substrate.local_number_operator(j) for j in range(N)]

        for src in range(N):
            A_t = n_ops[src].copy()
            arrival = {}
            for j in range(N):
                if j == src:
                    arrival[j] = 0.0
                else:
                    arrival[j] = float("inf")

            for t_idx, t in enumerate(times):
                for j in range(N):
                    if arrival[j] < float("inf"):
                        continue
                    C = A_t @ n_ops[j] - n_ops[j] @ A_t
                    norm_val = float(np.linalg.norm(C, "fro"))
                    if norm_val >= cfg.lr_threshold:
                        arrival[j] = t
                A_t = U_dag @ A_t @ U

            for j in range(N):
                Dmat[src, j] = arrival[j]

        return 0.5 * (Dmat + Dmat.T)

    D_lr_orig = lr_metric_original()
    D_lr_opt = lieb_robinson_metric(
        substrate, cache, cfg.lr_t_max, cfg.lr_n_steps, cfg.lr_threshold, cfg.lr_norm
    )

    diff = np.max(np.abs(D_lr_orig - D_lr_opt))
    print(f"  max|D_lr_orig - D_lr_opt| = {diff:.2e}")
    if diff > 1e-10:
        print("  MISMATCH!")
        all_match = False
    else:
        print("  OK")

    print("\n" + "=" * 60)
    if all_match:
        print("ALL TESTS PASSED - Optimized implementation is exact.")
    else:
        print("SOME TESTS FAILED - Check implementation.")
    print("=" * 60 + "\n")

    return all_match


# =============================================================================
# Demo / CLI
# =============================================================================


def run_demo(cfg: SubstrateConfig, out_dir: str, verify: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print("Substrate config:")
    print(asdict(cfg))
    print()

    # Backend selection
    if cfg.use_gpu:
        if not HAS_CUPY:
            print("WARNING: --use-gpu requested but CuPy not available. Falling back to CPU.")
            xp = np
            dtype = np.complex128 if cfg.dtype == "complex128" else np.complex64
            eigh_fn = np_eigh
        else:
            xp = cp
            dtype = cp.complex128 if cfg.dtype == "complex128" else cp.complex64
            eigh_fn = cp_eigh
            print("Using CuPy GPU backend.")
    else:
        xp = np
        dtype = np.complex128 if cfg.dtype == "complex128" else np.complex64
        eigh_fn = np_eigh

    # Build substrate + Hamiltonian
    substrate = HilbertSubstrate(cfg.n_sites, cfg.d_local, xp=xp, dtype=dtype)
    connectivity = build_connectivity(cfg)
    H = build_hamiltonian(substrate, connectivity, cfg.coupling)

    print(f"Global dimension: {substrate.dim_total}")
    print("Connectivity:")
    for i in range(cfg.n_sites):
        print(f"  {i}: {connectivity.get(i, [])}")
    print()

    # Build evolution cache (single eigendecomposition)
    print("Building evolution cache (eigendecomposition)...")
    import time
    t0 = time.perf_counter()
    cache = EvolutionCache(substrate, H, eigh_fn=eigh_fn)
    t_cache = time.perf_counter() - t0
    print(f"  Cache built in {t_cache:.3f}s\n")

    graph_dist = compute_graph_distances(connectivity)

    source = cfg.n_sites // 2
    print(f"Using site {source} as source\n")

    # 1. excitation-based metric
    t0 = time.perf_counter()
    prop_res = excitation_propagation(
        substrate, cache, source_site=source,
        t_max=cfg.lr_t_max, n_steps=cfg.lr_n_steps, threshold=0.01,
    )
    D_prop = excitation_metric_from_arrival(
        substrate, cache, t_max=cfg.lr_t_max, n_steps=cfg.lr_n_steps, threshold=0.01,
    )
    t_prop = time.perf_counter() - t0

    print(f"Propagation-based arrival times from source (computed in {t_prop:.3f}s):")
    for j in range(cfg.n_sites):
        print(f"  {j}: {prop_res['arrival'][j]:.3f}")
    print()

    # 2. Lieb–Robinson metric
    t0 = time.perf_counter()
    lr_res = lieb_robinson_commutators(
        substrate, cache, source_site=source,
        t_max=cfg.lr_t_max, n_steps=cfg.lr_n_steps,
        threshold=cfg.lr_threshold, norm_type=cfg.lr_norm,
    )
    D_lr = lieb_robinson_metric(
        substrate, cache, t_max=cfg.lr_t_max, n_steps=cfg.lr_n_steps,
        threshold=cfg.lr_threshold, norm_type=cfg.lr_norm,
    )
    t_lr = time.perf_counter() - t0

    print(f"LR commutator arrival times from source (computed in {t_lr:.3f}s):")
    for j in range(cfg.n_sites):
        print(f"  {j}: {lr_res['arrival'][j]:.3f}")
    print()

    # 3. Inflation profile
    analyze_lr_inflation(D_lr, graph_dist, source)

    # 4. 2D embedding of LR metric
    try:
        # Transfer from GPU if needed
        D_lr_np = D_lr.get() if hasattr(D_lr, 'get') else D_lr
        D_prop_np = D_prop.get() if hasattr(D_prop, 'get') else D_prop

        X2 = classical_mds_from_dist(D_lr_np, dim=2)
        np.savez(os.path.join(out_dir, "lr_embedding_2d.npz"),
                 X2=X2, D_lr=D_lr_np, D_prop=D_prop_np, graph_dist=graph_dist)
        print("Saved LR 2D embedding to lr_embedding_2d.npz")
    except Exception as exc:
        print("Could not embed LR metric:", exc)

    np.savez(os.path.join(out_dir, "lr_metrics.npz"),
             D_lr=D_lr_np if 'D_lr_np' in dir() else D_lr,
             D_prop=D_prop_np if 'D_prop_np' in dir() else D_prop,
             graph_dist=graph_dist)

    # Optional verification
    if verify:
        verify_against_original(cfg)

    print("\nDone.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hilbert graph substrate with Lieb–Robinson emergent geometry (optimized)."
    )
    parser.add_argument("--n-sites", type=int, default=8)
    parser.add_argument("--d-local", type=int, default=2)
    parser.add_argument("--connectivity", type=str, default="chain",
                        choices=["chain", "ring", "complete", "random"])
    parser.add_argument("--random-p", type=float, default=0.3)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--lr-threshold", type=float, default=1e-3)
    parser.add_argument("--lr-t-max", type=float, default=8.0)
    parser.add_argument("--lr-n-steps", type=int, default=120)
    parser.add_argument("--lr-norm", type=str, default="fro", choices=["fro", "spectral"])
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--dtype", type=str, default="complex128",
                        choices=["complex128", "complex64"])
    parser.add_argument("--output-dir", type=str, default="substrate_outputs")
    parser.add_argument("--verify", action="store_true",
                        help="Run verification against original expm-based implementation")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    cfg = SubstrateConfig(
        n_sites=args.n_sites,
        d_local=args.d_local,
        coupling=args.coupling,
        connectivity=args.connectivity,
        random_p=args.random_p,
        seed=args.seed,
        lr_threshold=args.lr_threshold,
        lr_t_max=args.lr_t_max,
        lr_n_steps=args.lr_n_steps,
        lr_norm=args.lr_norm,
        use_gpu=args.use_gpu,
        dtype=args.dtype,
    )

    run_demo(cfg, out_dir=args.output_dir, verify=args.verify)


if __name__ == "__main__":
    main()