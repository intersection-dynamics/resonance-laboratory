#!/usr/bin/env python3
"""
Batched GPU-oriented Hilbert substrate engine for proton pointer experiments.

- Uses CuPy on GPU if available, NumPy otherwise.
- Stores all node states in a single (n_nodes, dim) array.
- Builds local Hamiltonians H[n] in a batched array (n_nodes, dim, dim).
- Applies unitary evolution to all nodes in one batched GPU operation.

Exposes:
    Config
    Substrate
    run_simulation(config, n_steps, record_every)

And maintains compatibility with existing pointer/Zeno scripts via:
    substrate.Substrate.nodes -> dict[int, NodeView]
    NodeView.state
    NodeView.direction_amplitudes()
    NodeView.neighbor_ids
    NodeView.n_connections
    NodeView.total_entanglement
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as _np

# ---------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------

try:
    import cupy as cp
    import cupy.linalg as cpl

    HAVE_CUPY = True
    XP = cp
    print("[substrate] Using CuPy GPU backend (batched).")
except Exception:
    cp = None
    cpl = None
    HAVE_CUPY = False
    XP = _np
    print("[substrate] CuPy not available; using NumPy CPU backend (batched).")

# SciPy only for CPU SVD/expm fallback
try:
    from scipy.linalg import svd as scipy_svd, expm as scipy_expm

    HAVE_SCIPY = True
except Exception:
    scipy_svd = None
    scipy_expm = None
    HAVE_SCIPY = False
    from numpy.linalg import svd as numpy_svd  # type: ignore


# ---------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------


def xp_randn(*shape):
    if HAVE_CUPY:
        return cp.random.randn(*shape)
    else:
        return _np.random.randn(*shape)


def xp_norm(x, ord=None, axis=None, keepdims=False):
    if HAVE_CUPY:
        return cp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    else:
        return _np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def xp_to_numpy(x):
    if HAVE_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return _np.asarray(x)


# ---------------------------------------------------------------------
# Entanglement edge representation
# ---------------------------------------------------------------------


class EntanglementEdge:
    """
    Represents entanglement operator connecting node_i <-> node_j.

    Stored as:
        operator: (dim, dim) on XP backend
        src, dst: integers on CPU
    """

    def __init__(self, src: int, dst: int, dim: int):
        self.src = int(src)
        self.dst = int(dst)
        self.dim = int(dim)
        self.operator = self._random_entanglement(dim)

    def _random_entanglement(self, d: int):
        M = xp_randn(d, d) + 1j * xp_randn(d, d)
        M = M.astype(XP.complex128)
        M /= xp_norm(M, ord="fro")
        return M

    def entanglement_entropy(self) -> float:
        M = self.operator
        if HAVE_CUPY:
            _, s, _ = cpl.svd(M, full_matrices=False)
            s = cp.asnumpy(s)
        else:
            M_cpu = _np.asarray(M)
            if HAVE_SCIPY and scipy_svd is not None:
                _, s, _ = scipy_svd(M_cpu, full_matrices=False)
            else:
                _, s, _ = numpy_svd(M_cpu, full_matrices=False)  # type: ignore
        s = s[s > 1e-10]
        if s.size == 0:
            return 0.0
        s = s / s.sum()
        return float(-_np.sum(s * _np.log(s + 1e-10)))


# ---------------------------------------------------------------------
# Node view (compat layer)
# ---------------------------------------------------------------------


class NodeView:
    """
    Lightweight view of a single node, backed by Substrate arrays.
    """

    def __init__(self, parent: "Substrate", idx: int):
        self._parent = parent
        self.id = int(idx)

    # State as XP array
    @property
    def state(self):
        return self._parent.states[self.id]

    @state.setter
    def state(self, value):
        arr = value
        if HAVE_CUPY:
            if not isinstance(arr, cp.ndarray):
                arr = cp.asarray(arr, dtype=cp.complex128)
        else:
            arr = _np.asarray(arr, dtype=_np.complex128)

        # Normalize for safety
        nrm = xp_norm(arr)
        if nrm > 0:
            arr = arr / nrm
        self._parent.states[self.id] = arr

    def direction_amplitudes(self):
        return self.state

    @property
    def neighbor_ids(self) -> List[int]:
        return self._parent.neighbors[self.id]

    @property
    def n_connections(self) -> int:
        return len(self._parent.neighbors[self.id])

    @property
    def total_entanglement(self) -> float:
        # Sum entanglement entropies over incident edges
        total = 0.0
        for e_idx in self._parent.node_edges[self.id]:
            total += self._parent.edge_list[e_idx].entanglement_entropy()
        return total


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class Config:
    """
    Config for batched substrate.

    monogamy_budget loosely maps to graph connectivity.
    """
    n_nodes: int = 64
    internal_dim: int = 3
    monogamy_budget: float = 1.0
    defrag_rate: float = 0.1
    dt: float = 0.1
    seed: int = 42


# ---------------------------------------------------------------------
# Substrate (batched engine)
# ---------------------------------------------------------------------


class Substrate:
    """
    Batched Hilbert substrate with CuPy acceleration where available.
    """

    def __init__(self, config: Config):
        self.config = config
        self.n_nodes = int(config.n_nodes)
        self.dim = int(config.internal_dim)

        # Seed RNGs
        _np.random.seed(config.seed)
        if HAVE_CUPY:
            cp.random.seed(config.seed)  # type: ignore[attr-defined]

        # Graph connectivity from monogamy_budget
        conn = float(config.monogamy_budget)
        conn = max(0.05, min(0.5, conn))
        self.connectivity = conn

        # States: (n_nodes, dim) on XP
        psi = xp_randn(self.n_nodes, self.dim) + 1j * xp_randn(self.n_nodes, self.dim)
        psi = psi.astype(XP.complex128)
        norms = xp_norm(psi, axis=1, keepdims=True)
        norms = norms + (norms == 0)
        psi = psi / norms
        self.states = psi

        # Graph structure on CPU
        self.neighbors: Dict[int, List[int]] = {i: [] for i in range(self.n_nodes)}
        self.edge_list: List[EntanglementEdge] = []
        self.node_edges: Dict[int, List[int]] = {i: [] for i in range(self.n_nodes)}

        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if _np.random.random() < self.connectivity:
                    e_idx = len(self.edge_list)
                    edge = EntanglementEdge(i, j, self.dim)
                    self.edge_list.append(edge)
                    self.neighbors[i].append(j)
                    self.neighbors[j].append(i)
                    self.node_edges[i].append(e_idx)
                    self.node_edges[j].append(e_idx)

        print(f"[substrate] Batched substrate: {self.n_nodes} nodes, {len(self.edge_list)} edges")

        # Nodes dict for compatibility with existing scripts
        self.nodes: Dict[int, NodeView] = {i: NodeView(self, i) for i in range(self.n_nodes)}

    # ---- Diagnostics ----

    def total_entanglement_entropy(self) -> float:
        return float(sum(e.entanglement_entropy() for e in self.edge_list))

    # ---- Local Hamiltonians (CPU assembly, GPU diag) ----

    def build_local_hamiltonians(self):
        """
        Build local Hamiltonian H[n] for each node as a (n_nodes, dim, dim) array.

        CPU assembly (graph indexing) is usually cheap; heavy linear algebra
        is handled in a batched way on XP backend.
        """
        H_cpu = _np.zeros((self.n_nodes, self.dim, self.dim), dtype=_np.complex128)

        for edge in self.edge_list:
            i = edge.src
            j = edge.dst
            op = xp_to_numpy(edge.operator)
            A = op @ op.T
            B = op.T @ op
            H_cpu[i] += A
            H_cpu[j] += B

        # Hermitian symmetrize
        H_cpu = 0.5 * (H_cpu + _np.conjugate(_np.swapaxes(H_cpu, -1, -2)))

        if HAVE_CUPY:
            return cp.asarray(H_cpu)
        else:
            return H_cpu

    def step(self, dt: Optional[float] = None):
        """
        One time step of local unitary evolution for all nodes, batched.
        """
        if dt is None:
            dt = self.config.dt

        # Build local H[n]
        H = self.build_local_hamiltonians()  # (N, d, d) on XP

        if HAVE_CUPY:
            # Batched eigen-decomposition on GPU
            w, v = cpl.eigh(H)          # w: (N, d), v: (N, d, d)
            phase = cp.exp(-1j * w * dt)  # (N, d)

            psi = self.states           # (N, d)
            psi_col = psi[..., None]    # (N, d, 1)
            v_dag = cp.conjugate(cp.swapaxes(v, -1, -2))  # (N, d, d)

            # Transform to eigenbasis: v† psi
            psi_eig = v_dag @ psi_col   # (N, d, 1)
            psi_eig = psi_eig[..., 0]   # (N, d)

            # Apply phase
            psi_eig = psi_eig * phase   # (N, d)

            # Transform back: v psi_eig
            psi_out = v @ psi_eig[..., None]  # (N, d, 1)
            psi_out = psi_out[..., 0]         # (N, d)

            # Normalize
            norms = xp_norm(psi_out, axis=1, keepdims=True)
            norms = norms + (norms == 0)
            self.states = psi_out / norms

        else:
            # CPU path: use NumPy/SciPy per-node
            psi = _np.asarray(self.states)
            N, d = psi.shape
            psi_out = _np.zeros_like(psi)

            for n in range(N):
                Hn = _np.asarray(H[n])
                if HAVE_SCIPY and scipy_expm is not None:
                    U = scipy_expm(-1j * Hn * dt)
                else:
                    w, v = _np.linalg.eigh(Hn)
                    phase = _np.exp(-1j * w * dt)
                    U = v @ (phase[..., None] * _np.conjugate(v).T)
                psi_out[n] = U @ psi[n]

            norms = _np.linalg.norm(psi_out, axis=1, keepdims=True)
            norms = norms + (norms == 0)
            self.states = psi_out / norms

    def defrag_step(self, rate: float = 0.1):
        """
        Simple toy defrag: randomly pick an edge and shift entanglement
        to another edge sharing a node.
        """
        if not self.edge_list:
            return

        import random

        e_idx = random.randrange(len(self.edge_list))
        edge = self.edge_list[e_idx]

        # Collect candidate edges sharing an endpoint
        cand_indices: List[int] = []
        for idx, other in enumerate(self.edge_list):
            if idx == e_idx:
                continue
            if (other.src == edge.src or other.src == edge.dst or
                other.dst == edge.src or other.dst == edge.dst):
                cand_indices.append(idx)

        if not cand_indices:
            return

        other_idx = random.choice(cand_indices)
        e1 = self.edge_list[e_idx]
        e2 = self.edge_list[other_idx]

        # Update operators on XP backend
        e1.operator *= (1.0 - rate)
        e2.operator *= (1.0 + rate)
        e1.operator /= xp_norm(e1.operator, ord="fro")
        e2.operator /= xp_norm(e2.operator, ord="fro")

    def evolve(self, n_steps: int = 1, dt: Optional[float] = None, defrag_rate: Optional[float] = None):
        """
        Evolve for n_steps, with optional defrag each step.
        """
        if dt is None:
            dt = self.config.dt
        if defrag_rate is None:
            defrag_rate = self.config.defrag_rate

        for _ in range(n_steps):
            self.step(dt=dt)
            if defrag_rate and defrag_rate > 0.0:
                self.defrag_step(rate=defrag_rate)


# ---------------------------------------------------------------------
# run_simulation (API used by pointer/Zeno scripts)
# ---------------------------------------------------------------------


def run_simulation(config: Config, n_steps: int, record_every: int = 1):
    """
    Minimal harness for pointer / Zeno experiments:

      - builds a Substrate from Config
      - evolves for n_steps
      - returns (substrate, records) with a simple entanglement time series
    """
    substrate = Substrate(config)
    times: List[float] = []
    ents: List[float] = []
    t = 0.0

    for step in range(n_steps):
        if step % max(1, record_every) == 0:
            times.append(t)
            ents.append(substrate.total_entanglement_entropy())
        substrate.evolve(n_steps=1)
        t += config.dt

    records = {
        "times": _np.asarray(times, dtype=float),
        "total_entanglement": _np.asarray(ents, dtype=float),
    }
    return substrate, records


# ---------------------------------------------------------------------
# Optional: quick smoke test when run directly
# ---------------------------------------------------------------------

if __name__ == "__main__":
    cfg = Config()
    sub, rec = run_simulation(cfg, n_steps=10, record_every=1)
    print("[substrate] Smoke test complete. Ent samples:", rec["total_entanglement"])
