#!/usr/bin/env python3
"""
substrate_core.py — Substrate v0.1
Minimal Hilbert-on-a-graph engine satisfying:
  • Hilbert-space realism
  • Unitary evolution
  • No signaling (local H)
  • Classical emergence (observables only)

Coordinates:
  Sites 0...(N-1)
  Each site = qubit Hilbert space C^2
  Graph adjacency determines allowable couplings.

Hamiltonian:
  H = sum_{(i,j) in edges} J * (X_i X_j + Y_i Y_j)  +  sum_i h * Z_i

  • Nearest-neighbor only → Lieb–Robinson bound holds.
  • Local Z-field (h) acts like a “defrag tendency” without collapse.

Everything else is layered by experiments later.
"""

import numpy as np
from numpy.linalg import eigh


# ---------------------------------------------------------------------
# Pauli operators
# ---------------------------------------------------------------------
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)


def kronN(ops):
    """Kronecker product of a list of site-local operators."""
    out = ops[0]
    for k in range(1, len(ops)):
        out = np.kron(out, ops[k])
    return out


# ---------------------------------------------------------------------
# Build local terms
# ---------------------------------------------------------------------
def local_operator(pauli, site, N):
    """
    Build N-site operator with given Pauli at 'site' and I elsewhere.
    """
    ops = [I] * N
    ops[site] = pauli
    return kronN(ops)


def two_site_term(pauliA, i, pauliB, j, N):
    """Tensor product operator: pauliA_i ⊗ pauliB_j ⊗ I_elsewhere."""
    ops = [I] * N
    ops[i] = pauliA
    ops[j] = pauliB
    return kronN(ops)


# ---------------------------------------------------------------------
# Hamiltonian construction (local only)
# ---------------------------------------------------------------------
def build_hamiltonian(N, edges, J=1.0, h=0.0):
    """
    Build the minimal local Hamiltonian.

    H = J * sum_{(i,j)} (X_i X_j + Y_i Y_j)
        + h * sum_i Z_i

    This is fully local on the graph → satisfies no signaling.
    """
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)

    # two-site couplings
    for (i, j) in edges:
        H += J * two_site_term(X, i, X, j, N)
        H += J * two_site_term(Y, i, Y, j, N)

    # on-site “defrag” field
    for i in range(N):
        H += h * local_operator(Z, i, N)

    return H


# ---------------------------------------------------------------------
# Time evolution
# ---------------------------------------------------------------------
def diagonalize(H):
    """Diagonalize H once for fast exponentiation."""
    evals, evecs = eigh(H)
    return evals, evecs


def evolve(psi0, t, evals, evecs):
    """
    ψ(t) = e^{-iHt} ψ(0)
    using spectral decomposition.
    """
    phases = np.exp(-1j * evals * t)
    return evecs @ (phases * (evecs.conj().T @ psi0))


# ---------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------
def local_z_expectation(psi, N):
    """Compute <Z_i> for each site i."""
    out = np.zeros(N, dtype=float)
    for i in range(N):
        Zi = local_operator(Z, i, N)
        out[i] = np.real(psi.conj().T @ (Zi @ psi))
    return out


def local_excitation(psi, N):
    """
    n_i = (1 - <Z_i>)/2
    """
    z = local_z_expectation(psi, N)
    return 0.5 * (1 - z)


# ---------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------
def random_state(N, seed=None):
    """Random normalized state in dimension 2^N."""
    if seed is not None:
        np.random.seed(seed)
    psi = np.random.randn(2**N) + 1j * np.random.randn(2**N)
    return psi / np.linalg.norm(psi)
