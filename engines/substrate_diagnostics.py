#!/usr/bin/env python3
"""
substrate_diagnostics.py

Phase + topology diagnostics for the Substrate engine.

These tools do NOT modify the dynamics. They just read from a Substrate
snapshot (states + couplings + neighbors) and compute:

  1) Local loop phases (Berry-like phase around triangles):
       - Gauge-invariant phase around small closed loops
       - Candidate diagnostic for spinor / "half-vortex" behavior
         (e.g., fermion-like π phase vs boson-like 0 phase).

  2) Per-node "winding signatures":
       - Summary statistics of loop phases attached to each node
       - Lets you correlate "fermion-like" clusters with π-ish winding.

  3) Mode-occupancy statistics:
       - Treat |psi|^2 per node as a pointer-basis distribution
       - For a chosen subset of nodes (e.g. fermion cluster vs boson cluster),
         counts how many nodes "occupy" each internal basis mode above a
         probability threshold.
       - Used to probe exclusion (fermions: avoid double occupancy of
         the same mode) vs stacking (bosons: happily pile up in the same mode).

Usage pattern (from an experiment / REPL):

    from substrate import Config, Substrate
    from substrate_diagnostics import (
        compute_triangle_loop_phases,
        summarize_phase_winding,
        compute_mode_occupancy_stats,
    )

    sub = Substrate(cfg)
    sub.evolve(n_steps=... )

    tri_phases = compute_triangle_loop_phases(sub)
    winding = summarize_phase_winding(tri_phases)

    fermion_nodes = [...]  # indices from your cluster analysis
    boson_nodes = [...]
    occ = compute_mode_occupancy_stats(sub, fermion_nodes, boson_nodes)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore

    _HAS_CUPY = True
except Exception:
    cp = None  # type: ignore
    _HAS_CUPY = False


# ---------------------------------------------------------------------
# Helper: backend-agnostic to-numpy
# ---------------------------------------------------------------------


def _to_numpy(x) -> np.ndarray:
    """Convert an xp array (NumPy or CuPy) to a NumPy ndarray."""
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


# ---------------------------------------------------------------------
# 1. Triangle loop phases (Berry-like phase around small loops)
# ---------------------------------------------------------------------

@dataclass
class TrianglePhaseData:
    """
    Triangle-based phase diagnostics.

    Attributes:
      per_center_phases: list of lists; per_center_phases[i] is a list
                         of loop phases (in radians, [-pi, pi]) for all
                         triangles (i, j, k, i) where j, k are neighbors
                         of i and (j, k) is also an edge.
      n_triangles:       1D array, number of triangles attached to each node.
    """
    per_center_phases: List[List[float]]
    n_triangles: np.ndarray


def compute_triangle_loop_phases(sub) -> TrianglePhaseData:
    """
    Compute gauge-invariant complex phases around all triangles attached
    to each node.

    For each node i, for any pair of neighbors j, k such that j-k is an
    edge, we define a triangle (i, j, k, i) and compute:

        phi_ijk = Arg( <psi_i | psi_j> * <psi_j | psi_k> * <psi_k | psi_i> )

    This combination is gauge-invariant under local phase redefinitions
    psi_i -> e^{i θ_i} psi_i for each node. It is analogous to a
    discrete Berry phase around the loop.

    Intuition:
      - If a "fermion-like" excitation is associated with node i, you may
        find that triangles around i tend to have phases clustering
        around π (mod 2π) rather than 0, indicating a kind of half-vortex /
        spinor-like structure in phase space.

    Returns:
      TrianglePhaseData with per-center lists of loop phases and a count
      of triangles per node.
    """
    states = _to_numpy(sub.states).astype(np.complex128)
    neighbors = sub._neighbors  # list[list[int]]
    n_nodes = sub.n_nodes

    # For faster "is there an edge j-k?" queries, precompute neighbor sets.
    neighbor_sets: List[set[int]] = [set(nbs) for nbs in neighbors]

    per_center_phases: List[List[float]] = [[] for _ in range(n_nodes)]
    n_triangles = np.zeros(n_nodes, dtype=int)

    for i in range(n_nodes):
        nbs = list(neighbors[i])
        k = len(nbs)
        if k < 2:
            continue

        psi_i = states[i]

        # For each unordered pair (j, k) of neighbors of i, check if j-k
        # is also an edge; if so, form a triangle (i, j, k).
        for a_idx in range(k):
            j = nbs[a_idx]
            psi_j = states[j]
            for b_idx in range(a_idx + 1, k):
                k_node = nbs[b_idx]
                # Ensure j-k_node is an edge in the graph
                if k_node not in neighbor_sets[j]:
                    continue

                psi_k = states[k_node]

                s_ij = np.vdot(psi_i, psi_j)
                s_jk = np.vdot(psi_j, psi_k)
                s_ki = np.vdot(psi_k, psi_i)

                loop_amp = s_ij * s_jk * s_ki
                phi = float(np.angle(loop_amp))  # in [-pi, pi]

                per_center_phases[i].append(phi)
                n_triangles[i] += 1

    return TrianglePhaseData(per_center_phases=per_center_phases, n_triangles=n_triangles)


@dataclass
class PhaseWindingSummary:
    """
    Summary statistics for triangle loop phases at each node.

    Attributes:
      mean_phase:        circular mean phase per node (NaN if no triangles)
      phase_spread:      circular spread (1 - resultant length) per node
      has_pi_winding:    heuristic flag: True if phases cluster near +/- pi
      has_zero_winding:  heuristic flag: True if phases cluster near 0
    """
    mean_phase: np.ndarray
    phase_spread: np.ndarray
    has_pi_winding: np.ndarray
    has_zero_winding: np.ndarray


def summarize_phase_winding(
    tri_data: TrianglePhaseData,
    pi_window: float = np.pi / 3.0,
) -> PhaseWindingSummary:
    """
    Summarize triangle loop phases per node in a gauge-invariant way.

    For each node i, given its list of triangle phases phi_ijk, we compute:
      - circular mean:
           mean_phi_i = Arg( sum_jk e^{i phi_ijk} )
      - resultant length:
           R_i = | (1/M_i) sum_jk e^{i phi_ijk} |
           (M_i = number of triangles at node i)
      - "spread" = 1 - R_i

    Heuristic flags:
      - has_pi_winding[i]   = True if mean_phi_i is within `pi_window`
                              of π or -π AND spread is small
      - has_zero_winding[i] = True if mean_phi_i is within `pi_window`
                              of 0 AND spread is small

    This lets you:
      - correlate "fermion-like" nodes (from your cluster analysis) with
        π-type winding (spinor-ish exchange behavior),
      - distinguish them from boson-like nodes with near-zero winding.
    """
    per_center_phases = tri_data.per_center_phases
    n_nodes = len(per_center_phases)

    mean_phase = np.full(n_nodes, np.nan, dtype=float)
    phase_spread = np.full(n_nodes, np.nan, dtype=float)
    has_pi_winding = np.zeros(n_nodes, dtype=bool)
    has_zero_winding = np.zeros(n_nodes, dtype=bool)

    for i in range(n_nodes):
        phis = per_center_phases[i]
        if not phis:
            continue

        phis_arr = np.array(phis, dtype=float)
        # Circular mean via vector sum
        z = np.exp(1j * phis_arr)
        z_mean = z.mean()
        mean_phi = float(np.angle(z_mean))
        R = float(np.abs(z_mean))  # resultant length in [0, 1]
        spread = 1.0 - R

        mean_phase[i] = mean_phi
        phase_spread[i] = spread

        # Heuristic classification
        # "well-clustered" if spread < 0.5 (you can tune)
        if spread < 0.5:
            # Check proximity to 0 or π
            # Normalize distance to principal branch
            # For π, treat both +π and -π as equivalent.
            dist_zero = abs((mean_phi + np.pi) % (2.0 * np.pi) - np.pi)
            dist_pi = min(abs(mean_phi - np.pi), abs(mean_phi + np.pi))

            if dist_pi <= pi_window:
                has_pi_winding[i] = True
            if dist_zero <= pi_window:
                has_zero_winding[i] = True

    return PhaseWindingSummary(
        mean_phase=mean_phase,
        phase_spread=phase_spread,
        has_pi_winding=has_pi_winding,
        has_zero_winding=has_zero_winding,
    )


# ---------------------------------------------------------------------
# 2. Mode occupancy (exclusion vs stacking in internal basis)
# ---------------------------------------------------------------------

@dataclass
class ModeOccupancyStats:
    """
    Occupancy statistics for internal pointer-basis modes.

    For each internal basis index m (0..d-1), we count:

      occ_fermion[m]  = number of fermion nodes with p_i[m] >= thresh
      occ_boson[m]    = number of boson nodes with   p_i[m] >= thresh
      occ_all[m]      = number of all nodes with     p_i[m] >= thresh

    and also record:

      total_fermions  = size of fermion set
      total_bosons    = size of boson set
      total_nodes     = n_nodes
    """
    occ_fermion: np.ndarray
    occ_boson: np.ndarray
    occ_all: np.ndarray
    total_fermions: int
    total_bosons: int
    total_nodes: int
    prob_threshold: float


def compute_mode_occupancy_stats(
    sub,
    fermion_nodes: Sequence[int],
    boson_nodes: Sequence[int],
    prob_threshold: float = 0.5,
) -> ModeOccupancyStats:
    """
    Compute occupancy counts for each internal basis mode, separately
    for "fermion" and "boson" node sets.

    Args:
      sub:            Substrate instance (snapshot).
      fermion_nodes:  indices of nodes considered "fermion-like"
                      (e.g. from a cluster dominated by electron_score).
      boson_nodes:    indices of nodes considered "boson-like"
                      (e.g. from a photon-dominated cluster).
      prob_threshold: p_i[m] >= prob_threshold counts as "occupying mode m".

    Returns:
      ModeOccupancyStats with arrays of counts per basis mode.

    Interpretation:
      - If fermions obey exclusion in a given pointer basis, you'd expect
        few modes m with occ_fermion[m] >= 2 (very few double occupancies),
        whereas bosonic modes can have occ_boson[m] >> 1.
      - You can compare the empirical histogram of occ_fermion vs occ_boson
        against a random assignment baseline.
    """
    states = _to_numpy(sub.states).astype(np.complex128)
    n_nodes, d = states.shape

    # Classical pointer-basis probabilities
    probs = np.abs(states) ** 2
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)

    fermion_mask = np.zeros(n_nodes, dtype=bool)
    boson_mask = np.zeros(n_nodes, dtype=bool)

    for idx in fermion_nodes:
        if 0 <= idx < n_nodes:
            fermion_mask[int(idx)] = True
    for idx in boson_nodes:
        if 0 <= idx < n_nodes:
            boson_mask[int(idx)] = True

    occ_fermion = np.zeros(d, dtype=int)
    occ_boson = np.zeros(d, dtype=int)
    occ_all = np.zeros(d, dtype=int)

    for m in range(d):
        pm = probs[:, m]
        # mode m is "occupied" at node i if p_i[m] >= prob_threshold
        occ_all[m] = int(np.sum(pm >= prob_threshold))
        occ_fermion[m] = int(np.sum((pm >= prob_threshold) & fermion_mask))
        occ_boson[m] = int(np.sum((pm >= prob_threshold) & boson_mask))

    return ModeOccupancyStats(
        occ_fermion=occ_fermion,
        occ_boson=occ_boson,
        occ_all=occ_all,
        total_fermions=int(fermion_mask.sum()),
        total_bosons=int(boson_mask.sum()),
        total_nodes=n_nodes,
        prob_threshold=float(prob_threshold),
    )
