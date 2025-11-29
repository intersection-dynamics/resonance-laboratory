#!/usr/bin/env python3
"""
pattern_detector.py

Pattern detectors for emergent "lumps" in the Hilbert substrate.

Ontology:
  - There are no fundamental particles.
  - The Substrate is a big Hilbert sea with local DOFs and entanglement.
  - "Proton / electron / photon" are labels for *patterns* in this sea:
      proton-like : heavy, strongly entangled, localized composite
      electron-like: lighter, localized, less entangled / less redundant
      photon-like : bosonic, delocalized, *copyable* info:
                    many sites carry very similar pointer-basis distributions.

Implementation notes:
  - We extract per-node features:
      degree, clustering, total entanglement,
      localization, coherence, excitation,
      redundancy (copyability).
  - Redundancy is computed from classical |psi|^2 distributions:
      redundancy_i ~ average similarity to other nodes' |psi|^2
      (Bhattacharyya coefficient).
  - Photon score heavily favors high redundancy + low localization + coherence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from substrate import Substrate  # type: ignore

# Optional CuPy awareness for pulling node.state
try:
    import cupy as cp  # type: ignore

    HAVE_CUPY = True
except Exception:
    cp = None
    HAVE_CUPY = False


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def to_numpy(x) -> np.ndarray:
    """Convert a state to a NumPy array, handling CuPy arrays if present."""
    if HAVE_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


@dataclass
class NodeFeatures:
    """Per-node features extracted from a single Substrate snapshot."""
    degree: np.ndarray            # (N,)
    clustering: np.ndarray        # (N,)
    total_entanglement: np.ndarray  # (N,)
    localization: np.ndarray      # (N,)
    coherence: np.ndarray         # (N,)
    excitation: np.ndarray        # (N,)
    redundancy: np.ndarray        # (N,)  # "copyability" of pointer distributions


@dataclass
class SpeciesScores:
    """Per-node scores for each emergent species."""
    proton_score: np.ndarray   # (N,)
    electron_score: np.ndarray # (N,)
    photon_score: np.ndarray   # (N,)


@dataclass
class SpeciesCandidates:
    """Best candidate ids and scores for each emergent species at a snapshot."""
    proton_id: int
    proton_score: float
    electron_id: int
    electron_score: float
    photon_id: int
    photon_score: float


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------


def compute_node_features(substrate: Substrate) -> NodeFeatures:
    """
    Compute per-node features from a Substrate snapshot.

    Features:
      - degree        : number of neighbors
      - clustering    : simple local clustering coefficient
      - total_ent     : sum of entanglement entropy on incident edges
      - localization  : 1 - S(|psi|^2)/log(d)
      - coherence     : sum |rho_ij|, i != j, for pure state rho = |psi><psi|
      - excitation    : 1 - |<psi | psi_avg>|^2   (psi_avg over all nodes)
      - redundancy    : "copyability" of pointer distributions:
                        mean Bhattacharyya similarity of |psi|^2 to others'
    """
    nodes = substrate.nodes
    n_nodes = len(nodes)
    if n_nodes == 0:
        raise ValueError("Substrate has no nodes; cannot compute features.")

    # --- adjacency & degrees ---
    neighbors = substrate.neighbors  # dict[int, list[int]]
    degree = np.zeros(n_nodes, dtype=float)
    for i in range(n_nodes):
        degree[i] = float(len(neighbors.get(i, [])))

    # --- clustering coefficient (triangle density) ---
    clustering = np.zeros(n_nodes, dtype=float)
    for i in range(n_nodes):
        nbs = neighbors.get(i, [])
        k = len(nbs)
        if k < 2:
            clustering[i] = 0.0
            continue
        nb_set = set(nbs)
        edges_between = 0
        for j_idx in range(k):
            j = nbs[j_idx]
            for k_idx in range(j_idx + 1, k):
                l = nbs[k_idx]
                if l in neighbors.get(j, []):
                    edges_between += 1
        clustering[i] = 2.0 * edges_between / (k * (k - 1))

    # --- edge entanglement entropy, cached per edge ---
    edge_ent = np.zeros(len(substrate.edge_list), dtype=float)
    for e_idx, edge in enumerate(substrate.edge_list):
        try:
            edge_ent[e_idx] = float(edge.entanglement_entropy())
        except Exception:
            edge_ent[e_idx] = 0.0

    total_entanglement = np.zeros(n_nodes, dtype=float)
    for i in range(n_nodes):
        e_indices = substrate.node_edges.get(i, [])
        total_entanglement[i] = float(edge_ent[e_indices].sum()) if e_indices else 0.0

    # --- states & global average (for excitation) ---
    d = substrate.dim  # internal_dim of each node
    states = np.zeros((n_nodes, d), dtype=np.complex128)
    for i in range(n_nodes):
        psi = nodes[i].state
        psi_np = to_numpy(psi).astype(np.complex128)
        if psi_np.shape[0] != d:
            if psi_np.shape[0] < d:
                tmp = np.zeros(d, dtype=np.complex128)
                tmp[: psi_np.shape[0]] = psi_np
                psi_np = tmp
            else:
                psi_np = psi_np[:d]
        nrm = np.linalg.norm(psi_np)
        if nrm > 0:
            psi_np = psi_np / nrm
        states[i] = psi_np

    psi_avg = states.mean(axis=0)
    nrm_avg = np.linalg.norm(psi_avg)
    if nrm_avg > 0:
        psi_avg = psi_avg / nrm_avg

    localization = np.zeros(n_nodes, dtype=float)
    coherence = np.zeros(n_nodes, dtype=float)
    excitation = np.zeros(n_nodes, dtype=float)

    logd = np.log(d) if d > 1 else 1.0

    # classical probability distributions for redundancy
    probs = np.abs(states) ** 2
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    probs = np.clip(probs, 1e-12, 1.0)

    for i in range(n_nodes):
        psi = states[i]
        p = probs[i]

        # entropy / localization
        S = float(-np.sum(p * np.log(p)))
        localization[i] = 1.0 - S / logd

        # coherence: off-diagonal magnitude of rho = |psi><psi|
        rho = np.outer(psi, np.conjugate(psi))
        off = rho - np.diag(np.diag(rho))
        coherence[i] = float(np.sum(np.abs(off)))

        # excitation relative to global average pattern
        ov = np.vdot(psi, psi_avg)
        excitation[i] = float(1.0 - np.abs(ov) ** 2)

    # --- redundancy / copyability ---
    # Bhattacharyya coefficients between classical distributions:
    # BC(i,j) = sum_k sqrt(p_i[k] p_j[k])
    # redundancy_i = mean_{j != i} BC(i,j)
    sqrt_p = np.sqrt(probs)
    bc_matrix = sqrt_p @ sqrt_p.T  # (N,N)
    # zero out diagonal before averaging
    np.fill_diagonal(bc_matrix, 0.0)
    redundancy = bc_matrix.sum(axis=1) / np.maximum(n_nodes - 1, 1)

    return NodeFeatures(
        degree=degree,
        clustering=clustering,
        total_entanglement=total_entanglement,
        localization=localization,
        coherence=coherence,
        excitation=excitation,
        redundancy=redundancy,
    )


# ---------------------------------------------------------------------
# Species scoring
# ---------------------------------------------------------------------


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sigma


def compute_species_scores(features: NodeFeatures) -> SpeciesScores:
    """
    Build heuristic scores for proton-like, electron-like, photon-like patterns.

    We normalize features across nodes and then use simple linear combinations:

      proton_score  ~ +deg + ent + loc - small*excitation
      electron_score~ +loc + excitation - 0.5*deg - 0.5*ent - 0.5*redundancy
      photon_score  ~ -loc + coherence + 0.5*ent + 1.5*redundancy

    Intuition:
      - Proton: heavy, strongly entangled, localized, not too exotic.
      - Electron: localized & excited but NOT highly entangled / high-degree /
                  highly redundant.
      - Photon: bosonic, delocalized, coherent, *copyable* (high redundancy).
    """
    z_deg = _zscore(features.degree)
    z_ent = _zscore(features.total_entanglement)
    z_loc = _zscore(features.localization)
    z_coh = _zscore(features.coherence)
    z_exc = _zscore(features.excitation)
    z_red = _zscore(features.redundancy)

    # Proton: heavy, strongly entangled, localized, modest excitation
    proton_score = (
        1.0 * z_deg +
        1.0 * z_ent +
        1.0 * z_loc -
        0.2 * z_exc
    )

    # Electron: localized & excited, but not super-entangled or highly redundant
    electron_score = (
        1.0 * z_loc +
        1.0 * z_exc -
        0.5 * z_deg -
        0.5 * z_ent -
        0.5 * z_red
    )

    # Photon: delocalized, coherent, somewhat entangled, highly redundant
    photon_score = (
        -1.0 * z_loc +
        1.0 * z_coh +
        0.5 * z_ent +
        1.5 * z_red
    )

    return SpeciesScores(
        proton_score=proton_score,
        electron_score=electron_score,
        photon_score=photon_score,
    )


def pick_species_candidates(scores: SpeciesScores) -> SpeciesCandidates:
    """
    Choose best candidate ids and scores for each species from per-node scores.
    """
    p_idx = int(np.argmax(scores.proton_score))
    e_idx = int(np.argmax(scores.electron_score))
    ph_idx = int(np.argmax(scores.photon_score))

    return SpeciesCandidates(
        proton_id=p_idx,
        proton_score=float(scores.proton_score[p_idx]),
        electron_id=e_idx,
        electron_score=float(scores.electron_score[e_idx]),
        photon_id=ph_idx,
        photon_score=float(scores.photon_score[ph_idx]),
    )


# ---------------------------------------------------------------------
# High-level analysis entry point
# ---------------------------------------------------------------------


def analyze_substrate(substrate: Substrate) -> Tuple[NodeFeatures, SpeciesScores, SpeciesCandidates]:
    """
    Analyze a Substrate snapshot and return:

      - NodeFeatures      : per-node feature arrays
      - SpeciesScores     : per-node proton/electron/photon heuristic scores
      - SpeciesCandidates : best candidate ids & scores

    Use this as a high-level call from experiments.
    """
    feats = compute_node_features(substrate)
    scores = compute_species_scores(feats)
    cands = pick_species_candidates(scores)
    return feats, scores, cands
