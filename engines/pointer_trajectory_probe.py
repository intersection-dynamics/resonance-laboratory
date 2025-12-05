#!/usr/bin/env python3
"""
Pointer Trajectory Probe
========================

Goal
----
Sit on top of the strictly-unitary, single-mode Hilbert substrate and:

1. Let it self-organize (burn-in).
2. Detect emergent proton/electron/photon pattern roles via clustering.
3. Sculpt a hydrogen-like initial state |Psi_0> with:
     - proton pattern concentrated around a proton center,
     - bound electron pattern near that proton,
     - photon pattern near the electron.
4. Evolve unitarily, but at discrete "measurement" times:
     - perform a coarse-grained projective measurement in a
       pointer basis:

         electron_shell ∈ {inner, outer}
         photon_region  ∈ {near, far}

       where inner/outer and near/far are defined by graph distance
       from proton/electron centers.

     - Collapse the state accordingly by zeroing incompatible regions
       and renormalizing.

5. Record a classical pointer trajectory:

       (t_k, electron_shell_k, photon_region_k)

   along with the continuous pattern weights.

This is a quantum-trajectory-style view of emergent pointer states for
spacetime (graph structure) and proton/electron/photon patterns.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from substrate import Config as SubConfig, Substrate


# =============================================================================
# Experiment configuration
# =============================================================================


@dataclass
class ExperimentConfig:
    # Substrate parameters
    n_nodes: int = 64
    internal_dim: int = 16
    connectivity: float = 0.35
    monogamy_budget: float = 1.0

    # Evolution
    burn_in_steps: int = 500
    total_steps: int = 4000
    record_stride: int = 10
    defrag_rate: float = 0.05
    dt: float = 0.1

    # Clustering / pattern detection
    n_clusters: int = 3
    kmeans_iters: int = 50

    # Initial pattern norms (target global norm partitions)
    proton_target_norm: float = 0.3
    electron_target_norm: float = 0.3
    photon_target_norm: float = 0.4

    # Pointer partition parameters
    # electron shell split: by graph-distance quantiles from proton
    electron_inner_quantile: float = 0.5  # fraction of closest electron nodes
    # photon near/far split: by distance quantiles from electron center
    photon_near_quantile: float = 0.5

    # Measurement cadence (in steps)
    measurement_stride: int = 50

    # Random seed
    seed: int = 42

    # Output
    output_root: str = "outputs"
    tag: str = "pointer_trajectory"


# =============================================================================
# Utility: filesystem, JSON
# =============================================================================


def make_run_dir(exp_cfg: ExperimentConfig) -> Tuple[Path, str]:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{timestamp}_{exp_cfg.tag}"
    root = Path(exp_cfg.output_root).resolve()
    run_dir = root / "pointer_trajectory_probe" / run_id

    (run_dir / "data").mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=False)
    return run_dir, run_id


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


# =============================================================================
# K-means clustering (simple NumPy implementation)
# =============================================================================


def kmeans(X: np.ndarray, k: int, n_iters: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple k-means clustering.

    Parameters
    ----------
    X : (N, D) array
    k : int
    n_iters : int
    rng : np.random.Generator

    Returns
    -------
    centers : (k, D)
    labels : (N,)
    """
    N, D = X.shape
    assert k <= N

    idx0 = rng.choice(N, size=k, replace=False)
    centers = X[idx0].copy()

    labels = np.zeros(N, dtype=int)
    for _ in range(n_iters):
        # Assign
        diff = X[:, None, :] - centers[None, :, :]  # (N,k,D)
        dist2 = np.sum(diff * diff, axis=-1)        # (N,k)
        labels = np.argmin(dist2, axis=1)

        # Update
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = X[mask].mean(axis=0)

    return centers, labels


# =============================================================================
# Pattern features and labeling
# =============================================================================


def compute_node_features(sub: Substrate) -> np.ndarray:
    """
    Compute per-node features from the substrate:

      - local_norm: ||psi_i||^2
      - ent_proxy:  sum_j |J_ij|
      - degree:     number of neighbors

    Returns
    -------
    features : (n_nodes, 3)
    """
    xp = sub.xp
    psi = sub.states
    J = sub.couplings

    if hasattr(xp, "asnumpy"):
        psi_np = xp.asnumpy(psi)
        J_np = xp.asnumpy(J)
    else:
        psi_np = np.asarray(psi)
        J_np = np.asarray(J)

    n_nodes = psi_np.shape[0]

    local_norm = np.sum(np.abs(psi_np) ** 2, axis=1)
    ent = np.sum(np.abs(J_np), axis=1)
    degree = np.zeros(n_nodes, dtype=float)
    for i in range(n_nodes):
        degree[i] = float(len(sub.neighbors(i)))

    return np.stack([local_norm, ent, degree], axis=1)


def standardize_features(X: np.ndarray) -> np.ndarray:
    """
    Zero-mean, unit-variance per feature.
    """
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (X - mean) / std


def label_clusters_by_role(centers: np.ndarray) -> Dict[int, str]:
    """
    Label 3 clusters as proton/electron/photon by center stats.

    centers: (3, 3) with features [local_norm, ent_proxy, degree].

    Heuristic:
      - "proton":  highest ent_proxy
      - "photon":  lowest local_norm
      - "electron": remaining cluster
    """
    if centers.shape[0] != 3:
        raise ValueError("label_clusters_by_role assumes k=3 clusters")

    local_norm = centers[:, 0]
    ent = centers[:, 1]

    proton_idx = int(np.argmax(ent))
    photon_idx = int(np.argmin(local_norm))
    all_idx = {0, 1, 2}
    electron_idx = int((all_idx - {proton_idx, photon_idx}).pop())

    return {
        proton_idx: "proton",
        electron_idx: "electron",
        photon_idx: "photon",
    }


# =============================================================================
# Picking representative nodes
# =============================================================================


def pick_representative_nodes(
    sub: Substrate,
    feats: np.ndarray,
    labels: np.ndarray,
    role_map: Dict[int, str],
) -> Dict[str, int]:
    """
    Choose one representative node for each role:

      - proton_center:  node in proton cluster with max ent_proxy.
      - electron_center: node in electron cluster with max local_norm and
                         small graph distance to proton_center.
      - photon_center:  node in photon cluster with small distance to
                        electron_center.
    """
    local_norm = feats[:, 0]
    ent = feats[:, 1]

    # Map role -> cluster index
    cluster_for_role = {role: idx for idx, role in role_map.items()}

    # Proton
    proton_cluster = cluster_for_role["proton"]
    proton_candidates = np.where(labels == proton_cluster)[0]
    if proton_candidates.size == 0:
        raise RuntimeError("No proton-like nodes found.")
    proton_center = int(proton_candidates[np.argmax(ent[proton_candidates])])

    # Electron
    electron_cluster = cluster_for_role["electron"]
    electron_candidates = np.where(labels == electron_cluster)[0]
    if electron_candidates.size == 0:
        raise RuntimeError("No electron-like nodes found.")

    dists_to_proton = np.array(
        [sub.graph_distance(proton_center, i, max_radius=sub.n_nodes) for i in electron_candidates],
        dtype=float,
    )
    finite_mask = np.isfinite(dists_to_proton)
    norm_e = local_norm[electron_candidates]

    if not np.any(finite_mask):
        # Fallback: ignore distance
        electron_center = int(electron_candidates[np.argmax(norm_e)])
    else:
        # Score = high norm - distance
        norm_std = (norm_e - norm_e.mean()) / (norm_e.std() if norm_e.std() > 0 else 1.0)
        dist_std = (dists_to_proton - dists_to_proton[finite_mask].mean()) / (
            dists_to_proton[finite_mask].std() if dists_to_proton[finite_mask].std() > 0 else 1.0
        )
        score = norm_std - dist_std
        score[~finite_mask] = -1e9
        electron_center = int(electron_candidates[np.argmax(score)])

    # Photon
    photon_cluster = cluster_for_role["photon"]
    photon_candidates = np.where(labels == photon_cluster)[0]
    if photon_candidates.size == 0:
        raise RuntimeError("No photon-like nodes found.")

    dists_to_electron = np.array(
        [sub.graph_distance(electron_center, i, max_radius=sub.n_nodes) for i in photon_candidates],
        dtype=float,
    )
    valid_mask = np.isfinite(dists_to_electron) & (dists_to_electron > 0.0)
    if not np.any(valid_mask):
        photon_center = int(photon_candidates[np.argmax(local_norm[photon_candidates])])
    else:
        photon_center = int(photon_candidates[np.argmin(dists_to_electron[valid_mask])])

    return {
        "proton_center": proton_center,
        "electron_center": electron_center,
        "photon_center": photon_center,
    }


# =============================================================================
# Initial state sculpting
# =============================================================================


def prepare_hydrogen_state(
    sub: Substrate,
    exp: ExperimentConfig,
    labels: np.ndarray,
    role_map: Dict[int, str],
    centers: Dict[str, int],
) -> None:
    """
    Sculpt an initial hydrogen-like state |Psi_0>:

      - Proton pattern: Gaussian in graph distance around proton_center
                        on proton-labeled nodes.
      - Electron pattern: Gaussian around electron_center on electron nodes.
      - Photon pattern: Gaussian around photon_center on photon nodes, with
                        a simple phase gradient vs distance from electron.

    All amplitude lives in internal component 0 (single-mode picture) and
    is scaled to match target norms (proton/electron/photon).
    """
    xp = sub.xp
    n_nodes, d = sub.n_nodes, sub.d

    # Cluster->role indices
    cluster_for_role = {role: idx for idx, role in role_map.items()}
    proton_cluster = cluster_for_role["proton"]
    electron_cluster = cluster_for_role["electron"]
    photon_cluster = cluster_for_role["photon"]

    masks = {
        "proton": labels == proton_cluster,
        "electron": labels == electron_cluster,
        "photon": labels == photon_cluster,
    }

    proton_center = centers["proton_center"]
    electron_center = centers["electron_center"]
    photon_center = centers["photon_center"]

    # Graph distances
    dist_from_proton = np.array(
        [sub.graph_distance(proton_center, i, max_radius=sub.n_nodes) for i in range(n_nodes)],
        dtype=float,
    )
    dist_from_electron = np.array(
        [sub.graph_distance(electron_center, i, max_radius=sub.n_nodes) for i in range(n_nodes)],
        dtype=float,
    )
    dist_from_photon = np.array(
        [sub.graph_distance(photon_center, i, max_radius=sub.n_nodes) for i in range(n_nodes)],
        dtype=float,
    )

    # Proton pattern
    sigma_p = 1.0
    w_p = np.exp(-0.5 * (dist_from_proton / sigma_p) ** 2)
    w_p[~masks["proton"]] = 0.0

    # Electron pattern
    sigma_e = 0.8
    w_e = np.exp(-0.5 * (dist_from_electron / sigma_e) ** 2)
    w_e[~masks["electron"]] = 0.0

    # Photon pattern (with phase gradient)
    sigma_ph = 1.0
    w_ph = np.exp(-0.5 * (dist_from_photon / sigma_ph) ** 2)
    w_ph[~masks["photon"]] = 0.0

    phase_k = 0.8
    phase_ph = np.exp(1j * phase_k * dist_from_electron)
    w_ph_complex = w_ph * phase_ph

    # Scale to target norms
    def scale_to_target(weights: np.ndarray, target_norm: float) -> float:
        norm = np.sum(np.abs(weights) ** 2)
        if norm <= 0.0 or target_norm <= 0.0:
            return 0.0
        return np.sqrt(target_norm / norm)

    s_p = scale_to_target(w_p, exp.proton_target_norm)
    s_e = scale_to_target(w_e, exp.electron_target_norm)
    s_ph = scale_to_target(w_ph_complex, exp.photon_target_norm)

    psi0 = np.zeros((n_nodes, d), dtype=np.complex128)
    psi0[:, 0] = s_p * w_p + s_e * w_e + s_ph * w_ph_complex

    # Global renormalization
    global_norm = np.linalg.norm(psi0.ravel())
    if global_norm == 0.0:
        global_norm = 1.0
    psi0 /= global_norm

    sub.states = xp.asarray(psi0, dtype=xp.complex128)


# =============================================================================
# Pointer partition (inner/outer, near/far)
# =============================================================================


def build_pointer_masks(
    sub: Substrate,
    labels: np.ndarray,
    role_map: Dict[int, str],
    centers: Dict[str, int],
    exp: ExperimentConfig,
) -> Dict[str, np.ndarray]:
    """
    Build boolean masks corresponding to pointer sectors:

      - proton_mask, electron_mask, photon_mask
      - electron_inner_mask, electron_outer_mask
      - photon_near_mask, photon_far_mask

    Inner/outer and near/far are defined by **quantiles of graph distance**
    so that both sets are guaranteed non-empty.
    """
    n_nodes = sub.n_nodes

    # Map role -> cluster index
    cluster_for_role = {role: idx for idx, role in role_map.items()}
    proton_cluster = cluster_for_role["proton"]
    electron_cluster = cluster_for_role["electron"]
    photon_cluster = cluster_for_role["photon"]

    proton_mask = labels == proton_cluster
    electron_mask = labels == electron_cluster
    photon_mask = labels == photon_cluster

    proton_center = centers["proton_center"]
    electron_center = centers["electron_center"]

    # Distances from proton/electron
    dist_from_proton = np.array(
        [sub.graph_distance(proton_center, i, max_radius=sub.n_nodes) for i in range(n_nodes)],
        dtype=float,
    )
    dist_from_electron = np.array(
        [sub.graph_distance(electron_center, i, max_radius=sub.n_nodes) for i in range(n_nodes)],
        dtype=float,
    )

    # Electron inner/outer via quantile among electron nodes
    e_dists = dist_from_proton[electron_mask]
    if e_dists.size == 0:
        # Degenerate: no electron nodes; everything false
        electron_inner_mask = np.zeros(n_nodes, dtype=bool)
        electron_outer_mask = np.zeros(n_nodes, dtype=bool)
    else:
        q = np.quantile(e_dists, exp.electron_inner_quantile)
        electron_inner_mask = electron_mask & (dist_from_proton <= q)
        electron_outer_mask = electron_mask & (dist_from_proton > q)

    # Photon near/far via quantile among photon nodes
    ph_dists = dist_from_electron[photon_mask]
    if ph_dists.size == 0:
        photon_near_mask = np.zeros(n_nodes, dtype=bool)
        photon_far_mask = np.zeros(n_nodes, dtype=bool)
    else:
        q_ph = np.quantile(ph_dists, exp.photon_near_quantile)
        photon_near_mask = photon_mask & (dist_from_electron <= q_ph)
        photon_far_mask = photon_mask & (dist_from_electron > q_ph)

    return {
        "proton_mask": proton_mask,
        "electron_mask": electron_mask,
        "photon_mask": photon_mask,
        "electron_inner_mask": electron_inner_mask,
        "electron_outer_mask": electron_outer_mask,
        "photon_near_mask": photon_near_mask,
        "photon_far_mask": photon_far_mask,
    }


# =============================================================================
# Pointer measurement (coarse collapse)
# =============================================================================


def pointer_measurement(
    sub: Substrate,
    masks: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """
    Perform a coarse-grained pointer measurement:

      - electron_shell ∈ {0=inner, 1=outer}
      - photon_region ∈ {0=near, 1=far}

    Implementation:
      - Compute current local norms per node.
      - Compute probabilities for electron_inner vs electron_outer sectors.
      - Sample an outcome, then zero incompatible nodes' amplitudes and
        renormalize.
      - Do the same for photon_near vs photon_far.

    Returns
    -------
    (electron_shell, photon_region)
    """
    xp = sub.xp
    psi = sub.states
    if hasattr(xp, "asnumpy"):
        psi_np = xp.asnumpy(psi)
    else:
        psi_np = np.asarray(psi)

    local_norm = np.sum(np.abs(psi_np) ** 2, axis=1)
    global_norm2 = float(np.sum(local_norm))
    if global_norm2 <= 0.0:
        global_norm2 = 1.0

    e_in_mask = masks["electron_inner_mask"]
    e_out_mask = masks["electron_outer_mask"]
    ph_near_mask = masks["photon_near_mask"]
    ph_far_mask = masks["photon_far_mask"]

    # --- electron shell measurement ---
    p_e_in = float(local_norm[e_in_mask].sum() / global_norm2)
    p_e_out = float(local_norm[e_out_mask].sum() / global_norm2)
    # If one of them is zero, default to the other
    p_e_in = max(p_e_in, 0.0)
    p_e_out = max(p_e_out, 0.0)
    norm_p_e = p_e_in + p_e_out
    if norm_p_e <= 0.0:
        # Degenerate: no electron region populated; pick "inner" by convention
        electron_shell = 0
    else:
        p_e_in /= norm_p_e
        p_e_out /= norm_p_e
        electron_shell = int(rng.choice([0, 1], p=[p_e_in, p_e_out]))

    # Collapse electron sector
    psi_after = psi_np.copy()
    if electron_shell == 0:
        # keep inner, zero outer
        psi_after[e_out_mask, :] = 0.0
    else:
        # keep outer, zero inner
        psi_after[e_in_mask, :] = 0.0

    # Renormalize globally
    norm2_after = float(np.sum(np.abs(psi_after) ** 2))
    if norm2_after > 0.0:
        psi_after /= np.sqrt(norm2_after)

    # --- photon region measurement ---
    local_norm2 = np.sum(np.abs(psi_after) ** 2, axis=1)
    global_norm2b = float(np.sum(local_norm2))
    if global_norm2b <= 0.0:
        global_norm2b = 1.0

    p_ph_near = float(local_norm2[ph_near_mask].sum() / global_norm2b)
    p_ph_far = float(local_norm2[ph_far_mask].sum() / global_norm2b)
    p_ph_near = max(p_ph_near, 0.0)
    p_ph_far = max(p_ph_far, 0.0)
    norm_p_ph = p_ph_near + p_ph_far
    if norm_p_ph <= 0.0:
        photon_region = 0
    else:
        p_ph_near /= norm_p_ph
        p_ph_far /= norm_p_ph
        photon_region = int(rng.choice([0, 1], p=[p_ph_near, p_ph_far]))

    if photon_region == 0:
        psi_after[ph_far_mask, :] = 0.0
    else:
        psi_after[ph_near_mask, :] = 0.0

    # Final renormalization
    norm2_final = float(np.sum(np.abs(psi_after) ** 2))
    if norm2_final > 0.0:
        psi_after /= np.sqrt(norm2_final)

    # Push back to substrate
    sub.states = xp.asarray(psi_after, dtype=xp.complex128)

    return electron_shell, photon_region


# =============================================================================
# Main experiment
# =============================================================================


def run_experiment(exp: ExperimentConfig) -> Dict[str, Any]:
    # Set up run directory and logging
    run_dir, run_id = make_run_dir(exp)
    log_path = run_dir / "logs" / "run.log"

    header = [
        "=" * 70,
        "POINTER TRAJECTORY PROBE",
        "=" * 70,
        f"Run ID: {run_id}",
        "",
        f"n_nodes:            {exp.n_nodes}",
        f"internal_dim:       {exp.internal_dim}",
        f"connectivity:       {exp.connectivity}",
        f"monogamy_budget:    {exp.monogamy_budget}",
        "",
        f"burn_in_steps:      {exp.burn_in_steps}",
        f"total_steps:        {exp.total_steps}",
        f"record_stride:      {exp.record_stride}",
        f"measurement_stride: {exp.measurement_stride}",
        f"defrag_rate:        {exp.defrag_rate}",
        f"dt:                 {exp.dt}",
        "",
        f"n_clusters:         {exp.n_clusters}",
        f"kmeans_iters:       {exp.kmeans_iters}",
        "",
        f"proton_target_norm:   {exp.proton_target_norm}",
        f"electron_target_norm: {exp.electron_target_norm}",
        f"photon_target_norm:   {exp.photon_target_norm}",
        "",
        f"seed:               {exp.seed}",
        "",
    ]
    log_text = "\n".join(header)
    print(log_text)
    log_path.write_text(log_text + "\n")

    # Build substrate
    cfg = SubConfig(
        n_nodes=exp.n_nodes,
        internal_dim=exp.internal_dim,
        connectivity=exp.connectivity,
        monogamy_budget=exp.monogamy_budget,
        defrag_rate=exp.defrag_rate,
        dt=exp.dt,
        seed=exp.seed,
        use_gpu=False,
    )
    sub = Substrate(cfg)

    # RNG for clustering + pointer sampling
    rng = np.random.default_rng(exp.seed + 1)

    # -------------------------------------------------------------------------
    # Burn-in
    # -------------------------------------------------------------------------
    print(f"[burn-in] Evolving for {exp.burn_in_steps} steps...")
    for _ in range(exp.burn_in_steps):
        sub.evolve(n_steps=1, defrag_rate=exp.defrag_rate)

    # -------------------------------------------------------------------------
    # Pattern detection via clustering
    # -------------------------------------------------------------------------
    print("[patterns] Computing node features & clustering...")
    feats = compute_node_features(sub)        # (n_nodes, 3)
    feats_std = standardize_features(feats)

    centers, labels = kmeans(feats_std, k=exp.n_clusters, n_iters=exp.kmeans_iters, rng=rng)
    role_map = label_clusters_by_role(centers)

    print("[patterns] Cluster centers (standardized) & roles:")
    for j in range(exp.n_clusters):
        print(f"  cluster {j}: center={centers[j]}, role={role_map[j]}")

    # Representatives & initial state
    reps = pick_representative_nodes(sub, feats, labels, role_map)
    print("[patterns] Representative nodes:")
    print(f"  proton_center:   {reps['proton_center']}")
    print(f"  electron_center: {reps['electron_center']}")
    print(f"  photon_center:   {reps['photon_center']}")

    prepare_hydrogen_state(sub, exp, labels, role_map, reps)

    # Pointer masks (inner/outer, near/far)
    masks = build_pointer_masks(sub, labels, role_map, reps, exp)

    # -------------------------------------------------------------------------
    # Main evolution + pointer measurements
    # -------------------------------------------------------------------------
    steps = exp.total_steps
    stride = exp.record_stride
    meas_stride = exp.measurement_stride
    n_records = steps // stride + 1
    n_meas = steps // meas_stride + 1

    times = np.zeros(n_records, dtype=float)
    proton_weight = np.zeros(n_records, dtype=float)
    electron_weight = np.zeros(n_records, dtype=float)
    photon_weight = np.zeros(n_records, dtype=float)

    electron_inner = np.zeros(n_records, dtype=float)
    electron_outer = np.zeros(n_records, dtype=float)
    photon_near = np.zeros(n_records, dtype=float)
    photon_far = np.zeros(n_records, dtype=float)

    # Pointer trajectory arrays
    meas_times = np.zeros(n_meas, dtype=float)
    meas_electron_shell = np.zeros(n_meas, dtype=int)  # 0=inner, 1=outer
    meas_photon_region = np.zeros(n_meas, dtype=int)   # 0=near, 1=far

    record_idx = 0
    meas_idx = 0

    print(f"[evolve] Main evolution for {steps} steps.")
    for step in range(steps + 1):
        t = step * exp.dt

        # Record continuous observables
        if step % stride == 0:
            xp = sub.xp
            psi = sub.states
            if hasattr(xp, "asnumpy"):
                psi_np = xp.asnumpy(psi)
            else:
                psi_np = np.asarray(psi)

            local_norm = np.sum(np.abs(psi_np) ** 2, axis=1)

            times[record_idx] = t
            proton_weight[record_idx] = float(local_norm[masks["proton_mask"]].sum())
            electron_weight[record_idx] = float(local_norm[masks["electron_mask"]].sum())
            photon_weight[record_idx] = float(local_norm[masks["photon_mask"]].sum())

            electron_inner[record_idx] = float(local_norm[masks["electron_inner_mask"]].sum())
            electron_outer[record_idx] = float(local_norm[masks["electron_outer_mask"]].sum())
            photon_near[record_idx] = float(local_norm[masks["photon_near_mask"]].sum())
            photon_far[record_idx] = float(local_norm[masks["photon_far_mask"]].sum())

            record_idx += 1

        # Pointer measurement (coarse collapse)
        if step % meas_stride == 0:
            e_shell, ph_region = pointer_measurement(sub, masks, rng)
            meas_times[meas_idx] = t
            meas_electron_shell[meas_idx] = e_shell
            meas_photon_region[meas_idx] = ph_region
            meas_idx += 1

        # Advance dynamics
        if step < steps:
            sub.evolve(n_steps=1, defrag_rate=exp.defrag_rate)

    # Final norm
    xp = sub.xp
    psi_final = sub.states
    if hasattr(xp, "asnumpy"):
        psi_final_np = xp.asnumpy(psi_final)
    else:
        psi_final_np = np.asarray(psi_final)

    global_norm_final = float(np.linalg.norm(psi_final_np.ravel()))

    print("=" * 70)
    print("Run complete.")
    print(f"Run directory: {run_dir}")
    print(f"Global state norm (final): {global_norm_final:.6f}")
    print("Final total pattern weights:")
    print(f"  proton:   {proton_weight[-1]:.4f}")
    print(f"  electron: {electron_weight[-1]:.4f}")
    print(f"  photon:   {photon_weight[-1]:.4f}")
    print("=" * 70)

    metrics = {
        "global_norm_final": global_norm_final,
        "proton_weight_final": float(proton_weight[-1]),
        "electron_weight_final": float(electron_weight[-1]),
        "photon_weight_final": float(photon_weight[-1]),
        "proton_weight_max": float(proton_weight.max()),
        "electron_weight_max": float(electron_weight.max()),
        "photon_weight_max": float(photon_weight.max()),
        "electron_inner_max": float(electron_inner.max()),
        "electron_outer_max": float(electron_outer.max()),
        "photon_near_max": float(photon_near.max()),
        "photon_far_max": float(photon_far.max()),
    }

    diagnostics = {
        "cluster_centers": centers.tolist(),
        "cluster_roles": {int(k): v for k, v in role_map.items()},
        "cluster_counts": {
            "proton": int(masks["proton_mask"].sum()),
            "electron": int(masks["electron_mask"].sum()),
            "photon": int(masks["photon_mask"].sum()),
        },
        "representative_nodes": reps,
    }

    summary = {
        "script": "pointer_trajectory_probe.py",
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": asdict(exp),
        "metrics": metrics,
        "diagnostics": diagnostics,
    }

    data_dir = run_dir / "data"
    np.save(data_dir / "times.npy", times)
    np.save(data_dir / "proton_weight.npy", proton_weight)
    np.save(data_dir / "electron_weight.npy", electron_weight)
    np.save(data_dir / "photon_weight.npy", photon_weight)
    np.save(data_dir / "electron_inner.npy", electron_inner)
    np.save(data_dir / "electron_outer.npy", electron_outer)
    np.save(data_dir / "photon_near.npy", photon_near)
    np.save(data_dir / "photon_far.npy", photon_far)

    np.save(data_dir / "meas_times.npy", meas_times)
    np.save(data_dir / "meas_electron_shell.npy", meas_electron_shell)
    np.save(data_dir / "meas_photon_region.npy", meas_photon_region)

    np.save(data_dir / "cluster_labels.npy", labels)
    np.save(data_dir / "node_features.npy", feats)

    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "params.json", asdict(exp))

    log_path.write_text(
        log_text + "\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    return summary


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser(description="Pointer trajectory probe on Hilbert substrate.")
    p.add_argument("--n-nodes", type=int, default=64)
    p.add_argument("--internal-dim", type=int, default=16)
    p.add_argument("--connectivity", type=float, default=0.35)
    p.add_argument("--monogamy-budget", type=float, default=1.0)

    p.add_argument("--burn-in-steps", type=int, default=500)
    p.add_argument("--total-steps", type=int, default=4000)
    p.add_argument("--record-stride", type=int, default=10)
    p.add_argument("--measurement-stride", type=int, default=50)
    p.add_argument("--defrag-rate", type=float, default=0.05)
    p.add_argument("--dt", type=float, default=0.1)

    p.add_argument("--n-clusters", type=int, default=3)
    p.add_argument("--kmeans-iters", type=int, default=50)

    p.add_argument("--proton-target-norm", type=float, default=0.3)
    p.add_argument("--electron-target-norm", type=float, default=0.3)
    p.add_argument("--photon-target-norm", type=float, default=0.4)

    p.add_argument("--electron-inner-quantile", type=float, default=0.5)
    p.add_argument("--photon-near-quantile", type=float, default=0.5)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", type=str, default="outputs")
    p.add_argument("--tag", type=str, default="pointer_trajectory")

    args = p.parse_args()

    return ExperimentConfig(
        n_nodes=args.n_nodes,
        internal_dim=args.internal_dim,
        connectivity=args.connectivity,
        monogamy_budget=args.monogamy_budget,
        burn_in_steps=args.burn_in_steps,
        total_steps=args.total_steps,
        record_stride=args.record_stride,
        measurement_stride=args.measurement_stride,
        defrag_rate=args.defrag_rate,
        dt=args.dt,
        n_clusters=args.n_clusters,
        kmeans_iters=args.kmeans_iters,
        proton_target_norm=args.proton_target_norm,
        electron_target_norm=args.electron_target_norm,
        photon_target_norm=args.photon_target_norm,
        electron_inner_quantile=args.electron_inner_quantile,
        photon_near_quantile=args.photon_near_quantile,
        seed=args.seed,
        output_root=args.output_root,
        tag=args.tag,
    )


def main() -> None:
    exp_cfg = parse_args()
    run_experiment(exp_cfg)


if __name__ == "__main__":
    main()
