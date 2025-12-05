#!/usr/bin/env python3
"""
Substrate Hydrogen Mini
=======================

Goal
----
Take the strictly-unitary, single-mode Hilbert substrate and:

1. Let it self-organize for a burn-in period.
2. Detect three emergent pattern roles by clustering per-node features:
     - "proton-like": high entanglement proxy (sum_j |J_ij|)
     - "electron-like": high local norm ||psi_i||^2
     - "photon-like": low norm, low entanglement
3. Use those roles to sculpt an initial state |Psi_0> with:
     - a concentrated proton pattern,
     - a bound electron pattern near that proton,
     - a photon pattern centered elsewhere.
4. Evolve unitarily and track pattern observables over time:
     - proton/electron/photon total weights,
     - photon near vs far from the electron,
     - electron inner vs outer (rough ground/excited shells).

This is an emergent, substrate-native "hydrogen toy": no species in the engine;
all structure comes from the Hilbert state + J(t) and pattern detection.
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

    # Initial pattern norms (target partitions of global norm)
    proton_target_norm: float = 0.3
    electron_target_norm: float = 0.3
    photon_target_norm: float = 0.4

    # Radii (in graph hops) for "near" regions
    electron_bound_radius: int = 2
    photon_near_radius: int = 2

    # Random seed
    seed: int = 42

    # Output
    output_root: str = "outputs"
    tag: str = "hydrogen_mini"


# =============================================================================
# Utility: filesystem, JSON
# =============================================================================


def make_run_dir(exp_cfg: ExperimentConfig) -> Tuple[Path, str]:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{timestamp}_{exp_cfg.tag}"
    root = Path(exp_cfg.output_root).resolve()
    run_dir = root / "substrate_hydrogen_mini" / run_id

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
        Data points.
    k : int
        Number of clusters.
    n_iters : int
        Iterations.
    rng : np.random.Generator

    Returns
    -------
    centers : (k, D) array
    labels : (N,) ints in [0, k-1]
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

    For each node i:
      - local_norm: ||psi_i||^2
      - ent_proxy: sum_j |J_ij|
      - degree: number of neighbors

    Returns
    -------
    features : (n_nodes, 3) array
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

    local_norm = np.sum(np.abs(psi_np) ** 2, axis=1)  # (n_nodes,)
    ent = np.sum(np.abs(J_np), axis=1)                # (n_nodes,)
    degree = np.zeros(n_nodes, dtype=float)
    for i in range(n_nodes):
        degree[i] = float(len(sub.neighbors(i)))

    features = np.stack([local_norm, ent, degree], axis=1)
    return features


def standardize_features(X: np.ndarray) -> np.ndarray:
    """
    Zero-mean, unit-variance per feature (with safety for zero variance).
    """
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (X - mean) / std


def label_clusters_by_role(centers: np.ndarray) -> Dict[int, str]:
    """
    Heuristically label 3 clusters as proton/electron/photon based on center stats.

    centers : (3, 3) with features [local_norm, ent, degree]

    Heuristic:
      - "proton": highest ent_proxy
      - "photon": lowest local_norm
      - "electron": remaining cluster
    """
    k = centers.shape[0]
    if k != 3:
        raise ValueError("label_clusters_by_role assumes k=3")

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
# Picking representative nodes for proton / electron / photon
# =============================================================================


def pick_representative_nodes(
    sub: Substrate,
    feats: np.ndarray,
    labels: np.ndarray,
    role_map: Dict[int, str],
) -> Dict[str, int]:
    """
    Choose one representative node for each role:

    - proton_center: node in proton cluster with max ent_proxy.
    - electron_center: node in electron cluster with max local_norm and minimal
      graph distance to proton_center (bound electron).
    - photon_center: node in photon cluster with minimal nonzero graph distance
      to electron_center (incoming photon near electron).
    """
    local_norm = feats[:, 0]
    ent = feats[:, 1]

    # Cluster indices
    cluster_for_role = {role: idx for idx, role in role_map.items()}

    # Proton center: max ent in proton cluster
    proton_cluster = cluster_for_role["proton"]
    proton_candidates = np.where(labels == proton_cluster)[0]
    if proton_candidates.size == 0:
        raise RuntimeError("No proton-like nodes found.")
    proton_center = int(proton_candidates[np.argmax(ent[proton_candidates])])

    # Electron center: in electron cluster, max local_norm, close to proton_center
    electron_cluster = cluster_for_role["electron"]
    electron_candidates = np.where(labels == electron_cluster)[0]
    if electron_candidates.size == 0:
        raise RuntimeError("No electron-like nodes found.")

    # Compute distances to proton for these candidates
    dists_to_proton = np.array(
        [sub.graph_distance(proton_center, i, max_radius=sub.n_nodes) for i in electron_candidates],
        dtype=float,
    )
    # If some are unreachable, mask them out
    finite_mask = np.isfinite(dists_to_proton)
    if not np.any(finite_mask):
        # Fallback: ignore distance, just pick max local_norm
        electron_center = int(electron_candidates[np.argmax(local_norm[electron_candidates])])
    else:
        # Score: prefer high norm and small distance
        norm_e = local_norm[electron_candidates]
        # Normalize scores
        norm_e_std = (norm_e - norm_e.mean()) / (norm_e.std() if norm_e.std() > 0 else 1.0)
        dist_std = (dists_to_proton - dists_to_proton[finite_mask].mean()) / (
            dists_to_proton[finite_mask].std() if dists_to_proton[finite_mask].std() > 0 else 1.0
        )
        # High norm (large) good, small distance good -> score = norm - dist
        score = norm_e_std - dist_std
        # Mask unreachable as very bad score
        score[~finite_mask] = -1e9
        electron_center = int(electron_candidates[np.argmax(score)])

    # Photon center: in photon cluster, minimal nonzero distance to electron_center
    photon_cluster = cluster_for_role["photon"]
    photon_candidates = np.where(labels == photon_cluster)[0]
    if photon_candidates.size == 0:
        raise RuntimeError("No photon-like nodes found.")

    dists_to_electron = np.array(
        [sub.graph_distance(electron_center, i, max_radius=sub.n_nodes) for i in photon_candidates],
        dtype=float,
    )
    # nonzero finite distances
    mask_valid = np.isfinite(dists_to_electron) & (dists_to_electron > 0.0)
    if not np.any(mask_valid):
        # Fallback: just pick the photon candidate with largest local_norm
        photon_center = int(photon_candidates[np.argmax(local_norm[photon_candidates])])
    else:
        photon_center = int(photon_candidates[np.argmin(dists_to_electron[mask_valid])])

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
    Sculpt an initial state |Psi_0> with:

      - Proton pattern: amplitude concentrated around proton_center
                        on proton-labeled nodes.
      - Electron pattern: amplitude around electron_center on electron nodes.
      - Photon pattern: amplitude around photon_center on photon nodes,
                        with a simple phase gradient.

    All amplitude is placed in internal component 0 (single-mode picture),
    and scaled to match the target norms in exp.
    """
    xp = sub.xp
    n_nodes, d = sub.n_nodes, sub.d

    # Masks
    masks = {
        "proton": labels == [k for k, v in role_map.items() if v == "proton"][0],
        "electron": labels == [k for k, v in role_map.items() if v == "electron"][0],
        "photon": labels == [k for k, v in role_map.items() if v == "photon"][0],
    }

    proton_center = centers["proton_center"]
    electron_center = centers["electron_center"]
    photon_center = centers["photon_center"]

    # Precompute graph distances
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

    # Proton pattern: Gaussian in graph distance on proton nodes
    sigma_p = 1.0
    w_p = np.exp(-0.5 * (dist_from_proton / sigma_p) ** 2)
    w_p[~masks["proton"]] = 0.0

    # Electron pattern: tighter Gaussian on electron nodes near proton
    sigma_e = 0.8
    w_e = np.exp(-0.5 * (dist_from_electron / sigma_e) ** 2)
    w_e[~masks["electron"]] = 0.0

    # Photon pattern: ring-like around electron, on photon nodes, with phase gradient
    sigma_ph = 1.0
    w_ph = np.exp(-0.5 * (dist_from_photon / sigma_ph) ** 2)
    w_ph[~masks["photon"]] = 0.0

    phase_k = 0.8  # simple phase gradient factor
    phase_ph = np.exp(1j * phase_k * dist_from_electron)  # phase vs distance from electron
    w_ph_complex = w_ph * phase_ph

    # Scale to target norms
    # Proton
    norm_p = np.sum(np.abs(w_p) ** 2)
    if norm_p > 0:
        s_p = np.sqrt(exp.proton_target_norm / norm_p)
    else:
        s_p = 0.0

    # Electron
    norm_e = np.sum(np.abs(w_e) ** 2)
    if norm_e > 0:
        s_e = np.sqrt(exp.electron_target_norm / norm_e)
    else:
        s_e = 0.0

    # Photon
    norm_ph = np.sum(np.abs(w_ph_complex) ** 2)
    if norm_ph > 0:
        s_ph = np.sqrt(exp.photon_target_norm / norm_ph)
    else:
        s_ph = 0.0

    # Build psi0 in NumPy, then push to backend
    psi0 = np.zeros((n_nodes, d), dtype=np.complex128)
    psi0[:, 0] = s_p * w_p + s_e * w_e + s_ph * w_ph_complex

    # Global renormalization for safety
    global_norm = np.linalg.norm(psi0.ravel())
    if global_norm == 0.0:
        global_norm = 1.0
    psi0 /= global_norm

    sub.states = xp.asarray(psi0, dtype=xp.complex128)


# =============================================================================
# Main experiment
# =============================================================================


def run_experiment(exp: ExperimentConfig) -> Dict[str, Any]:
    # Prepare run directory
    run_dir, run_id = make_run_dir(exp)
    log_path = run_dir / "logs" / "run.log"

    header = [
        "=" * 70,
        "SUBSTRATE HYDROGEN MINI",
        "=" * 70,
        f"Run ID: {run_id}",
        "",
        f"n_nodes:          {exp.n_nodes}",
        f"internal_dim:     {exp.internal_dim}",
        f"connectivity:     {exp.connectivity}",
        f"monogamy_budget:  {exp.monogamy_budget}",
        "",
        f"burn_in_steps:    {exp.burn_in_steps}",
        f"total_steps:      {exp.total_steps}",
        f"record_stride:    {exp.record_stride}",
        f"defrag_rate:      {exp.defrag_rate}",
        f"dt:               {exp.dt}",
        "",
        f"n_clusters:       {exp.n_clusters}",
        f"kmeans_iters:     {exp.kmeans_iters}",
        "",
        f"proton_target:    {exp.proton_target_norm}",
        f"electron_target:  {exp.electron_target_norm}",
        f"photon_target:    {exp.photon_target_norm}",
        "",
        f"seed:             {exp.seed}",
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

    # -------------------------------------------------------------------------
    # Burn-in
    # -------------------------------------------------------------------------
    print(f"[burn-in] Evolving for {exp.burn_in_steps} steps with defrag_rate={exp.defrag_rate}...")
    for _ in range(exp.burn_in_steps):
        sub.evolve(n_steps=1, defrag_rate=exp.defrag_rate)

    # -------------------------------------------------------------------------
    # Pattern detection via clustering
    # -------------------------------------------------------------------------
    print("[patterns] Computing node features and clustering...")
    feats = compute_node_features(sub)     # (n_nodes, 3)
    feats_std = standardize_features(feats)

    rng = np.random.default_rng(exp.seed + 1)
    centers, labels = kmeans(feats_std, k=exp.n_clusters, n_iters=exp.kmeans_iters, rng=rng)

    role_map = label_clusters_by_role(centers)
    print("[patterns] Cluster centers (standardized) and roles:")
    for j in range(exp.n_clusters):
        print(f"  cluster {j}: center={centers[j]}, role={role_map[j]}")

    # -------------------------------------------------------------------------
    # Pick representative nodes & sculpt initial hydrogen-like state
    # -------------------------------------------------------------------------
    reps = pick_representative_nodes(sub, feats, labels, role_map)
    print("[patterns] Representative nodes:")
    print(f"  proton_center:  {reps['proton_center']}")
    print(f"  electron_center:{reps['electron_center']}")
    print(f"  photon_center:  {reps['photon_center']}")

    prepare_hydrogen_state(sub, exp, labels, role_map, reps)

    # Build masks per role
    n_nodes = exp.n_nodes
    masks = {
        "proton": np.zeros(n_nodes, dtype=bool),
        "electron": np.zeros(n_nodes, dtype=bool),
        "photon": np.zeros(n_nodes, dtype=bool),
    }
    for i in range(n_nodes):
        role = role_map[int(labels[i])]
        masks[role][i] = True

    # Precompute distances from proton/electron centers for "near" regions
    proton_center = reps["proton_center"]
    electron_center = reps["electron_center"]
    dist_from_proton = np.array(
        [sub.graph_distance(proton_center, i, max_radius=sub.n_nodes) for i in range(n_nodes)],
        dtype=float,
    )
    dist_from_electron = np.array(
        [sub.graph_distance(electron_center, i, max_radius=sub.n_nodes) for i in range(n_nodes)],
        dtype=float,
    )

    electron_inner_mask = masks["electron"] & (dist_from_proton <= exp.electron_bound_radius)
    electron_outer_mask = masks["electron"] & (dist_from_proton > exp.electron_bound_radius)

    photon_near_mask = masks["photon"] & (dist_from_electron <= exp.photon_near_radius)
    photon_far_mask = masks["photon"] & (dist_from_electron > exp.photon_near_radius)

    # -------------------------------------------------------------------------
    # Main evolution and observables
    # -------------------------------------------------------------------------
    steps = exp.total_steps
    stride = exp.record_stride
    n_records = steps // stride + 1

    times = np.zeros(n_records, dtype=float)
    proton_weight = np.zeros(n_records, dtype=float)
    electron_weight = np.zeros(n_records, dtype=float)
    photon_weight = np.zeros(n_records, dtype=float)

    electron_inner = np.zeros(n_records, dtype=float)
    electron_outer = np.zeros(n_records, dtype=float)
    photon_near = np.zeros(n_records, dtype=float)
    photon_far = np.zeros(n_records, dtype=float)

    record_idx = 0
    print(f"[evolve] Main evolution for {steps} steps, record_stride={stride}...")
    for step in range(steps + 1):
        if step % stride == 0:
            t = step * exp.dt
            times[record_idx] = t

            xp = sub.xp
            psi = sub.states
            if hasattr(xp, "asnumpy"):
                psi_np = xp.asnumpy(psi)
            else:
                psi_np = np.asarray(psi)

            local_norm = np.sum(np.abs(psi_np) ** 2, axis=1)

            proton_weight[record_idx] = float(local_norm[masks["proton"]].sum())
            electron_weight[record_idx] = float(local_norm[masks["electron"]].sum())
            photon_weight[record_idx] = float(local_norm[masks["photon"]].sum())

            electron_inner[record_idx] = float(local_norm[electron_inner_mask].sum())
            electron_outer[record_idx] = float(local_norm[electron_outer_mask].sum())

            photon_near[record_idx] = float(local_norm[photon_near_mask].sum())
            photon_far[record_idx] = float(local_norm[photon_far_mask].sum())

            record_idx += 1

        if step < steps:
            sub.evolve(n_steps=1, defrag_rate=exp.defrag_rate)

    # Final global norm
    global_norm_final = float(np.linalg.norm(psi_np.ravel()))

    print("=" * 70)
    print("Run complete.")
    print(f"Run directory: {run_dir}")
    print(f"Global state norm (final): {global_norm_final:.6f}")
    print("Final total pattern weights:")
    print(f"  proton:   {proton_weight[-1]:.4f}")
    print(f"  electron: {electron_weight[-1]:.4f}")
    print(f"  photon:   {photon_weight[-1]:.4f}")
    print("Final inner/outer & near/far splits:")
    print(f"  electron_inner: {electron_inner[-1]:.4f}")
    print(f"  electron_outer: {electron_outer[-1]:.4f}")
    print(f"  photon_near:    {photon_near[-1]:.4f}")
    print(f"  photon_far:     {photon_far[-1]:.4f}")
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
            "proton": int(masks["proton"].sum()),
            "electron": int(masks["electron"].sum()),
            "photon": int(masks["photon"].sum()),
        },
        "representative_nodes": reps,
    }

    summary = {
        "script": "substrate_hydrogen_mini.py",
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
    p = argparse.ArgumentParser(description="Substrate Hydrogen Mini: emergent bound electron + photon patterns.")
    p.add_argument("--n-nodes", type=int, default=64)
    p.add_argument("--internal-dim", type=int, default=16)
    p.add_argument("--connectivity", type=float, default=0.35)
    p.add_argument("--monogamy-budget", type=float, default=1.0)

    p.add_argument("--burn-in-steps", type=int, default=500)
    p.add_argument("--total-steps", type=int, default=4000)
    p.add_argument("--record-stride", type=int, default=10)
    p.add_argument("--defrag-rate", type=float, default=0.05)
    p.add_argument("--dt", type=float, default=0.1)

    p.add_argument("--n-clusters", type=int, default=3)
    p.add_argument("--kmeans-iters", type=int, default=50)

    p.add_argument("--proton-target-norm", type=float, default=0.3)
    p.add_argument("--electron-target-norm", type=float, default=0.3)
    p.add_argument("--photon-target-norm", type=float, default=0.4)

    p.add_argument("--electron-bound-radius", type=int, default=2)
    p.add_argument("--photon-near-radius", type=int, default=2)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", type=str, default="outputs")
    p.add_argument("--tag", type=str, default="hydrogen_mini")

    args = p.parse_args()

    return ExperimentConfig(
        n_nodes=args.n_nodes,
        internal_dim=args.internal_dim,
        connectivity=args.connectivity,
        monogamy_budget=args.monogamy_budget,
        burn_in_steps=args.burn_in_steps,
        total_steps=args.total_steps,
        record_stride=args.record_stride,
        defrag_rate=args.defrag_rate,
        dt=args.dt,
        n_clusters=args.n_clusters,
        kmeans_iters=args.kmeans_iters,
        proton_target_norm=args.proton_target_norm,
        electron_target_norm=args.electron_target_norm,
        photon_target_norm=args.photon_target_norm,
        electron_bound_radius=args.electron_bound_radius,
        photon_near_radius=args.photon_near_radius,
        seed=args.seed,
        output_root=args.output_root,
        tag=args.tag,
    )


def main() -> None:
    exp_cfg = parse_args()
    run_experiment(exp_cfg)


if __name__ == "__main__":
    main()
