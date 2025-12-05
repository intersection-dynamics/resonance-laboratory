#!/usr/bin/env python3
"""
Substrate QED Mini
==================

Goal
----
Build the *emergent pattern* layer on top of the unified, strictly-unitary
substrate:

- We do NOT mark any node as "proton" or "electron" in the engine.
- We run the substrate for a while, then
  * extract simple per-node features from (states, couplings),
  * cluster nodes into 3 pattern types,
  * label those clusters heuristically as "proton-like", "electron-like",
    and "photon-like".
- We then evolve further and track the total "proton", "electron", and
  "photon" pattern weights as functions of time.

This is a first emergent pattern layer. Later, we can tune the features
and initial conditions to approach a true hydrogen-like transition
experiment (electron + photon <-> excited electron).
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

    # Random seed
    seed: int = 42

    # Output
    output_root: str = "outputs"
    tag: str = "qed_mini"


# =============================================================================
# Utility: filesystem, JSON
# =============================================================================


def make_run_dir(exp_cfg: ExperimentConfig) -> Tuple[Path, str]:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{timestamp}_{exp_cfg.tag}"
    root = Path(exp_cfg.output_root).resolve()
    run_dir = root / "substrate_qed_mini" / run_id

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
        RNG.

    Returns
    -------
    centers : (k, D) array
    labels : (N,) array of ints in [0, k-1]
    """
    N, D = X.shape
    assert k <= N

    # Initialize centers by picking random points
    idx0 = rng.choice(N, size=k, replace=False)
    centers = X[idx0].copy()

    labels = np.zeros(N, dtype=int)
    for _ in range(n_iters):
        # Assign
        # (N,1,D) - (1,k,D) -> (N,k,D) -> (N,k)
        diff = X[:, None, :] - centers[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        labels = np.argmin(dist2, axis=1)

        # Update centers
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
      - degree: number of neighbors
      - entanglement_proxy: sum_j |J_ij|

    Returns
    -------
    features : (n_nodes, 3) array
    """
    xp = sub.xp
    psi = sub.states
    J = sub.couplings

    psi_np = np.asarray(sub.xp.asnumpy(psi) if hasattr(xp, "asnumpy") else psi)
    J_np = np.asarray(sub.xp.asnumpy(J) if hasattr(xp, "asnumpy") else J)

    n_nodes = psi_np.shape[0]

    local_norm = np.sum(np.abs(psi_np) ** 2, axis=1)  # (n_nodes,)
    degree = np.zeros(n_nodes, dtype=float)
    ent = np.zeros(n_nodes, dtype=float)

    for i in range(n_nodes):
        nbrs = sub.neighbors(i)
        degree[i] = float(len(nbrs))
        ent[i] = float(np.sum(np.abs(J_np[i, :])))

    features = np.stack([local_norm, degree, ent], axis=1)  # (n_nodes, 3)
    return features


def standardize_features(X: np.ndarray) -> np.ndarray:
    """
    Zero-mean, unit-variance per feature (with safety for zero variance).
    """
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (X - mean) / std


def label_clusters_by_role(
    centers: np.ndarray,
) -> Dict[int, str]:
    """
    Heuristically label 3 clusters as proton/electron/photon based on center stats.

    centers : (3, 3) with features [local_norm, degree, ent]

    Heuristic:
      - "proton": highest entanglement_proxy
      - "photon": lowest local_norm
      - "electron": remaining cluster
    """
    k = centers.shape[0]
    if k != 3:
        raise ValueError("label_clusters_by_role assumes k=3")

    ent = centers[:, 2]
    norm = centers[:, 0]

    proton_idx = int(np.argmax(ent))
    photon_idx = int(np.argmin(norm))
    # electron is whichever is left
    all_idx = {0, 1, 2}
    electron_idx = int((all_idx - {proton_idx, photon_idx}).pop())

    return {
        proton_idx: "proton",
        electron_idx: "electron",
        photon_idx: "photon",
    }


# =============================================================================
# Main experiment
# =============================================================================


def run_experiment(exp: ExperimentConfig) -> Dict[str, Any]:
    # Prepare run directory
    run_dir, run_id = make_run_dir(exp)
    log_path = run_dir / "logs" / "run.log"

    header = [
        "=" * 70,
        "SUBSTRATE QED MINI",
        "=" * 70,
        f"Run ID: {run_id}",
        "",
        f"n_nodes:        {exp.n_nodes}",
        f"internal_dim:   {exp.internal_dim}",
        f"connectivity:   {exp.connectivity}",
        f"monogamy_budget:{exp.monogamy_budget}",
        "",
        f"burn_in_steps:  {exp.burn_in_steps}",
        f"total_steps:    {exp.total_steps}",
        f"record_stride:  {exp.record_stride}",
        f"defrag_rate:    {exp.defrag_rate}",
        f"dt:             {exp.dt}",
        "",
        f"n_clusters:     {exp.n_clusters}",
        f"kmeans_iters:   {exp.kmeans_iters}",
        "",
        f"seed:           {exp.seed}",
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
        use_gpu=False,  # explicit CPU for portability
    )
    sub = Substrate(cfg)

    # -------------------------------------------------------------------------
    # Burn-in: let the graph + state develop some structure
    # -------------------------------------------------------------------------
    print(f"[burn-in] Evolving for {exp.burn_in_steps} steps with defrag_rate={exp.defrag_rate}...")
    for _ in range(exp.burn_in_steps):
        sub.evolve(n_steps=1, defrag_rate=exp.defrag_rate)

    # -------------------------------------------------------------------------
    # Pattern detection: cluster node features into 3 pattern types
    # -------------------------------------------------------------------------
    print("[patterns] Computing node features for clustering...")
    feats = compute_node_features(sub)   # (n_nodes, 3)
    feats_std = standardize_features(feats)

    rng = np.random.default_rng(exp.seed + 1)
    centers, labels = kmeans(feats_std, k=exp.n_clusters, n_iters=exp.kmeans_iters, rng=rng)

    role_map = label_clusters_by_role(centers)
    print("[patterns] Cluster centers (standardized):")
    for j in range(exp.n_clusters):
        print(f"  cluster {j}: center={centers[j]}, role={role_map[j]}")

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

    # -------------------------------------------------------------------------
    # Main evolution: track pattern weights over time
    # -------------------------------------------------------------------------
    steps = exp.total_steps
    stride = exp.record_stride
    n_records = steps // stride + 1

    times = np.zeros(n_records, dtype=float)
    proton_weight = np.zeros(n_records, dtype=float)
    electron_weight = np.zeros(n_records, dtype=float)
    photon_weight = np.zeros(n_records, dtype=float)

    record_idx = 0
    print(f"[evolve] Main evolution for {steps} steps, record_stride={stride}...")
    for step in range(steps + 1):
        if step % stride == 0:
            t = step * exp.dt
            times[record_idx] = t

            # pattern weights = sum over masked nodes of local ||psi_i||^2
            psi_np = np.asarray(sub.xp.asnumpy(sub.states) if hasattr(sub.xp, "asnumpy") else sub.states)
            local_norm = np.sum(np.abs(psi_np) ** 2, axis=1)

            proton_weight[record_idx] = float(local_norm[masks["proton"]].sum())
            electron_weight[record_idx] = float(local_norm[masks["electron"]].sum())
            photon_weight[record_idx] = float(local_norm[masks["photon"]].sum())

            record_idx += 1

        if step < steps:
            sub.evolve(n_steps=1, defrag_rate=exp.defrag_rate)

    # -------------------------------------------------------------------------
    # Summaries and persistence
    # -------------------------------------------------------------------------
    total_norm_final = float(np.linalg.norm(psi_np.ravel()))
    print("=" * 70)
    print("Run complete.")
    print(f"Run directory: {run_dir}")
    print(f"global state norm (final): {total_norm_final:.6f}")
    print("Final pattern weights:")
    print(f"  proton:  {proton_weight[-1]:.4f}")
    print(f"  electron:{electron_weight[-1]:.4f}")
    print(f"  photon:  {photon_weight[-1]:.4f}")
    print("=" * 70)

    metrics = {
        "global_norm_final": total_norm_final,
        "proton_weight_final": float(proton_weight[-1]),
        "electron_weight_final": float(electron_weight[-1]),
        "photon_weight_final": float(photon_weight[-1]),
        "proton_weight_max": float(proton_weight.max()),
        "electron_weight_max": float(electron_weight.max()),
        "photon_weight_max": float(photon_weight.max()),
    }

    diagnostics = {
        "cluster_centers": centers.tolist(),
        "cluster_roles": {int(k): v for k, v in role_map.items()},
        "cluster_counts": {
            "proton": int(masks["proton"].sum()),
            "electron": int(masks["electron"].sum()),
            "photon": int(masks["photon"].sum()),
        },
    }

    summary = {
        "script": "substrate_qed_mini.py",
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
    np.save(data_dir / "cluster_labels.npy", labels)
    np.save(data_dir / "node_features.npy", feats)

    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "params.json", asdict(exp))

    log_path.write_text(log_text + "\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n")

    return summary


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser(description="Substrate QED Mini: emergent proton/electron/photon patterns.")
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

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", type=str, default="outputs")
    p.add_argument("--tag", type=str, default="qed_mini")

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
        seed=args.seed,
        output_root=args.output_root,
        tag=args.tag,
    )


def main() -> None:
    exp_cfg = parse_args()
    run_experiment(exp_cfg)


if __name__ == "__main__":
    main()
