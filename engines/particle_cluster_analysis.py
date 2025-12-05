#!/usr/bin/env python3
"""
particle_cluster_analysis.py

Cluster Hilbert-space node features from a pattern_detector run to identify
recurring "species-like" niches: heavy cores, bond/glue nodes, localized
lumps, delocalized bosonic patterns, etc.

Input:
  - A completed pattern_detector scan with --save-frames, i.e.:

      outputs/pattern_detector_scan/<source_run_id>/data/frames/frame_step_*.npz

Each frame NPZ is expected to contain arrays:
  degree, clustering, total_entanglement, localization,
  coherence, excitation, redundancy,
  proton_score, electron_score, photon_score,
  step, t

We stack features for all nodes across many frames and run a simple k-means
clustering in feature space.

Output:
  outputs/particle_cluster_analysis/<run_id>/
    params.json
    summary.json
    data/cluster_assignments.csv

Where cluster_assignments.csv has one row per (frame, node):

  step,t,node_id,cluster_id,proton_score,electron_score,photon_score

Usage example (Windows):

  python particle_cluster_analysis.py ^
      --source-run 20251201T183513Z_proton_electron_photon_scan ^
      --k 6 ^
      --max-frames 200 ^
      --output-root outputs ^
      --tag quark_gluon_probe

"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_run_id(tag: str | None) -> str:
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if tag:
        safe_tag = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in str(tag)
        )
        return f"{now}_{safe_tag}"
    return now


def _write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------
# K-means implementation (NumPy, no sklearn needed)
# ---------------------------------------------------------------------


@dataclass
class KMeansResult:
    centers: np.ndarray        # (k, D)
    assignments: np.ndarray    # (N,)
    inertia: float             # sum of squared distances


def _kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 0,
) -> KMeansResult:
    """
    Simple k-means clustering.

    Args:
      X: (N, D) array of standardized features
      k: number of clusters
      max_iter: maximum iterations
      tol: center shift tolerance
      seed: RNG seed

    Returns:
      KMeansResult with centers, assignments, and inertia.
    """
    N, D = X.shape
    if k <= 0:
        raise ValueError("k must be positive")
    if k > N:
        raise ValueError("k cannot exceed number of points")

    rng = np.random.default_rng(seed)

    # Initialize centers by sampling k distinct points
    indices = rng.choice(N, size=k, replace=False)
    centers = X[indices].copy()

    assignments = np.zeros(N, dtype=np.int64)
    prev_centers = centers.copy()

    for it in range(max_iter):
        # Assign step
        # squared distance: (x - c)^2 = x^2 + c^2 - 2 x·c
        # For simplicity, just brute-force compute distances.
        # dist2[i,j] = ||X[i] - centers[j]||^2
        # Shape: (N, k)
        dist2 = np.empty((N, k), dtype=X.dtype)
        for j in range(k):
            diff = X - centers[j]
            dist2[:, j] = np.sum(diff * diff, axis=1)

        assignments = np.argmin(dist2, axis=1)

        # Update step
        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = assignments == j
            if not np.any(mask):
                # Empty cluster: reinitialize to a random point
                idx = rng.integers(0, N)
                new_centers[j] = X[idx]
            else:
                new_centers[j] = X[mask].mean(axis=0)

        shift = np.sqrt(np.sum((new_centers - centers) ** 2))
        centers = new_centers

        if shift < tol:
            break

        prev_centers = centers.copy()

    # Final inertia
    dist2 = np.empty((N, k), dtype=X.dtype)
    for j in range(k):
        diff = X - centers[j]
        dist2[:, j] = np.sum(diff * diff, axis=1)
    nearest = np.min(dist2, axis=1)
    inertia = float(np.sum(nearest))

    return KMeansResult(centers=centers, assignments=assignments, inertia=inertia)


# ---------------------------------------------------------------------
# Feature loading & clustering
# ---------------------------------------------------------------------


@dataclass
class FeatureData:
    X: np.ndarray              # (N, D) raw feature matrix
    meta_step: np.ndarray      # (N,)
    meta_t: np.ndarray         # (N,)
    meta_node: np.ndarray      # (N,)
    meta_proton: np.ndarray    # (N,)
    meta_electron: np.ndarray  # (N,)
    meta_photon: np.ndarray    # (N,)


def _load_features_from_frames(
    frames_dir: str,
    max_frames: int | None = None,
) -> FeatureData:
    """
    Load per-node features across many frames.

    Each NPZ frame is expected to contain:
      - degree, clustering, total_entanglement, localization,
        coherence, excitation, redundancy,
        proton_score, electron_score, photon_score,
        step, t
    """
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"frames_dir does not exist: {frames_dir}")

    all_files = [
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.endswith(".npz")
    ]
    if not all_files:
        raise RuntimeError(f"No .npz frames found in {frames_dir}")

    # Sort frames by step encoded in filename
    def _extract_step(path: str) -> int:
        # Expect pattern: frame_step_000010.npz
        name = os.path.basename(path)
        # Fallback: just use lexicographic if parsing fails
        try:
            base = os.path.splitext(name)[0]
            parts = base.split("_")
            for p in parts:
                if p.isdigit():
                    return int(p)
        except Exception:
            pass
        return 0

    all_files.sort(key=_extract_step)

    if max_frames is not None and max_frames > 0 and len(all_files) > max_frames:
        # Sample frames roughly evenly
        idxs = np.linspace(0, len(all_files) - 1, max_frames, dtype=int)
        files = [all_files[i] for i in idxs]
    else:
        files = all_files

    feature_rows: List[np.ndarray] = []
    steps_list: List[int] = []
    t_list: List[float] = []
    node_list: List[int] = []
    proton_list: List[float] = []
    electron_list: List[float] = []
    photon_list: List[float] = []

    for frame_path in files:
        with np.load(frame_path) as data:
            degree = data["degree"]
            clustering = data["clustering"]
            total_entanglement = data["total_entanglement"]
            localization = data["localization"]
            coherence = data["coherence"]
            excitation = data["excitation"]
            redundancy = data["redundancy"]
            proton_score = data["proton_score"]
            electron_score = data["electron_score"]
            photon_score = data["photon_score"]
            step = int(data["step"])
            t = float(data["t"])

        n_nodes = degree.shape[0]

        for node_id in range(n_nodes):
            # Raw feature vector for this node in this frame
            row = np.array(
                [
                    degree[node_id],
                    clustering[node_id],
                    total_entanglement[node_id],
                    localization[node_id],
                    coherence[node_id],
                    excitation[node_id],
                    redundancy[node_id],
                    proton_score[node_id],
                    electron_score[node_id],
                    photon_score[node_id],
                ],
                dtype=float,
            )
            feature_rows.append(row)
            steps_list.append(step)
            t_list.append(t)
            node_list.append(node_id)
            proton_list.append(float(proton_score[node_id]))
            electron_list.append(float(electron_score[node_id]))
            photon_list.append(float(photon_score[node_id]))

    X = np.vstack(feature_rows)
    meta_step = np.array(steps_list, dtype=int)
    meta_t = np.array(t_list, dtype=float)
    meta_node = np.array(node_list, dtype=int)
    meta_proton = np.array(proton_list, dtype=float)
    meta_electron = np.array(electron_list, dtype=float)
    meta_photon = np.array(photon_list, dtype=float)

    return FeatureData(
        X=X,
        meta_step=meta_step,
        meta_t=meta_t,
        meta_node=meta_node,
        meta_proton=meta_proton,
        meta_electron=meta_electron,
        meta_photon=meta_photon,
    )


def _standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score each feature dimension.

    Returns:
      X_std: standardized features
      mean: (D,) mean
      std:  (D,) std (with minimum floor)
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    X_std = (X - mean) / std_safe
    return X_std, mean, std_safe


# ---------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cluster Hilbert-space node features from a pattern_detector run "
            "to identify recurrent species-like niches."
        )
    )

    p.add_argument(
        "--source-root",
        type=str,
        default="outputs/pattern_detector_scan",
        help="Root directory where pattern_detector runs are stored.",
    )
    p.add_argument(
        "--source-run",
        type=str,
        required=True,
        help=(
            "Source run ID under source-root, e.g. "
            "20251201T183513Z_proton_electron_photon_scan"
        ),
    )
    p.add_argument(
        "--k",
        type=int,
        default=6,
        help="Number of clusters (k in k-means).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help=(
            "Maximum number of frames to subsample for clustering. "
            "0 or negative = use all frames."
        ),
    )
    p.add_argument(
        "--cluster-seed",
        type=int,
        default=0,
        help="Random seed for k-means initialization.",
    )

    p.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory for cluster analysis outputs.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="particle_cluster_analysis",
        help="Tag for this analysis run.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    source_run_dir = os.path.join(args.source_root, args.source_run)
    frames_dir = os.path.join(source_run_dir, "data", "frames")
    if not os.path.isdir(source_run_dir):
        raise FileNotFoundError(f"source run dir does not exist: {source_run_dir}")
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"frames dir does not exist: {frames_dir}")

    run_id = _make_run_id(args.tag)
    run_root = os.path.join(args.output_root, "particle_cluster_analysis", run_id)
    data_dir = os.path.join(run_root, "data")
    _ensure_dir(run_root)
    _ensure_dir(data_dir)

    print("=" * 60)
    print("  Particle Cluster Analysis")
    print("=" * 60)
    print(f"Source run: {source_run_dir}")
    print(f"Frames dir: {frames_dir}")
    print(f"Output dir: {run_root}")
    print(f"k:          {args.k}")
    print(f"max_frames: {args.max_frames if args.max_frames > 0 else 'ALL'}")
    print("=" * 60)

    # --------------------------------------------------------------
    # Load features
    # --------------------------------------------------------------
    max_frames = args.max_frames if args.max_frames > 0 else None
    feats = _load_features_from_frames(frames_dir, max_frames=max_frames)

    print(f"Loaded feature points: {feats.X.shape[0]} (rows = node-frame pairs)")
    print(f"Feature dimension:     {feats.X.shape[1]}")
    print("Standardizing features...")
    X_std, mean_vec, std_vec = _standardize_features(feats.X)

    # --------------------------------------------------------------
    # Run k-means
    # --------------------------------------------------------------
    print("Running k-means clustering...")
    km_res = _kmeans(
        X_std,
        k=args.k,
        max_iter=100,
        tol=1e-4,
        seed=args.cluster_seED if hasattr(args, "cluster_seED") else args.cluster_seed,
    )
    assignments = km_res.assignments
    centers = km_res.centers
    inertia = km_res.inertia

    print(f"k-means inertia: {inertia:.4f}")
    print("Building cluster summaries...")

    # --------------------------------------------------------------
    # Build cluster summaries
    # --------------------------------------------------------------
    N, D = feats.X.shape
    feature_names = [
        "degree",
        "clustering",
        "total_entanglement",
        "localization",
        "coherence",
        "excitation",
        "redundancy",
        "proton_score",
        "electron_score",
        "photon_score",
    ]

    cluster_summaries: Dict[str, Any] = {
        "run_id": run_id,
        "source_run": args.source_run,
        "k": args.k,
        "total_points": int(N),
        "inertia": inertia,
        "feature_names": feature_names,
        "clusters": [],
    }

    for cid in range(args.k):
        mask = assignments == cid
        count = int(np.sum(mask))
        frac = float(count) / float(N) if N > 0 else 0.0

        if count > 0:
            # Means in original (unstandardized) feature space
            mean_feat = feats.X[mask].mean(axis=0)
            # Fractions of which species score dominates within this cluster
            p_dom = feats.meta_proton[mask]
            e_dom = feats.meta_electron[mask]
            ph_dom = feats.meta_photon[mask]
            # "winner" counts by heuristic species at this point
            wins_proton = int(np.sum((p_dom > e_dom) & (p_dom > ph_dom)))
            wins_electron = int(np.sum((e_dom > p_dom) & (e_dom > ph_dom)))
            wins_photon = int(np.sum((ph_dom > p_dom) & (ph_dom > e_dom)))
        else:
            mean_feat = np.zeros(D, dtype=float)
            wins_proton = 0
            wins_electron = 0
            wins_photon = 0

        summary = {
            "cluster_id": cid,
            "count": count,
            "fraction_of_points": frac,
            "mean_features": {
                name: float(val) for name, val in zip(feature_names, mean_feat)
            },
            "species_domination": {
                "proton_like_fraction": (
                    float(wins_proton) / float(count) if count > 0 else 0.0
                ),
                "electron_like_fraction": (
                    float(wins_electron) / float(count) if count > 0 else 0.0
                ),
                "photon_like_fraction": (
                    float(wins_photon) / float(count) if count > 0 else 0.0
                ),
            },
        }
        cluster_summaries["clusters"].append(summary)

    # --------------------------------------------------------------
    # Write summary.json and params.json
    # --------------------------------------------------------------
    params = {
        "run_id": run_id,
        "source_run": args.source_run,
        "source_root": args.source_root,
        "k": args.k,
        "max_frames": args.max_frames,
        "cluster_seed": args.cluster_seed,
    }
    _write_json(os.path.join(run_root, "params.json"), params)
    _write_json(os.path.join(run_root, "summary.json"), cluster_summaries)

    # --------------------------------------------------------------
    # Write cluster assignments CSV
    # --------------------------------------------------------------
    csv_path = os.path.join(data_dir, "cluster_assignments.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "step,t,node_id,cluster_id,"
            "proton_score,electron_score,photon_score\n"
        )
        for i in range(N):
            f.write(
                f"{feats.meta_step[i]},"
                f"{feats.meta_t[i]:.8f},"
                f"{feats.meta_node[i]},"
                f"{assignments[i]},"
                f"{feats.meta_proton[i]:.6f},"
                f"{feats.meta_electron[i]:.6f},"
                f"{feats.meta_photon[i]:.6f}\n"
            )

    print("=" * 60)
    print("Cluster analysis complete.")
    print(f"Summary:          {os.path.join(run_root, 'summary.json')}")
    print(f"Assignments CSV:  {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
