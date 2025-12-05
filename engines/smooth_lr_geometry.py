#!/usr/bin/env python3
"""
smooth_lr_geometry.py
=====================

Go "full smooth manifold mode" on an emergent LR geometry.

This script takes LR metric assets produced by substrate.py and constructs
a smoothed 3D embedding using Laplacian eigenmaps (diffusion-map style):

  - Load:
      asset_dir/lr_metrics.npz      (D_lr, D_prop, graph_dist)
      asset_dir/lr_embedding_3d.npz (X)           [optional but recommended]

  - Build a weighted graph from either:
      * D_lr using a Gaussian kernel, or
      * graph_dist using unit weights on edges with distance == 1.

  - Construct the (unnormalized) graph Laplacian:
      L = D - W

  - Solve L u_k = lambda_k u_k and take the first 3 *nontrivial*
    eigenvectors (skip the constant mode) as a smooth 3D embedding:

      X_le_raw = [u_1, u_2, u_3]   (shape (N,3))

  - If an original embedding X is available, align X_le_raw to X using
    Procrustes-style rigid alignment (rotation + uniform scale + translation)
    to get X_smooth that roughly preserves the original orientation/scale.

Outputs:
  - asset_dir/<output_file> (default: lr_embedding_3d_smooth.npz) with:
      X_smooth   : (N,3) aligned smooth embedding
      X_le_raw   : (N,3) raw Laplacian eigenmap coordinates
      eigenvals  : (N,) Laplacian eigenvalues (sorted)
      W_stats    : basic info about weight matrix
      config     : dict of parameters used

Use this as a drop-in replacement for the original X in downstream
scripts that need a smoother emergent geometry.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


# =============================================================================
# Config dataclass
# =============================================================================


@dataclass
class SmoothConfig:
    asset_dir: str = "substrate_cube2_gpu_v1"
    metrics_file: str = "lr_metrics.npz"
    embedding_file: str = "lr_embedding_3d.npz"
    output_file: str = "lr_embedding_3d_smooth.npz"

    # How to build weights
    use_lr_kernel: bool = True  # True: use D_lr + Gaussian kernel; False: use graph_dist==1
    kernel_scale: float = 1.0   # W_ij ~ exp( -(D_ij / (scale * median_nonzero_D))^2 )

    # Numerical options
    eps_diag: float = 1e-12     # tiny regularization for degrees


# =============================================================================
# Loading helpers
# =============================================================================


def load_metrics(asset_dir: str, metrics_file: str):
    path = os.path.join(asset_dir, metrics_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find metrics file at {path}")
    data = np.load(path)
    # Be tolerant about key names
    D_lr = data.get("D_lr")
    D_prop = data.get("D_prop")
    graph_dist = data.get("graph_dist")
    return D_lr, D_prop, graph_dist


def load_embedding(asset_dir: str, embedding_file: str):
    path = os.path.join(asset_dir, embedding_file)
    if not os.path.exists(path):
        return None
    data = np.load(path)
    X = data.get("X")
    return X


# =============================================================================
# Weight matrix and Laplacian
# =============================================================================


def build_weight_matrix(
    D_lr: Optional[np.ndarray],
    graph_dist: Optional[np.ndarray],
    cfg: SmoothConfig,
) -> np.ndarray:
    """
    Build symmetric weight matrix W_ij.

    If cfg.use_lr_kernel and D_lr is available:
      W_ij = exp( -(D_lr[i,j] / sigma)^2 ) for i != j
    Else if graph_dist is available:
      W_ij = 1.0 if graph_dist[i,j] == 1 else 0
    """
    if cfg.use_lr_kernel:
        if D_lr is None:
            raise ValueError("cfg.use_lr_kernel=True but D_lr is None in metrics.")
        D = np.array(D_lr, dtype=float)
        if D.shape[0] != D.shape[1]:
            raise ValueError("D_lr must be square.")
        N = D.shape[0]

        # Mask zero distances (self-distances)
        mask = D > 0
        if not np.any(mask):
            raise ValueError("D_lr has no non-zero entries; cannot build kernel.")

        median_nonzero = np.median(D[mask])
        sigma = cfg.kernel_scale * median_nonzero
        if sigma <= 0:
            raise ValueError("Computed sigma <= 0; check D_lr and kernel_scale.")

        W = np.exp(- (D / sigma) ** 2)
        np.fill_diagonal(W, 0.0)
        # symmetrize
        W = 0.5 * (W + W.T)
        return W

    else:
        if graph_dist is None:
            raise ValueError("cfg.use_lr_kernel=False but graph_dist is None in metrics.")
        G = np.array(graph_dist, dtype=float)
        if G.shape[0] != G.shape[1]:
            raise ValueError("graph_dist must be square.")
        N = G.shape[0]

        W = np.zeros((N, N), dtype=float)
        # unit weight for nearest neighbors (graph distance == 1)
        neighbors = np.isclose(G, 1.0)
        W[neighbors] = 1.0
        np.fill_diagonal(W, 0.0)
        # symmetrize
        W = 0.5 * (W + W.T)
        return W


def build_laplacian(W: np.ndarray, eps_diag: float = 1e-12) -> np.ndarray:
    """
    Build unnormalized graph Laplacian L = D - W,
    where D is diagonal with D_ii = sum_j W_ij.

    eps_diag is added to D_ii to avoid zero-degree issues.
    """
    d = W.sum(axis=1)
    d = d + eps_diag
    D_mat = np.diag(d)
    L = D_mat - W
    return L


# =============================================================================
# Laplacian eigenmaps and alignment
# =============================================================================


def laplacian_eigenmap(L: np.ndarray, n_components: int = 3):
    """
    Compute Laplacian eigenmap coordinates:

      L u_k = lambda_k u_k

    Take the first n_components nontrivial eigenvectors (skip constant mode).
    Returns:
      eigenvals (N,),
      X_le_raw (N, n_components)
    """
    # eigh returns eigenvalues in ascending order
    evals, evecs = np.linalg.eigh(L)
    # The constant eigenvector should be the first (lambda ~ 0).
    # We'll skip it and take the next n_components.
    idx_sorted = np.argsort(evals)
    evals = evals[idx_sorted]
    evecs = evecs[:, idx_sorted]

    # Skip first eigenvector (constant). Take next n_components.
    start = 1
    end = 1 + n_components
    if end > evecs.shape[1]:
        raise ValueError(
            f"Not enough eigenvectors for {n_components} components; "
            f"only {evecs.shape[1]} available."
        )
    X_le_raw = evecs[:, start:end].real
    return evals, X_le_raw


def procrustes_align(
    X_source: np.ndarray,
    X_target: np.ndarray,
) -> np.ndarray:
    """
    Align X_source to X_target via Procrustes (orthogonal rotation + uniform scale + translation).

    Both inputs: shape (N,3).

    Returns:
      X_aligned: shape (N,3)
    """
    if X_source.shape != X_target.shape:
        raise ValueError("X_source and X_target must have the same shape for alignment.")

    N, d = X_source.shape
    if d != 3:
        raise ValueError("Alignment currently expects d=3.")

    # Center each
    mu_src = X_source.mean(axis=0)
    mu_tgt = X_target.mean(axis=0)
    Xs = X_source - mu_src
    Xt = X_target - mu_tgt

    # Compute optimal rotation via SVD
    # Solve: minimize ||Xs R - Xt||_F with R orthogonal
    M = Xs.T @ Xt  # (3,3)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    # Ensure det(R)=+1 (proper rotation)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Compute scale factor
    norm_src = np.sqrt((Xs**2).sum())
    norm_tgt = np.sqrt((Xt**2).sum())
    if norm_src < 1e-15:
        s = 1.0
    else:
        s = norm_tgt / norm_src

    # Apply transform
    X_aligned = s * (Xs @ R) + mu_tgt
    return X_aligned


# =============================================================================
# Main driver
# =============================================================================


def run_smoothing(cfg: SmoothConfig) -> None:
    print("Smooth LR geometry config:")
    print(asdict(cfg))
    print()

    # Load metrics
    D_lr, D_prop, graph_dist = load_metrics(cfg.asset_dir, cfg.metrics_file)
    if D_lr is not None:
        print("Loaded D_lr with shape:", D_lr.shape)
    if graph_dist is not None:
        print("Loaded graph_dist with shape:", graph_dist.shape)
    print()

    # Load original embedding (optional)
    X_orig = load_embedding(cfg.asset_dir, cfg.embedding_file)
    if X_orig is not None:
        print("Loaded original embedding X with shape:", X_orig.shape)
    else:
        print("No original embedding found; will output only raw Laplacian embedding.")
    print()

    # Build weights and Laplacian
    if cfg.use_lr_kernel:
        print("Building weight matrix from D_lr using Gaussian kernel...")
    else:
        print("Building weight matrix from graph_dist (unit weights on graph_dist == 1)...")

    W = build_weight_matrix(D_lr, graph_dist, cfg)
    print("Weight matrix W shape:", W.shape)
    print("  W min/max (off-diagonal):",
          float(W[W > 0].min()) if np.any(W > 0) else 0.0,
          float(W.max()))
    print()

    L = build_laplacian(W, eps_diag=cfg.eps_diag)
    print("Laplacian L shape:", L.shape)
    print()

    # Laplacian eigenmap
    print("Computing Laplacian eigenmap (3D)...")
    evals, X_le_raw = laplacian_eigenmap(L, n_components=3)
    print("Eigenvalues (first 10):", evals[:10])
    print("X_le_raw shape:", X_le_raw.shape)
    print()

    # Alignment (if original embedding present)
    if X_orig is not None and X_orig.shape == X_le_raw.shape:
        print("Aligning smooth embedding to original X (Procrustes)...")
        X_smooth = procrustes_align(X_le_raw, X_orig)
    else:
        X_smooth = X_le_raw.copy()

    # Save output
    out_path = os.path.join(cfg.asset_dir, cfg.output_file)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    W_stats = {
        "min_positive": float(W[W > 0].min()) if np.any(W > 0) else 0.0,
        "max": float(W.max()),
        "mean": float(W.mean()),
    }

    np.savez(
        out_path,
        X_smooth=X_smooth,
        X_le_raw=X_le_raw,
        eigenvals=evals,
        W_stats=W_stats,
        config=asdict(cfg),
    )

    print(f"Saved smoothed embedding to {out_path}")
    print("Example coordinates (first few nodes):")
    for i in range(min(4, X_smooth.shape[0])):
        print(f"  node {i}: {X_smooth[i]}")


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smooth LR-derived geometry using Laplacian eigenmaps."
    )
    parser.add_argument(
        "--asset-dir",
        type=str,
        default="substrate_cube2_gpu_v1",
        help="Directory containing lr_metrics.npz and lr_embedding_3d.npz.",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="lr_metrics.npz",
        help="Name of LR metrics file inside asset-dir.",
    )
    parser.add_argument(
        "--embedding-file",
        type=str,
        default="lr_embedding_3d.npz",
        help="Name of original embedding file inside asset-dir.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="lr_embedding_3d_smooth.npz",
        help="Name of output smoothed embedding file inside asset-dir.",
    )
    parser.add_argument(
        "--no-lr-kernel",
        action="store_true",
        help="Use graph_dist==1 for weights instead of Gaussian kernel on D_lr.",
    )
    parser.add_argument(
        "--kernel-scale",
        type=float,
        default=1.0,
        help="Scale factor for Gaussian kernel width (sigma = scale * median_nonzero(D_lr)).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    cfg = SmoothConfig(
        asset_dir=args.asset_dir,
        metrics_file=args.metrics_file,
        embedding_file=args.embedding_file,
        output_file=args.output_file,
        use_lr_kernel=not args.no_lr_kernel,
        kernel_scale=args.kernel_scale,
    )
    run_smoothing(cfg)


if __name__ == "__main__":
    main()
