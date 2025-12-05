#!/usr/bin/env python3
"""
spinor_particle_analysis.py
===========================

First-pass analysis of a spinor precipitating_event run.

Given a run directory containing:

    precipitating_event_timeseries.npz

produced by precipitating_event.py, this script:

  - loads:
      times   : (T,)
      psi_t   : (T, N, 2) complex spinor field
      X       : (N, 3) positions (embedding)
      S*_hist : (T,) global spin expectations
      config  : dict

  - extracts the final snapshot psi_final = psi_t[-1]

  - for each site i:
      * normalizes the local spinor (if non-zero),
      * computes local spin expectation s_i = <sigma>_i
        (Bloch vector in R^3)

  - saves:
      spinor_particle_features.npz with:
        X         : (N,3)
        s_vec     : (N,3)  [Sx_i, Sy_i, Sz_i]
        s_norm    : (N,)   [|s_i|]
        times     : (T,)
        Sx_hist   : (T,)
        Sy_hist   : (T,)
        Sz_hist   : (T,)
        config    : dict

  - writes spinor_particle_summary.txt with some basic diagnostics.

  - produces a simple 3D scatter plot:
        nodes at X, colored by Sz (final spin),
    saved as spinor_spin3d.png

This is a *local* spin analysis only. It does not yet compute any
topological charges (vortices, skyrmions, etc.). The goal is to see
whether the freeze-in produced nontrivial spin textures over the cube,
and to prepare data that a later "emergent spinor particles" script
can use for clustering and topology.

Usage
-----

  python spinor_particle_analysis.py --run-dir PATH\TO\RUN

where PATH\TO\RUN contains precipitating_event_timeseries.npz.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


def pauli_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Pauli matrices (2x2 complex)."""
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return sx, sy, sz


def load_timeseries(run_dir: str) -> Dict[str, Any]:
    """Load precipitating_event_timeseries.npz from the run directory."""
    npz_path = os.path.join(run_dir, "precipitating_event_timeseries.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Could not find {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    # Required keys:
    required = ["times", "psi_t", "X"]
    for key in required:
        if key not in data:
            raise KeyError(f"{npz_path} does not contain required key '{key}'")

    times = data["times"]
    psi_t = data["psi_t"]
    X = data["X"]

    Sx_hist = data["Sx_hist"] if "Sx_hist" in data else None
    Sy_hist = data["Sy_hist"] if "Sy_hist" in data else None
    Sz_hist = data["Sz_hist"] if "Sz_hist" in data else None
    config = data["config"].item() if "config" in data else {}

    return {
        "times": times,
        "psi_t": psi_t,
        "X": X,
        "Sx_hist": Sx_hist,
        "Sy_hist": Sy_hist,
        "Sz_hist": Sz_hist,
        "config": config,
    }


def compute_local_spin(psi_final: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given psi_final of shape (N, 2), compute local spin expectations:

       s_i = <sigma>_i = (Sx_i, Sy_i, Sz_i)

    where each local spinor at site i is normalized before computing
    expectation values. Sites with zero norm get s_i = (0,0,0).

    Returns:
      s_vec  : (N,3) array
      s_norm : (N,) array of |s_vec_i| (Bloch vector magnitude)
    """
    N, spin_dim = psi_final.shape
    if spin_dim < 2:
        raise ValueError(
            f"Expected spin_dim >= 2 for spin computation, got {spin_dim}"
        )

    sx, sy, sz = pauli_matrices()
    s_vec = np.zeros((N, 3), dtype=float)

    for i in range(N):
        v = psi_final[i].astype(np.complex128)
        norm = np.linalg.norm(v)
        if norm == 0.0:
            s_vec[i] = np.array([0.0, 0.0, 0.0], dtype=float)
            continue
        v = v / norm
        v = v.reshape(2, 1)  # (2,1)
        vdag = np.conj(v).T  # (1,2)

        Sx = (vdag @ (sx @ v))[0, 0]
        Sy = (vdag @ (sy @ v))[0, 0]
        Sz = (vdag @ (sz @ v))[0, 0]

        s_vec[i] = np.array(
            [Sx.real, Sy.real, Sz.real], dtype=float
        )

    s_norm = np.linalg.norm(s_vec, axis=1)
    return s_vec, s_norm


def save_features(
    run_dir: str,
    times: np.ndarray,
    X: np.ndarray,
    s_vec: np.ndarray,
    s_norm: np.ndarray,
    Sx_hist: np.ndarray | None,
    Sy_hist: np.ndarray | None,
    Sz_hist: np.ndarray | None,
    config: Dict[str, Any],
) -> str:
    """
    Save spinor particle features to spinor_particle_features.npz in run_dir.

    Returns the path to the saved file.
    """
    out_path = os.path.join(run_dir, "spinor_particle_features.npz")
    np.savez(
        out_path,
        X=X,
        s_vec=s_vec,
        s_norm=s_norm,
        times=times,
        Sx_hist=Sx_hist,
        Sy_hist=Sy_hist,
        Sz_hist=Sz_hist,
        config=config,
    )
    return out_path


def write_summary(
    run_dir: str,
    times: np.ndarray,
    X: np.ndarray,
    s_vec: np.ndarray,
    s_norm: np.ndarray,
    Sx_hist: np.ndarray | None,
    Sy_hist: np.ndarray | None,
    Sz_hist: np.ndarray | None,
    config: Dict[str, Any],
) -> str:
    """
    Write a human-readable summary to spinor_particle_summary.txt.
    """
    out_path = os.path.join(run_dir, "spinor_particle_summary.txt")
    N = X.shape[0]
    t0 = float(times[0])
    t_end = float(times[-1])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Spinor Particle Analysis\n")
        f.write("========================\n\n")
        f.write("Config:\n")
        f.write(repr(config) + "\n\n")
        f.write(f"N sites      = {N}\n")
        f.write(f"time range   = [{t0}, {t_end}] with {len(times)} steps\n")
        f.write("\n")

        # Global spin history (if available)
        if Sx_hist is not None and Sy_hist is not None and Sz_hist is not None:
            f.write("Global spin history (sampled):\n")
            sample_indices = np.linspace(
                0, len(times) - 1, min(12, len(times)), dtype=int
            )
            for idx in sample_indices:
                f.write(
                    f"  t={times[idx]:7.3f}  "
                    f"Sx={Sx_hist[idx]:+7.3f}  "
                    f"Sy={Sy_hist[idx]:+7.3f}  "
                    f"Sz={Sz_hist[idx]:+7.3f}\n"
                )
            f.write("\n")

        # Local spin statistics
        f.write("Local spin statistics (final snapshot):\n")
        f.write(f"  mean |s_i| = {np.mean(s_norm):.6f}\n")
        f.write(f"  min  |s_i| = {np.min(s_norm):.6f}\n")
        f.write(f"  max  |s_i| = {np.max(s_norm):.6f}\n")
        f.write("\n")

        # A few sample sites
        f.write("Sample sites (i, x, y, z, Sx, Sy, Sz, |s|):\n")
        show_idx = np.linspace(0, N - 1, min(10, N), dtype=int)
        for i in show_idx:
            x, y, z = X[i]
            sx, sy, sz = s_vec[i]
            f.write(
                f"  i={i:2d}  X=({x:+.3f},{y:+.3f},{z:+.3f})  "
                f"S=({sx:+.3f},{sy:+.3f},{sz:+.3f})  |s|={s_norm[i]:.3f}\n"
            )

    return out_path


def make_spin_scatter_plot(
    run_dir: str,
    X: np.ndarray,
    s_vec: np.ndarray,
) -> str:
    """
    Make a simple 3D scatter plot of nodes at X, colored by Sz, and save it
    as spinor_spin3d.png in run_dir.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D proj)

    sz = s_vec[:, 2]
    x, y, z = X[:, 0], X[:, 1], X[:, 2]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(x, y, z, c=sz, s=60, cmap="coolwarm")
    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("S_z (final)")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Spinor precipitating event: final S_z field")

    plt.tight_layout()
    out_path = os.path.join(run_dir, "spinor_spin3d.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def analyze_run(run_dir: str) -> None:
    """
    Top-level function: load run, compute local spin, save features,
    write summary, and make a simple plot.
    """
    print(f"Analyzing spinor precipitating event in: {run_dir}")
    data = load_timeseries(run_dir)
    times = data["times"]
    psi_t = data["psi_t"]
    X = data["X"]
    Sx_hist = data["Sx_hist"]
    Sy_hist = data["Sy_hist"]
    Sz_hist = data["Sz_hist"]
    config = data["config"]

    if psi_t.ndim != 3 or psi_t.shape[2] < 2:
        raise ValueError(
            f"psi_t must have shape (T, N, spin_dim>=2); got {psi_t.shape}"
        )

    T, N, spin_dim = psi_t.shape
    print(f"  Loaded psi_t with T={T}, N={N}, spin_dim={spin_dim}")
    print(f"  Time range: {times[0]} -> {times[-1]}")
    print("  Computing local spin on final snapshot...")

    psi_final = psi_t[-1]  # (N,2)
    s_vec, s_norm = compute_local_spin(psi_final)

    print("  Saving features...")
    features_path = save_features(
        run_dir, times, X, s_vec, s_norm, Sx_hist, Sy_hist, Sz_hist, config
    )
    print(f"    -> {features_path}")

    print("  Writing summary...")
    summary_path = write_summary(
        run_dir, times, X, s_vec, s_norm, Sx_hist, Sy_hist, Sz_hist, config
    )
    print(f"    -> {summary_path}")

    print("  Making 3D scatter plot of S_z...")
    plot_path = make_spin_scatter_plot(run_dir, X, s_vec)
    print(f"    -> {plot_path}")

    print("Analysis complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a spinor precipitating_event run by computing local "
            "spin expectations and basic diagnostics."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory with precipitating_event_timeseries.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    analyze_run(run_dir)


if __name__ == "__main__":
    main()
