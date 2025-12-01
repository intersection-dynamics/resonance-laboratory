#!/usr/bin/env python3
"""
proton_entanglement_movie.py

Build an animation of the substrate around the probed proton
using the data saved in `proton_entanglement_wavefront.npz`.

Expected keys in the NPZ (from proton_pointer_suite.py):
    times        : (T,)
    entanglement : (T, N)  -- per-node entanglement/activity
    distances    : (N,)
    node_ids     : (N,) optional

Outputs:
    output_dir/frame_0000.png, frame_0001.png, ...
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------

def layout_nodes_by_shell(distances: np.ndarray):
    """
    Place nodes on concentric rings based on integer distance shell.

    Parameters
    ----------
    distances : (N,) array
        Graph distance of each node from the probed node.

    Returns
    -------
    pos : (N, 2) array of (x, y) positions
    shells : dict[int, list[int]]
        Mapping shell_index -> list of node indices
    """
    shells = {}
    for idx, d in enumerate(distances):
        shell = int(round(float(d)))
        shells.setdefault(shell, []).append(idx)

    if not shells:
        return np.zeros((len(distances), 2)), {}

    n_shells = max(shells.keys())
    pos = np.zeros((len(distances), 2), dtype=float)

    # Radius spacing
    r_min = 0.1
    r_max = 1.0
    r_step = (r_max - r_min) / max(n_shells, 1) if n_shells > 0 else 0.0

    for shell, idxs in shells.items():
        if shell == 0:
            # Probed node at center
            for i in idxs:
                pos[i] = np.array([0.0, 0.0])
        else:
            r = r_min + (shell - 1) * r_step
            n_in_shell = len(idxs)
            for k, i in enumerate(idxs):
                theta = 2 * np.pi * k / max(n_in_shell, 1)
                pos[i] = np.array([r * np.cos(theta), r * np.sin(theta)])

    return pos, shells


def make_frame(
    out_path: Path,
    t_value: float,
    ent_t: np.ndarray,
    ent_max: float,
    positions: np.ndarray,
    distances: np.ndarray,
    probed_index: int,
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """
    Render a single PNG frame.

    ent_t is the entanglement/activity per node at time t.
    """

    norm_ent = ent_t / (ent_max + 1e-12)

    # Sizes: emphasize relative entanglement
    sizes = 50.0 + 450.0 * norm_ent

    # Colors: map entanglement → colormap, gently modulated by distance
    dist_norm = distances / (distances.max() + 1e-12)
    color_values = 0.5 * norm_ent + 0.5 * (1.0 - dist_norm)

    fig, ax = plt.subplots(figsize=(6, 6))

    sc = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=color_values,
        s=sizes,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
    )

    # Highlight the probed node with a red ring
    ax.scatter(
        positions[probed_index, 0],
        positions[probed_index, 1],
        s=sizes[probed_index] * 1.5,
        facecolors="none",
        edgecolors="red",
        linewidths=2.0,
    )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Entanglement bubble around proton pointer\n t = {t_value:.2f} (arb. units)")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("relative entanglement / activity")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Render entanglement bubble movie from NPZ.")
    parser.add_argument(
        "--input",
        type=str,
        default="proton_entanglement_wavefront.npz",
        help="Input .npz file (from proton_pointer_suite).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="proton_entanglement_frames",
        help="Directory to save PNG frames into.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on number of frames (for quick tests).",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input NPZ not found: {in_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(in_path)

    # Robust key lookup: prefer "entanglement", fallback to "ent_by_node"
    if "entanglement" in data.files:
        ent_by_node = data["entanglement"]
    elif "ent_by_node" in data.files:
        ent_by_node = data["ent_by_node"]
    else:
        raise KeyError(f"No 'entanglement' or 'ent_by_node' array in {in_path}")

    times = data["times"]             # (T,)
    distances = data["distances"]     # (N,)
    node_ids = data.get("node_ids", None)  # optional

    T, N = ent_by_node.shape
    print("Loaded entanglement wavefront:")
    print(f"  times shape      : {times.shape}")
    print(f"  ent_by_node shape: {ent_by_node.shape}")
    print(f"  distances shape  : {distances.shape}")
    if node_ids is not None:
        print(f"  node_ids shape   : {node_ids.shape}")

    # Layout nodes in shells
    positions, shells = layout_nodes_by_shell(distances)
    print(f"  shells: { {k: len(v) for k, v in shells.items()} }")

    # Probed node is the (or a) node with distance 0
    shell0 = shells.get(0, [])
    if not shell0:
        probed_index = int(np.argmin(distances))
        print(f"WARNING: no shell-0 nodes found; using node index {probed_index} as center.")
    else:
        probed_index = shell0[0]

    # Global normalization so relative changes are visible
    ent_max = float(ent_by_node.max())
    print(f"  max entanglement over all (t,node): {ent_max:.4f}")

    vmin, vmax = 0.0, 1.0

    # How many frames to render
    n_frames = T if args.max_frames is None else min(T, args.max_frames)
    print(f"Rendering {n_frames} frames into {out_dir} ...")

    for k in range(n_frames):
        t_val = float(times[k])
        ent_t = ent_by_node[k]  # (N,)

        frame_path = out_dir / f"frame_{k:04d}.png"
        make_frame(
            frame_path,
            t_val,
            ent_t,
            ent_max,
            positions,
            distances,
            probed_index,
            vmin=vmin,
            vmax=vmax,
        )

    print("Done.")
    print(f"Frames written to: {out_dir}")


if __name__ == "__main__":
    main()
