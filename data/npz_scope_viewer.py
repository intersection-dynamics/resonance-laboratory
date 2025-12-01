#!/usr/bin/env python3
"""
npz_scope_viewer.py

Drop this script into any directory and run:

    python npz_scope_viewer.py

It will scan for *.npz files and, for each one:

1) If it finds:
       times, states
   it will create:
       <basename>_amps.png
   showing component probabilities |psi_i(t)|^2 vs time.

2) If it finds:
       times, coherence
   it will create:
       <basename>_coherence.png
   showing off-diagonal coherence vs time.

3) If it finds:
       times, entanglement, distances
   it will create:
       <basename>_ent_shells.png  (entanglement vs time by distance shell)
       <basename>_ent_heatmap.png (entanglement heatmap over nodes & time).
"""

import os
import glob
import json
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe_get(arr_dict: Dict[str, Any], key: str):
    if key in arr_dict:
        return arr_dict[key]
    return None


def load_npz(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    result: Dict[str, Any] = {}

    for k in data.files:
        result[k] = data[k]

    if "meta_json" in result:
        raw = result["meta_json"]
        try:
            # meta_json may be a 0-d array of string
            if isinstance(raw, np.ndarray):
                raw = raw.item()
            meta = json.loads(raw)
        except Exception:
            meta = {}
        result["meta"] = meta
    else:
        result["meta"] = {}
    return result


def plot_amplitudes(times: np.ndarray, states: np.ndarray, out_path: str, title_suffix: str = "") -> None:
    """
    Plot component probabilities |psi_i(t)|^2 vs time.
    states shape: (T, d)
    """
    if times.ndim != 1 or states.ndim != 2:
        return

    T, d = states.shape
    if T == 0 or d == 0:
        return

    probs = np.abs(states) ** 2
    denom = np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    probs = probs / denom

    plt.figure(figsize=(8, 5))
    for i in range(d):
        plt.plot(times, probs[:, i], label=f"|psi_{i}|^2")

    plt.xlabel("time")
    plt.ylabel("component probability")
    title = "Component probabilities vs time"
    if title_suffix:
        title += f" ({title_suffix})"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_coherence(times: np.ndarray, coherence: np.ndarray, out_path: str, title_suffix: str = "") -> None:
    """
    Plot decoherence metric vs time.
    """
    if times.ndim != 1 or coherence.ndim != 1:
        return
    if times.size == 0 or coherence.size == 0:
        return

    plt.figure(figsize=(8, 4))
    plt.plot(times, coherence, marker="o", markersize=2, linewidth=1)
    plt.xlabel("time")
    plt.ylabel("off-diagonal coherence (L1 norm)")
    title = "Decoherence metric vs time"
    if title_suffix:
        title += f" ({title_suffix})"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_entanglement_shells(times: np.ndarray,
                             ent: np.ndarray,
                             distances: np.ndarray,
                             out_path: str) -> None:
    """
    Plot entanglement vs time, averaged by distance shells.
    ent shape: (T, N)
    distances shape: (N,)
    """
    if ent.ndim != 2 or times.ndim != 1 or distances.ndim != 1:
        return
    T, N = ent.shape
    if N == 0 or T == 0 or distances.size != N:
        return

    # Group nodes by rounded distance
    dist_rounded = np.round(distances).astype(int)
    shells = sorted(set(int(d) for d in dist_rounded))

    plt.figure(figsize=(8, 5))
    for shell in shells:
        mask = dist_rounded == shell
        if not np.any(mask):
            continue
        shell_ent = ent[:, mask].mean(axis=1)
        plt.plot(times, shell_ent, label=f"dist={shell}")

    plt.xlabel("time")
    plt.ylabel("avg entanglement (per node)")
    plt.title("Entanglement by distance shell")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_entanglement_heatmap(times: np.ndarray,
                              ent: np.ndarray,
                              out_path: str) -> None:
    """
    Heatmap of entanglement: time vs node index.
    ent shape: (T, N)
    """
    if ent.ndim != 2 or times.ndim != 1:
        return
    T, N = ent.shape
    if T == 0 or N == 0:
        return

    plt.figure(figsize=(8, 5))
    # imshow expects (rows, cols) = (Y, X) so we put nodes on y-axis
    plt.imshow(ent.T, aspect="auto", origin="lower",
               extent=[times[0], times[-1], 0, N])
    plt.colorbar(label="entanglement")
    plt.xlabel("time")
    plt.ylabel("node index")
    plt.title("Entanglement heatmap (nodes vs time)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    npz_files = sorted(glob.glob("*.npz"))
    if not npz_files:
        print("No .npz files found in current directory.")
        return

    print("Found .npz files:")
    for f in npz_files:
        print("  ", f)
    print()

    for path in npz_files:
        base = os.path.splitext(os.path.basename(path))[0]
        print(f"Processing {path} ...")

        try:
            data = load_npz(path)
        except Exception as e:
            print(f"  Error loading {path}: {e}")
            continue

        times = _safe_get(data, "times")
        states = _safe_get(data, "states")
        coherence = _safe_get(data, "coherence")
        ent = _safe_get(data, "entanglement")
        distances = _safe_get(data, "distances")

        meta = data.get("meta", {})
        exp_type = meta.get("experiment", "")

        title_suffix = exp_type if exp_type else base

        # 1) Amplitudes plot
        if times is not None and states is not None:
            try:
                amp_fig = f"{base}_amps.png"
                plot_amplitudes(times, states, amp_fig, title_suffix=title_suffix)
                print(f"  Saved amplitudes plot to {amp_fig}")
            except Exception as e:
                print(f"  Skipped amplitudes for {base}: {e}")

        # 2) Coherence plot
        if times is not None and coherence is not None:
            try:
                coh_fig = f"{base}_coherence.png"
                plot_coherence(times, coherence, coh_fig, title_suffix=title_suffix)
                print(f"  Saved coherence plot to {coh_fig}")
            except Exception as e:
                print(f"  Skipped coherence for {base}: {e}")

        # 3) Entanglement wavefront plots
        if times is not None and ent is not None and distances is not None:
            try:
                shells_fig = f"{base}_ent_shells.png"
                plot_entanglement_shells(times, ent, distances, shells_fig)
                print(f"  Saved entanglement shells plot to {shells_fig}")
            except Exception as e:
                print(f"  Skipped ent shells for {base}: {e}")

            try:
                heat_fig = f"{base}_ent_heatmap.png"
                plot_entanglement_heatmap(times, ent, heat_fig)
                print(f"  Saved entanglement heatmap to {heat_fig}")
            except Exception as e:
                print(f"  Skipped ent heatmap for {base}: {e}")

        print()

    print("Done.")


if __name__ == "__main__":
    main()
