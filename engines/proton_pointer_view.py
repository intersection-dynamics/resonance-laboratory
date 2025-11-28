#!/usr/bin/env python3
"""
proton_pointer_view.py

Visualize the results from proton_pointer_scope.npz:

- Component probabilities |psi_i(t)|^2 vs time
- Decoherence metric vs time

Requires matplotlib. If you don't have it, install via:
    pip install matplotlib
"""

import json
import os
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt


def load_results(path: str = "proton_pointer_scope.npz") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    data = np.load(path, allow_pickle=True)

    meta_json = data.get("meta_json", "{}").item()
    try:
        meta = json.loads(meta_json)
    except Exception:
        meta = {}

    results = {
        "times": data["times"],
        "states": data["states"],
        "original_state": data["original_state"],
        "pointer_state": data["pointer_state"],
        "amps_original": data["amps_original"],
        "amps_pointer": data["amps_pointer"],
        "coherence": data["coherence"],
        "scalogram_widths": data["scalogram_widths"],
        "scalogram_cwt": data["scalogram_cwt"],
        "meta": meta,
    }
    return results


def plot_amplitudes(times: np.ndarray, states: np.ndarray, out_path: str) -> None:
    """
    Plot component probabilities |psi_i(t)|^2 vs time.
    states shape: (T, d)
    """
    T, d = states.shape
    probs = np.abs(states) ** 2  # (T, d)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)

    plt.figure(figsize=(8, 5))
    for i in range(d):
        plt.plot(times, probs[:, i], label=f"|psi_{i}|^2")

    plt.xlabel("time")
    plt.ylabel("component probability")
    plt.title("Proton pointer: component probabilities vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_coherence(times: np.ndarray, coherence: np.ndarray, out_path: str) -> None:
    """
    Plot decoherence metric vs time.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(times, coherence, marker="o", markersize=2, linewidth=1)
    plt.xlabel("time")
    plt.ylabel("off-diagonal coherence (L1 norm)")
    plt.title("Proton pointer: decoherence metric vs time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    results = load_results("proton_pointer_scope.npz")

    times = results["times"]
    states = results["states"]
    coherence = results["coherence"]
    amps_original = results["amps_original"]
    amps_pointer = results["amps_pointer"]
    meta = results["meta"]

    print("=" * 72)
    print("PROTON POINTER VIEW")
    print("=" * 72)

    # Basic meta info
    config = meta.get("config", {})
    print("Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    print("Proton candidate:")
    print(f"  node id:       {meta.get('proton_node_id')}")
    print(f"  degree:        {meta.get('degree')}")
    print(f"  entanglement:  {meta.get('total_entanglement')}")
    print(f"  p_internal:    {meta.get('internal_probs')}")
    print(f"  S_internal:    {meta.get('internal_entropy')} (norm: {meta.get('internal_entropy_normalized')})")
    print()

    print("Pointer collapse:")
    print(f"  original amps: {np.round(amps_original, 4)}")
    print(f"  pointer amps:  {np.round(amps_pointer, 4)}")
    print()

    # Plot probabilities vs time
    amp_fig = "proton_pointer_amplitudes.png"
    plot_amplitudes(times, states, amp_fig)
    print(f"Saved component probabilities plot to: {os.path.abspath(amp_fig)}")

    # Plot coherence vs time
    coh_fig = "proton_pointer_coherence.png"
    plot_coherence(times, coherence, coh_fig)
    print(f"Saved decoherence metric plot to:     {os.path.abspath(coh_fig)}")
    print()

    # Tiny note if scalogram is present
    if results["scalogram_widths"].size > 0 and results["scalogram_cwt"].size > 0:
        print("Scalogram data present in NPZ (not plotted here).")
    else:
        print("No scalogram data present (SciPy CWT not available or skipped).")


if __name__ == "__main__":
    main()
