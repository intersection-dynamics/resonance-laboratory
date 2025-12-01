#!/usr/bin/env python3
"""
analyze_photon_events_multi.py

Like analyze_photon_events.py, but works on the stacked multi-seed
ensemble in photon_event_probe_multi.npz and avoids mixing hops
across different seeds.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np


def load_data(fname: str = "photon_event_probe_multi.npz"):
    data = np.load(fname, allow_pickle=True)
    times = data["times"]  # (T_total,)
    dist_pe = data["dist_pe"]  # (T_total,)
    shell_p = data["photon_shell_proton"]  # (T_total, R+1)
    shell_e = data["photon_shell_electron"]  # (T_total, R+1)
    seed_ids = data["seed_ids"]  # (T_total,)
    meta_json = data["meta_json"].item()
    meta = json.loads(meta_json)
    return times, dist_pe, shell_p, shell_e, seed_ids, meta


def classify_hops_multi(dist_pe: np.ndarray, seed_ids: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Identify up-hops and down-hops from dist_pe[t], making sure we
    do NOT cross seed boundaries.

    Up-hop:   1 -> 2 (within same seed)
    Down-hop: 2 -> 1 (within same seed)
    """
    T = len(dist_pe)
    up_indices: List[int] = []
    down_indices: List[int] = []
    for t in range(1, T):
        if seed_ids[t] != seed_ids[t - 1]:
            continue  # skip cross-seed boundary
        d_prev = int(dist_pe[t - 1])
        d_now = int(dist_pe[t])
        if d_prev == d_now:
            continue
        if d_prev == 1 and d_now == 2:
            up_indices.append(t)
        elif d_prev == 2 and d_now == 1:
            down_indices.append(t)
    return up_indices, down_indices


def aggregate_shell_deltas(
    shell: np.ndarray,  # (T, R+1)
    hop_indices: List[int],
    label: str,
) -> Dict[str, np.ndarray]:
    """
    Same as in the single-run analyzer: compute baseline shell profile,
    then averaged Δshell before/at/after hops.
    """
    T, Rplus1 = shell.shape
    if T == 0 or Rplus1 == 0:
        raise ValueError("Shell array is empty.")

    baseline = np.mean(shell, axis=0)  # (R+1,)

    if not hop_indices:
        return {
            "label": label,
            "baseline": baseline,
            "mean_Δbefore": np.zeros_like(baseline),
            "mean_Δat": np.zeros_like(baseline),
            "mean_Δafter": np.zeros_like(baseline),
            "n_hops": 0,
        }

    deltas_before = []
    deltas_at = []
    deltas_after = []

    for t in hop_indices:
        if t <= 0 or t >= T:
            continue
        shell_before = shell[t - 1]
        shell_at = shell[t]
        Δbefore = shell_before - baseline
        Δat = shell_at - baseline
        deltas_before.append(Δbefore)
        deltas_at.append(Δat)
        if t + 1 < T:
            shell_after = shell[t + 1]
            Δafter = shell_after - baseline
            deltas_after.append(Δafter)

    mean_Δbefore = np.mean(np.stack(deltas_before, axis=0), axis=0) if deltas_before else np.zeros_like(baseline)
    mean_Δat = np.mean(np.stack(deltas_at, axis=0), axis=0) if deltas_at else np.zeros_like(baseline)
    mean_Δafter = np.mean(np.stack(deltas_after, axis=0), axis=0) if deltas_after else np.zeros_like(baseline)

    return {
        "label": label,
        "baseline": baseline,
        "mean_Δbefore": mean_Δbefore,
        "mean_Δat": mean_Δat,
        "mean_Δafter": mean_Δafter,
        "n_hops": len(hop_indices),
    }


def print_shell_stats(name: str, stats: Dict[str, np.ndarray]) -> None:
    label = stats["label"]
    baseline = stats["baseline"]
    Δb = stats["mean_Δbefore"]
    Δa = stats["mean_Δat"]
    Δf = stats["mean_Δafter"]
    n_hops = stats["n_hops"]
    Rplus1 = baseline.shape[0]

    print(f"\n{name} ({label}-hops):")
    print(f"  n_hops: {n_hops}")
    print("  r  baseline    meanΔ_before   meanΔ_at       meanΔ_after")
    for r in range(Rplus1):
        print(
            f"  {r:1d}  {baseline[r]:9.4f}  "
            f"{Δb[r]:12.4e}  {Δa[r]:12.4e}  {Δf[r]:12.4e}"
        )


def main():
    print("Loading photon_event_probe_multi.npz...")
    times, dist_pe, shell_p, shell_e, seed_ids, meta = load_data()
    T, Rplus1 = shell_p.shape
    print(f"Loaded T_total = {T} frames, shell radius R = {Rplus1-1}.")
    print("Meta:", json.dumps(meta, indent=2))

    up_indices, down_indices = classify_hops_multi(dist_pe, seed_ids)
    print()
    print(f"Total up-hops (1 -> 2):   {len(up_indices)}")
    print(f"Total down-hops (2 -> 1): {len(down_indices)}")

    # Proton-centered shells
    up_stats_p = aggregate_shell_deltas(shell_p, up_indices, label="up")
    down_stats_p = aggregate_shell_deltas(shell_p, down_indices, label="down")

    # Electron-centered shells
    up_stats_e = aggregate_shell_deltas(shell_e, up_indices, label="up")
    down_stats_e = aggregate_shell_deltas(shell_e, down_indices, label="down")

    print("\n=== Photon shell statistics around proton (center-of-mass-ish) ===")
    print_shell_stats("proton-centered shells", up_stats_p)
    print_shell_stats("proton-centered shells", down_stats_p)

    print("\n=== Photon shell statistics around electron ===")
    print_shell_stats("electron-centered shells", up_stats_e)
    print_shell_stats("electron-centered shells", down_stats_e)

    print(
        "\nInterpretation hint:\n"
        "  With multiple seeds stacked, any systematic differences between up- and\n"
        "  down-hops in the shell Δ profiles should stand out more sharply and be\n"
        "  less sensitive to peculiarities of a single substrate realization."
    )


if __name__ == "__main__":
    main()
