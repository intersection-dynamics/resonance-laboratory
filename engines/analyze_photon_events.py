#!/usr/bin/env python3
"""
analyze_photon_events.py

Post-process photon_event_probe.npz to answer:

  "When the bound electron hops (1->2 or 2->1),
   how does the *photon-like field* around the atom change?"

We use:
  dist_pe[t]                 : proton/electron graph distance at frame t
  photon_shell_proton[t, r]  : sum of photon_score over nodes at distance r from proton
  photon_shell_electron[t, r]: same around electron

We:
  - Build baseline shell profiles (time-averaged) around proton & electron.
  - Identify hop frames:
      up-hops:   1 -> 2
      down-hops: 2 -> 1
  - For each hop frame t, look at:
      shell_before = shell[t-1]
      shell_at     = shell[t]
      shell_after  = shell[t+1] (when available)
    and compute deviations from baseline:
      Δshell = shell - baseline
  - Aggregate over up-hops and down-hops to see if there is a systematic
    pattern of photon-field changes that differs by hop direction.

Note:
  This is still heuristic/statistical; we're not calling anything a
  "photon event" yet, just seeing whether the photon-like field responds
  differently to inward vs outward transitions.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np


def load_data(fname: str = "photon_event_probe.npz"):
    data = np.load(fname, allow_pickle=True)
    times = data["times"]  # (T,)
    dist_pe = data["dist_pe"]  # (T,)
    photon_shell_proton = data["photon_shell_proton"]  # (T, R+1)
    photon_shell_electron = data["photon_shell_electron"]  # (T, R+1)
    meta_json = data["meta_json"].item()
    meta = json.loads(meta_json)
    return times, dist_pe, photon_shell_proton, photon_shell_electron, meta


def classify_hops(dist_pe: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Identify up-hops and down-hops from dist_pe[t].

    Framework:
      dist_pe[t] is typically in {1, 2} for a bound pair.
      Up-hop:   1 -> 2  (electron moves outward)
      Down-hop: 2 -> 1  (electron moves inward)

    We look for frames t >= 1 where dist_pe[t-1] != dist_pe[t]:
      - If  dist_pe[t-1] = 1 and dist_pe[t] = 2  -> up-hop at t
      - If  dist_pe[t-1] = 2 and dist_pe[t] = 1  -> down-hop at t
    All other transitions (0->1, 1->0, etc.) are ignored.
    """
    T = len(dist_pe)
    up_indices: List[int] = []
    down_indices: List[int] = []
    for t in range(1, T):
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
    For a given set of hop frames and shell data:

      shell[t, r] = photon_shell (around proton or electron)

    We compute:
      baseline_shell[r] = time-averaged shell over all frames
      For each hop t:
        shell_before = shell[t-1]
        shell_at     = shell[t]
        shell_after  = shell[t+1] (if exists)
        Δbefore = shell_before - baseline_shell
        Δat     = shell_at - baseline_shell
        Δafter  = shell_after - baseline_shell

    Then return aggregated means over hops:

      mean_Δbefore[r], mean_Δat[r], mean_Δafter[r]

    plus the baseline.
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
    """
    Pretty-print shell statistics for either proton-centered or
    electron-centered shells and either up- or down-hops.
    """
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
    print("Loading photon_event_probe.npz...")
    times, dist_pe, shell_p, shell_e, meta = load_data()
    T, Rplus1 = shell_p.shape
    print(f"Loaded T = {T} frames, shell radius R = {Rplus1-1}.")
    print("Meta:", json.dumps(meta, indent=2))

    up_indices, down_indices = classify_hops(dist_pe)
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
        "  - Positive meanΔ_at / meanΔ_after at larger r for down-hops could hint at\n"
        "    emission-like outward photon field compared to baseline.\n"
        "  - Positive meanΔ_before at small r for up-hops could hint at\n"
        "    absorption-like incoming photon field before the hop.\n"
        "But right now we're just exposing the shell-resolved Δ field so we can see\n"
        "whether up vs down hops look different in any systematic way."
    )


if __name__ == "__main__":
    main()
