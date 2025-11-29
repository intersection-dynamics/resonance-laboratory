#!/usr/bin/env python3
"""
analyze_hydrogen_transitions.py

Post-process hydrogen_transition_probe.npz to see whether:

  - Up-hops (electron moving "outward": dist_p-e 1 -> 2) look like photon *absorption*,
  - Down-hops (electron moving "inward": dist_p-e 2 -> 1) look like photon *emission*.

We define:
  dist_pe[t]           : proton-like / electron-like graph distance at frame t
  delta_dist_pe[t]     : dist_pe[t] - dist_pe[t-1]  (delta_dist_pe[0] = 0)
  dist_pair_ph_min[t]  : min( dist_p_ph[t], dist_e_ph[t] ), photon distance to the pair

Heuristics:
  - Up-hop:   dist_pe[t-1] = 1, dist_pe[t] = 2
  - Down-hop: dist_pe[t-1] = 2, dist_pe[t] = 1

  - "Photon near":     dist_pair_ph_min <= NEAR_THRESH  (default 2)
  - "Photon far-ish":  dist_pair_ph_min > NEAR_THRESH

  Absorption-like up-hop:
    - photon near before (t-1),
    - photon near at hop (t),
    - photon far-ish after (t+1, if exists).

  Emission-like down-hop:
    - photon far-ish before (t-1),
    - photon near at or just after hop (t or t+1).

We also dump aggregate stats:
  - counts of up/down hops,
  - mean photon distance before/at/after for each hop type,
  - fraction of hops with photon-near before/at/after.
"""

from __future__ import annotations

import json
from typing import List

import numpy as np


NEAR_THRESH = 2  # photons considered "near" if <= 2 hops from pair


def load_data(fname: str = "hydrogen_transition_probe.npz"):
    data = np.load(fname, allow_pickle=True)
    dist_pe = data["dist_pe"]                  # (T,)
    delta_dist_pe = data["delta_dist_pe"]      # (T,)
    dist_pair_ph_min = data["dist_pair_ph_min"]  # (T,)
    times = data["times"]
    meta_json = data["meta_json"].item()
    meta = json.loads(meta_json)
    return times, dist_pe, delta_dist_pe, dist_pair_ph_min, meta


def classify_hops(dist_pe: np.ndarray, delta_dist_pe: np.ndarray):
    """
    Identify up-hops and down-hops.

    We look at frames t >= 1 where |delta_dist_pe[t]| >= 1 and:
      - Up-hop:   dist_pe[t-1] = 1, dist_pe[t] = 2
      - Down-hop: dist_pe[t-1] = 2, dist_pe[t] = 1

    Returns:
      up_indices   : list[int] of frames t that are up-hops
      down_indices : list[int] of frames t that are down-hops
    """
    T = len(dist_pe)
    up_indices: List[int] = []
    down_indices: List[int] = []

    for t in range(1, T):
        d_now = dist_pe[t]
        d_prev = dist_pe[t - 1]
        d_del = delta_dist_pe[t]

        if abs(d_del) < 1:
            continue

        # Up-hop: inner shell -> outer shell
        if d_prev == 1 and d_now == 2:
            up_indices.append(t)
        # Down-hop: outer shell -> inner shell
        elif d_prev == 2 and d_now == 1:
            down_indices.append(t)
        else:
            # weird transition (e.g., 0->1, 1->0, 2->0); ignore for this analysis
            continue

    return up_indices, down_indices


def hop_window_stats(
    hop_indices: List[int], dist_pair_ph_min: np.ndarray, label: str
):
    """
    Compute statistics for photon distance around hops of a given type.

    For each hop at frame t, we look at:
      d_before = dist_pair_ph_min[t-1]
      d_at     = dist_pair_ph_min[t]
      d_after  = dist_pair_ph_min[t+1]  (when t+1 < T)

    Returns a dict of aggregate stats.
    """
    T = len(dist_pair_ph_min)
    if not hop_indices:
        return {
            "n_hops": 0,
        }

    d_before_list = []
    d_at_list = []
    d_after_list = []

    near_before = 0
    near_at = 0
    near_after = 0
    count_after = 0  # some hops might be at last frame, no t+1

    for t in hop_indices:
        if t <= 0 or t >= T:
            continue

        d_before = dist_pair_ph_min[t - 1]
        d_at = dist_pair_ph_min[t]
        d_before_list.append(d_before)
        d_at_list.append(d_at)

        if d_before <= NEAR_THRESH:
            near_before += 1
        if d_at <= NEAR_THRESH:
            near_at += 1

        if t + 1 < T:
            d_after = dist_pair_ph_min[t + 1]
            d_after_list.append(d_after)
            count_after += 1
            if d_after <= NEAR_THRESH:
                near_after += 1

    d_before_arr = np.asarray(d_before_list, dtype=float)
    d_at_arr = np.asarray(d_at_list, dtype=float)
    d_after_arr = np.asarray(d_after_list, dtype=float) if d_after_list else np.zeros(0)

    stats = {
        "label": label,
        "n_hops": len(hop_indices),
        "mean_d_before": float(np.mean(d_before_arr)) if d_before_arr.size else None,
        "mean_d_at": float(np.mean(d_at_arr)) if d_at_arr.size else None,
        "mean_d_after": float(np.mean(d_after_arr)) if d_after_arr.size else None,
        "frac_near_before": float(near_before) / max(len(hop_indices), 1),
        "frac_near_at": float(near_at) / max(len(hop_indices), 1),
        "frac_near_after": float(near_after) / max(count_after, 1) if count_after > 0 else None,
    }

    return stats


def classify_absorption_emission(
    up_indices: List[int],
    down_indices: List[int],
    dist_pair_ph_min: np.ndarray,
):
    """
    Use simple heuristics to classify individual hops as absorption-like or emission-like.

    Absorption-like up-hop (t in up_indices):
      photon near at t-1 AND t,
      photon far-ish at t+1 (if t+1 exists).

    Emission-like down-hop (t in down_indices):
      photon far-ish at t-1,
      photon near at t or t+1.

    Returns:
      dict with counts and fractions.
    """
    T = len(dist_pair_ph_min)

    def is_near(d: float) -> bool:
        return d <= NEAR_THRESH

    # Up-hop: absorption-like
    n_up = len(up_indices)
    n_up_absorption_like = 0

    for t in up_indices:
        if t <= 0 or t >= T:
            continue
        d_before = dist_pair_ph_min[t - 1]
        d_at = dist_pair_ph_min[t]
        if t + 1 < T:
            d_after = dist_pair_ph_min[t + 1]
        else:
            d_after = None

        if is_near(d_before) and is_near(d_at):
            if d_after is None or not is_near(d_after):
                n_up_absorption_like += 1

    # Down-hop: emission-like
    n_down = len(down_indices)
    n_down_emission_like = 0

    for t in down_indices:
        if t <= 0 or t >= T:
            continue
        d_before = dist_pair_ph_min[t - 1]
        d_at = dist_pair_ph_min[t]
        if t + 1 < T:
            d_after = dist_pair_ph_min[t + 1]
        else:
            d_after = None

        # photon far-ish before, near at or just after
        before_far = not is_near(d_before)
        at_near = is_near(d_at)
        after_near = is_near(d_after) if d_after is not None else False

        if before_far and (at_near or after_near):
            n_down_emission_like += 1

    frac_up_absorption_like = float(n_up_absorption_like) / max(n_up, 1) if n_up > 0 else 0.0
    frac_down_emission_like = float(n_down_emission_like) / max(n_down, 1) if n_down > 0 else 0.0

    return {
        "n_up": n_up,
        "n_up_absorption_like": n_up_absorption_like,
        "frac_up_absorption_like": frac_up_absorption_like,
        "n_down": n_down,
        "n_down_emission_like": n_down_emission_like,
        "frac_down_emission_like": frac_down_emission_like,
    }


def main():
    print("Loading hydrogen_transition_probe.npz...")
    times, dist_pe, delta_dist_pe, dist_pair_ph_min, meta = load_data()
    T = len(times)
    print(f"Loaded T = {T} frames.")
    print("Meta:", json.dumps(meta, indent=2))

    up_indices, down_indices = classify_hops(dist_pe, delta_dist_pe)
    print()
    print(f"Total up-hops (1 -> 2):   {len(up_indices)}")
    print(f"Total down-hops (2 -> 1): {len(down_indices)}")

    up_stats = hop_window_stats(up_indices, dist_pair_ph_min, label="up")
    down_stats = hop_window_stats(down_indices, dist_pair_ph_min, label="down")

    print("\n=== Photon distance stats around hops ===")
    for stats in (up_stats, down_stats):
        if stats["n_hops"] == 0:
            print(f"{stats['label']}-hops: none")
            continue
        print(f"{stats['label']}-hops:")
        print(f"  n_hops              : {stats['n_hops']}")
        print(f"  mean d_before       : {stats['mean_d_before']}")
        print(f"  mean d_at           : {stats['mean_d_at']}")
        print(f"  mean d_after        : {stats['mean_d_after']}")
        print(f"  frac photon-near before: {stats['frac_near_before']:.3f}")
        print(f"  frac photon-near at     : {stats['frac_near_at']:.3f}")
        print(f"  frac photon-near after  : {stats['frac_near_after']:.3f}")

    ae_stats = classify_absorption_emission(up_indices, down_indices, dist_pair_ph_min)
    print("\n=== Absorption / emission-like classification ===")
    print(f"Up-hops (1 -> 2):")
    print(f"  n_up                    : {ae_stats['n_up']}")
    print(f"  n_up_absorption_like    : {ae_stats['n_up_absorption_like']}")
    print(f"  frac_up_absorption_like : {ae_stats['frac_up_absorption_like']:.3f}")
    print(f"Down-hops (2 -> 1):")
    print(f"  n_down                  : {ae_stats['n_down']}")
    print(f"  n_down_emission_like    : {ae_stats['n_down_emission_like']}")
    print(f"  frac_down_emission_like : {ae_stats['frac_down_emission_like']:.3f}")


if __name__ == "__main__":
    main()
