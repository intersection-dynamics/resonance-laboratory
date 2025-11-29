#!/usr/bin/env python3
"""
hydrogen_transition_probe.py

Goal:
  Climb the ladder from simple "coexistence" to actual
  "transition + photon" structure.

We:
  - Evolve the batched Substrate.
  - At regular intervals, run pattern detection (proton-like, electron-like, photon-like).
  - Track:
      * proton/electron graph distance (dist_pe)
      * changes in that distance (delta_dist_pe "shell hops")
      * motion of each pattern (graph distance of ids vs previous snapshot)
      * photon distance to proton/electron and to the pair (min distance)
  - Look for correlation between:
      * electron shell hops (|delta_dist_pe| >= 1)
      * photon-like patterns being near the proton/electron pair.

Outputs:
  Writes hydrogen_transition_probe.npz containing:
    times
    proton_ids, electron_ids, photon_ids
    dist_pe, delta_dist_pe
    p_move, e_move, ph_move
    dist_p_ph, dist_e_ph, dist_pair_ph_min
    plus summary stats in meta_json.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List

import numpy as np

from substrate import Config, Substrate  # type: ignore
import pattern_detector as pd  # type: ignore


# ---------------------------------------------------------------------
# Graph distance helper
# ---------------------------------------------------------------------


def graph_distance(substrate: Substrate, a_id: int, b_id: int, max_radius: int = 10) -> int:
    """
    BFS distance between nodes a and b in the substrate graph.
    If not found within max_radius, return max_radius + 1.
    """
    if a_id == b_id:
        return 0

    neighbors = substrate.neighbors  # dict[int, list[int]]
    from collections import deque

    visited = {a_id}
    q = deque([(a_id, 0)])

    while q:
        nid, d = q.popleft()
        if d >= max_radius:
            continue
        for nb in neighbors.get(nid, []):
            if nb == b_id:
                return d + 1
            if nb not in visited:
                visited.add(nb)
                q.append((nb, d + 1))

    return max_radius + 1  # treat as "far"


# ---------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------


def run_hydrogen_transition_probe(
    config: Config,
    burn_in_steps: int = 300,
    total_steps: int = 3000,
    record_stride: int = 10,
    max_dist_radius: int = 10,
    photon_near_threshold: int = 2,
) -> None:
    print("------------------------------------------------------------")
    print("Hydrogen transition + photon correlation probe")
    print(
        f"burn_in_steps={burn_in_steps}, total_steps={total_steps}, "
        f"record_stride={record_stride}"
    )
    print("------------------------------------------------------------")

    substrate = Substrate(config)

    # Burn-in
    print("Burn-in evolution...")
    substrate.evolve(n_steps=burn_in_steps)

    dt = getattr(config, "dt", 0.1)

    # Time series containers
    times: List[float] = []

    p_ids: List[int] = []
    e_ids: List[int] = []
    ph_ids: List[int] = []

    p_scores: List[float] = []
    e_scores: List[float] = []
    ph_scores: List[float] = []

    dist_pe: List[int] = []

    p_move: List[int] = []
    e_move: List[int] = []
    ph_move: List[int] = []

    dist_p_ph: List[int] = []
    dist_e_ph: List[int] = []
    dist_pair_ph_min: List[int] = []

    t = 0.0

    # To compute "movement" and delta_dist_pe we need previous snapshot
    prev_p_id: int | None = None
    prev_e_id: int | None = None
    prev_ph_id: int | None = None
    prev_dist_pe: int | None = None

    def record(step_idx: int, t: float):
        nonlocal prev_p_id, prev_e_id, prev_ph_id, prev_dist_pe

        feats, scores, cands = pd.analyze_substrate(substrate)

        times.append(t)

        p_id = cands.proton_id
        e_id = cands.electron_id
        ph_id = cands.photon_id

        p_ids.append(p_id)
        e_ids.append(e_id)
        ph_ids.append(ph_id)

        p_scores.append(cands.proton_score)
        e_scores.append(cands.electron_score)
        ph_scores.append(cands.photon_score)

        # Distances
        d_pe = graph_distance(substrate, p_id, e_id, max_radius=max_dist_radius)
        d_p_ph = graph_distance(substrate, p_id, ph_id, max_radius=max_dist_radius)
        d_e_ph = graph_distance(substrate, e_id, ph_id, max_radius=max_dist_radius)

        dist_pe.append(d_pe)
        dist_p_ph.append(d_p_ph)
        dist_e_ph.append(d_e_ph)
        dist_pair_ph_min.append(min(d_p_ph, d_e_ph))

        # Movement (distance between current and previous ids)
        if prev_p_id is None:
            p_move.append(0)
            e_move.append(0)
            ph_move.append(0)
            delta_d_pe = 0
        else:
            p_move.append(graph_distance(substrate, prev_p_id, p_id, max_radius=max_dist_radius))
            e_move.append(graph_distance(substrate, prev_e_id, e_id, max_radius=max_dist_radius))
            ph_move.append(graph_distance(substrate, prev_ph_id, ph_id, max_radius=max_dist_radius))
            delta_d_pe = d_pe - int(prev_dist_pe)

        # Log a compact summary
        print(
            f"[t={t:.2f}] "
            f"p_id={p_id} (score={cands.proton_score:.2f}), "
            f"e_id={e_id} (score={cands.electron_score:.2f}), "
            f"ph_id={ph_id} (score={cands.photon_score:.2f}), "
            f"dist_p-e={d_pe}, Δdist_p-e={delta_d_pe}, "
            f"dist_pair-ph_min={dist_pair_ph_min[-1]}"
        )

        # Update prev
        prev_p_id = p_id
        prev_e_id = e_id
        prev_ph_id = ph_id
        prev_dist_pe = d_pe

    # Initial snapshot
    record(step_idx=0, t=t)

    # Main evolution loop
    for step in range(1, total_steps + 1):
        substrate.evolve(n_steps=1)
        t += dt

        if (step % record_stride == 0) or (step == total_steps):
            record(step_idx=step, t=t)

    # Convert lists to arrays
    times_arr = np.asarray(times, dtype=float)
    p_ids_arr = np.asarray(p_ids, dtype=int)
    e_ids_arr = np.asarray(e_ids, dtype=int)
    ph_ids_arr = np.asarray(ph_ids, dtype=int)

    p_scores_arr = np.asarray(p_scores, dtype=float)
    e_scores_arr = np.asarray(e_scores, dtype=float)
    ph_scores_arr = np.asarray(ph_scores, dtype=float)

    dist_pe_arr = np.asarray(dist_pe, dtype=int)
    dist_p_ph_arr = np.asarray(dist_p_ph, dtype=int)
    dist_e_ph_arr = np.asarray(dist_e_ph, dtype=int)
    dist_pair_ph_min_arr = np.asarray(dist_pair_ph_min, dtype=int)

    p_move_arr = np.asarray(p_move, dtype=int)
    e_move_arr = np.asarray(e_move, dtype=int)
    ph_move_arr = np.asarray(ph_move, dtype=int)

    # Derived: delta distance proton-electron between snapshots
    delta_dist_pe_arr = np.zeros_like(dist_pe_arr, dtype=int)
    delta_dist_pe_arr[1:] = dist_pe_arr[1:] - dist_pe_arr[:-1]

    # -----------------------------------------------------------------
    # Event statistics
    # -----------------------------------------------------------------

    # Electron "shell hops": frames where |Δdist_p-e| >= 1
    hop_mask = np.abs(delta_dist_pe_arr) >= 1
    n_frames = len(times_arr)
    n_hops = int(np.sum(hop_mask))

    # Photon "near pair": min distance <= photon_near_threshold
    photon_near_mask = dist_pair_ph_min_arr <= photon_near_threshold

    # Hops with photon near
    hop_and_photon_near_mask = hop_mask & photon_near_mask
    n_hops_with_photon_near = int(np.sum(hop_and_photon_near_mask))

    frac_hops = float(n_hops) / max(n_frames, 1)
    frac_hops_with_photon_near = float(n_hops_with_photon_near) / max(n_hops, 1) if n_hops > 0 else 0.0

    mean_dist_pe = float(np.mean(dist_pe_arr))
    frac_dist1 = float(np.mean(dist_pe_arr <= 1))
    frac_dist2 = float(np.mean(dist_pe_arr <= 2))

    print("------------------------------------------------------------")
    print(f"Total frames recorded:                {n_frames}")
    print(f"Mean proton-like / electron-like dist:{mean_dist_pe:.3f}")
    print(f"Fraction time dist<=1 (neighbors):    {frac_dist1:.3f}")
    print(f"Fraction time dist<=2:                {frac_dist2:.3f}")
    print(f"Total 'shell hop' events (|Δdist|>=1): {n_hops}")
    print(f"Fraction of frames with a hop:        {frac_hops:.3f}")
    print(
        f"Fraction of hops with photon-near (≤{photon_near_threshold}): "
        f"{frac_hops_with_photon_near:.3f}"
    )
    print("------------------------------------------------------------")

    # Metadata for easier reading later
    meta: Dict[str, object] = {
        "config": asdict(config),
        "burn_in_steps": int(burn_in_steps),
        "total_steps": int(total_steps),
        "record_stride": int(record_stride),
        "max_dist_radius": int(max_dist_radius),
        "photon_near_threshold": int(photon_near_threshold),
        "mean_pe_distance": mean_dist_pe,
        "fraction_bind1": frac_dist1,
        "fraction_bind2": frac_dist2,
        "n_frames": n_frames,
        "n_hops": n_hops,
        "fraction_frames_with_hop": frac_hops,
        "n_hops_with_photon_near": n_hops_with_photon_near,
        "fraction_hops_with_photon_near": frac_hops_with_photon_near,
    }

    fname = "hydrogen_transition_probe.npz"
    np.savez(
        fname,
        times=times_arr,
        proton_ids=p_ids_arr,
        electron_ids=e_ids_arr,
        photon_ids=ph_ids_arr,
        proton_scores=p_scores_arr,
        electron_scores=e_scores_arr,
        photon_scores=ph_scores_arr,
        dist_pe=dist_pe_arr,
        delta_dist_pe=delta_dist_pe_arr,
        proton_move=p_move_arr,
        electron_move=e_move_arr,
        photon_move=ph_move_arr,
        dist_p_ph=dist_p_ph_arr,
        dist_e_ph=dist_e_ph_arr,
        dist_pair_ph_min=dist_pair_ph_min_arr,
        meta_json=json.dumps(meta, indent=2),
    )

    print(f"Saved {fname} (T={len(times_arr)})")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    config = Config(
        n_nodes=64,
        internal_dim=3,
        monogamy_budget=1.0,
        defrag_rate=0.1,
        seed=654,
    )
    if not hasattr(config, "dt"):
        config.dt = 0.1  # type: ignore[attr-defined]

    print("============================================================")
    print("  Hydrogen transition + photon correlation probe")
    print("============================================================")
    print(
        f"Config: n_nodes={config.n_nodes}, d={config.internal_dim}, "
        f"monogamy={config.monogamy_budget}, defrag={config.defrag_rate}, "
        f"dt={getattr(config, 'dt', 'N/A')}"
    )
    print("============================================================\n")

    run_hydrogen_transition_probe(
        config=config,
        burn_in_steps=300,
        total_steps=3000,
        record_stride=10,
        max_dist_radius=10,
        photon_near_threshold=2,
    )

    print("\nHydrogen transition probe complete.")


if __name__ == "__main__":
    main()
