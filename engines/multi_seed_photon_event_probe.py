#!/usr/bin/env python3
"""
multi_seed_photon_event_probe.py

Run the photon event probe (around a hydrogen-ish bound pair)
for multiple random seeds and stack all frames into a single
ensemble dataset to reduce noise.

Output:
  photon_event_probe_multi.npz, with:
    times               : (T_total,)
    dist_pe             : (T_total,)
    photon_shell_proton : (T_total, R+1)
    photon_shell_electron:(T_total, R+1)
    seed_ids            : (T_total,)  # which seed each frame came from
    meta_json           : JSON string with config & multi-seed info

Updated for new substrate.py API:
  - substrate._neighbors is list[list[int]], not dict
  - Direct indexing instead of .get()
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np

from substrate import Config, Substrate  # type: ignore
import pattern_detector as pd  # type: ignore


# ---------------------------------------------------------------------
# Graph distance helpers
# ---------------------------------------------------------------------


def bfs_distances(neighbors: List[List[int]], src: int, max_radius: int) -> np.ndarray:
    """
    Compute BFS distances from src to all nodes up to max_radius.

    neighbors: list of neighbor lists indexed by node id
    Returns:
      dist: np.ndarray of shape (N,), dist[i] = graph distance(src, i)
            or max_radius + 1 if further.
    """
    n_nodes = len(neighbors)
    dist = np.full(n_nodes, max_radius + 1, dtype=int)
    if n_nodes == 0:
        return dist

    from collections import deque

    dist[src] = 0
    q = deque([src])
    while q:
        nid = q.popleft()
        d = dist[nid]
        if d >= max_radius:
            continue
        for nb in neighbors[nid]:
            if nb < 0 or nb >= n_nodes:
                continue
            if dist[nb] > d + 1:
                dist[nb] = d + 1
                q.append(nb)
    return dist


def graph_distance(neighbors: List[List[int]], a_id: int, b_id: int, max_radius: int) -> int:
    """
    BFS distance between nodes a and b in the substrate graph.
    """
    if a_id == b_id:
        return 0
    dist = bfs_distances(neighbors, a_id, max_radius=max_radius)
    return int(dist[b_id])


# ---------------------------------------------------------------------
# Single-seed probe (in-memory, no saving)
# ---------------------------------------------------------------------


def run_single_seed_probe(
    base_config: Config,
    seed: int,
    burn_in_steps: int = 300,
    total_steps: int = 3000,
    record_stride: int = 10,
    max_graph_radius: int = 10,
    max_shell_radius: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one photon event probe for a given seed and return:

      times               : (T,)
      dist_pe             : (T,)
      photon_shell_proton : (T, R+1)
      photon_shell_electron:(T, R+1)

    We *don't* save inside this function; the caller is responsible
    for stacking and saving multi-run data.
    """
    config = Config(
        n_nodes=base_config.n_nodes,
        internal_dim=base_config.internal_dim,
        monogamy_budget=base_config.monogamy_budget,
        defrag_rate=base_config.defrag_rate,
        seed=seed,
    )
    if hasattr(base_config, "dt"):
        config.dt = base_config.dt  # type: ignore[attr-defined]
    else:
        if not hasattr(config, "dt"):
            config.dt = 0.1  # type: ignore[attr-defined]

    print("------------------------------------------------------------")
    print(f"  Single-seed photon event probe (seed={seed})")
    print(
        f"  burn_in_steps={burn_in_steps}, total_steps={total_steps}, "
        f"record_stride={record_stride}"
    )
    print("------------------------------------------------------------")

    substrate = Substrate(config)
    neighbors = substrate._neighbors  # list[list[int]]
    n_nodes = substrate.n_nodes

    print("  Burn-in evolution...")
    substrate.evolve(n_steps=burn_in_steps)

    dt = getattr(config, "dt", 0.1)

    times: List[float] = []
    dist_pe: List[int] = []
    photon_shell_proton: List[np.ndarray] = []
    photon_shell_electron: List[np.ndarray] = []

    t = 0.0

    def record(step_idx: int, t: float):
        feats, scores, cands = pd.analyze_substrate(substrate)

        p_id = cands.proton_id
        e_id = cands.electron_id

        times.append(t)

        d_pe = graph_distance(neighbors, p_id, e_id, max_radius=max_graph_radius)
        dist_pe.append(d_pe)

        ph_score = scores.photon_score  # (N,)

        # BFS from proton & electron to define shells
        dist_from_proton = bfs_distances(neighbors, p_id, max_radius=max_shell_radius)
        dist_from_electron = bfs_distances(neighbors, e_id, max_radius=max_shell_radius)

        shell_p = np.zeros(max_shell_radius + 1, dtype=float)
        shell_e = np.zeros(max_shell_radius + 1, dtype=float)

        for nid in range(n_nodes):
            d_p = dist_from_proton[nid]
            d_e = dist_from_electron[nid]

            if d_p <= max_shell_radius:
                shell_p[d_p] += ph_score[nid]
            if d_e <= max_shell_radius:
                shell_e[d_e] += ph_score[nid]

        photon_shell_proton.append(shell_p)
        photon_shell_electron.append(shell_e)

        print(
            f"  [seed={seed}, t={t:.2f}] p_id={p_id}, e_id={e_id}, dist_p-e={d_pe}"
        )

    # initial snapshot
    record(step_idx=0, t=t)

    # main evolution
    for step in range(1, total_steps + 1):
        substrate.evolve(n_steps=1)
        t += dt
        if (step % record_stride == 0) or (step == total_steps):
            record(step_idx=step, t=t)

    times_arr = np.asarray(times, dtype=float)
    dist_pe_arr = np.asarray(dist_pe, dtype=int)
    shell_p_arr = np.stack(photon_shell_proton, axis=0)
    shell_e_arr = np.stack(photon_shell_electron, axis=0)

    mean_pe_distance = float(np.mean(dist_pe_arr))
    frac_bind1 = float(np.mean(dist_pe_arr <= 1))
    frac_bind2 = float(np.mean(dist_pe_arr <= 2))

    print("  --- seed summary ---")
    print(f"  frames: {len(times_arr)}")
    print(f"  mean dist_pe: {mean_pe_distance:.3f}")
    print(f"  frac dist<=1: {frac_bind1:.3f}, dist<=2: {frac_bind2:.3f}")

    return times_arr, dist_pe_arr, shell_p_arr, shell_e_arr


# ---------------------------------------------------------------------
# Multi-seed driver
# ---------------------------------------------------------------------


def main():
    # Base config shared across seeds
    base_config = Config(
        n_nodes=64,
        internal_dim=3,
        monogamy_budget=1.0,
        defrag_rate=0.1,
        seed=0,
    )
    if not hasattr(base_config, "dt"):
        base_config.dt = 0.1  # type: ignore[attr-defined]

    seeds = [987, 1234, 2025, 4242, 7777]  # edit list if you want more/less

    burn_in_steps = 300
    total_steps = 3000
    record_stride = 10
    max_graph_radius = 10
    max_shell_radius = 5

    print("============================================================")
    print("  Multi-seed photon event probe")
    print("============================================================")
    print(
        f"Config: n_nodes={base_config.n_nodes}, d={base_config.internal_dim}, "
        f"monogamy={base_config.monogamy_budget}, defrag={base_config.defrag_rate}, "
        f"dt={base_config.dt}"
    )
    print(f"Seeds: {seeds}")
    print("============================================================\n")

    all_times: List[np.ndarray] = []
    all_dist_pe: List[np.ndarray] = []
    all_shell_p: List[np.ndarray] = []
    all_shell_e: List[np.ndarray] = []
    all_seed_ids: List[np.ndarray] = []

    t_offset = 0.0
    frames_per_seed: List[int] = []

    for s in seeds:
        times_s, dist_s, shell_p_s, shell_e_s = run_single_seed_probe(
            base_config=base_config,
            seed=s,
            burn_in_steps=burn_in_steps,
            total_steps=total_steps,
            record_stride=record_stride,
            max_graph_radius=max_graph_radius,
            max_shell_radius=max_shell_radius,
        )

        # offset times so that multi-run timeline is monotonic
        times_s_offset = times_s + t_offset

        all_times.append(times_s_offset)
        all_dist_pe.append(dist_s)
        all_shell_p.append(shell_p_s)
        all_shell_e.append(shell_e_s)
        all_seed_ids.append(np.full(times_s.shape, s, dtype=int))

        frames_per_seed.append(len(times_s))

        # update offset for next run
        if len(times_s) > 0:
            t_offset = times_s_offset[-1] + base_config.dt

    # Stack all runs
    times_all = np.concatenate(all_times, axis=0)
    dist_pe_all = np.concatenate(all_dist_pe, axis=0)
    shell_p_all = np.concatenate(all_shell_p, axis=0)
    shell_e_all = np.concatenate(all_shell_e, axis=0)
    seed_ids_all = np.concatenate(all_seed_ids, axis=0)

    mean_pe_distance = float(np.mean(dist_pe_all))
    frac_bind1 = float(np.mean(dist_pe_all <= 1))
    frac_bind2 = float(np.mean(dist_pe_all <= 2))

    print("\n============================================================")
    print("  Multi-seed ensemble summary")
    print("============================================================")
    print(f"Total frames: {len(times_all)} across {len(seeds)} seeds")
    print(f"Mean proton/electron distance: {mean_pe_distance:.3f}")
    print(f"Fraction time dist<=1:        {frac_bind1:.3f}")
    print(f"Fraction time dist<=2:        {frac_bind2:.3f}")
    print("Frames per seed:", frames_per_seed)
    print("============================================================")

    meta: Dict[str, object] = {
        "base_config": asdict(base_config),
        "seeds": list(seeds),
        "burn_in_steps": int(burn_in_steps),
        "total_steps": int(total_steps),
        "record_stride": int(record_stride),
        "max_graph_radius": int(max_graph_radius),
        "max_shell_radius": int(max_shell_radius),
        "mean_pe_distance": mean_pe_distance,
        "fraction_bind1": frac_bind1,
        "fraction_bind2": frac_bind2,
        "n_frames_total": int(len(times_all)),
        "frames_per_seed": frames_per_seed,
    }

    fname = "photon_event_probe_multi.npz"
    np.savez(
        fname,
        times=times_all,
        dist_pe=dist_pe_all,
        photon_shell_proton=shell_p_all,
        photon_shell_electron=shell_e_all,
        seed_ids=seed_ids_all,
        meta_json=json.dumps(meta, indent=2),
    )

    print(f"\nSaved multi-seed dataset to {fname}")


if __name__ == "__main__":
    main()