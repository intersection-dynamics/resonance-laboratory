#!/usr/bin/env python3
"""
photon_event_probe.py

Goal:
  Record the *photon-like field* around a hydrogen-ish bound pair,
  so we can later ask:

    - When the electron hops outward (1 -> 2), does a photon-like
      excitation get "kicked" near the atom and propagate outward?
    - When it hops inward (2 -> 1), do we see a different photon
      pattern (emission-like) compared to up-hops?

We:
  - Evolve the batched Substrate.
  - At regular intervals:
      * run pattern detection (proton-like, electron-like, photon-like),
      * record full photon_score[i] per node,
      * compute proton/electron distance,
      * compute radial sums of photon_score in shells around proton and electron.

Outputs:
  Saves photon_event_probe.npz with:
    times                    : (T,)
    proton_ids               : (T,)
    electron_ids             : (T,)
    proton_scores            : (T,)
    electron_scores          : (T,)
    dist_pe                  : (T,)   # graph distance proton<->electron
    photon_scores_all        : (T, N) # per-frame photon_score for each node
    photon_shell_proton      : (T, R+1)
    photon_shell_electron    : (T, R+1)
    meta_json                : JSON string with config & run info

Analysis will be done by analyze_photon_events.py.
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


def bfs_distances(neighbors: Dict[int, List[int]], src: int, max_radius: int) -> np.ndarray:
    """
    Compute BFS distances from src to all nodes up to max_radius.

    Returns:
      dist: np.ndarray of shape (N,), dist[i] = graph distance(src, i)
            or max_radius + 1 if further.
    """
    n_nodes = max(neighbors.keys()) + 1 if neighbors else 0
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
        for nb in neighbors.get(nid, []):
            if dist[nb] > d + 1:
                dist[nb] = d + 1
                q.append(nb)
    return dist


def graph_distance(neighbors: Dict[int, List[int]], a_id: int, b_id: int, max_radius: int) -> int:
    """
    BFS distance between nodes a and b in the substrate graph.
    """
    if a_id == b_id:
        return 0
    dist = bfs_distances(neighbors, a_id, max_radius=max_radius)
    return int(dist[b_id])


# ---------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------


def run_photon_event_probe(
    config: Config,
    burn_in_steps: int = 300,
    total_steps: int = 3000,
    record_stride: int = 10,
    max_graph_radius: int = 10,
    max_shell_radius: int = 5,
) -> None:
    print("------------------------------------------------------------")
    print("Photon event probe (field-level recording around hydrogen-ish pair)")
    print(
        f"burn_in_steps={burn_in_steps}, total_steps={total_steps}, "
        f"record_stride={record_stride}"
    )
    print("------------------------------------------------------------")

    substrate = Substrate(config)
    neighbors = substrate.neighbors  # dict[int, list[int]]
    n_nodes = len(substrate.nodes)

    # Burn-in
    print("Burn-in evolution...")
    substrate.evolve(n_steps=burn_in_steps)

    dt = getattr(config, "dt", 0.1)

    times: List[float] = []

    proton_ids: List[int] = []
    electron_ids: List[int] = []
    proton_scores: List[float] = []
    electron_scores: List[float] = []
    dist_pe: List[int] = []

    # photon_scores_all[t, i] = photon_score for node i at frame t
    photon_scores_all: List[np.ndarray] = []

    # photon_shell_proton[t, r], photon_shell_electron[t, r]
    # r = 0..max_shell_radius
    photon_shell_proton: List[np.ndarray] = []
    photon_shell_electron: List[np.ndarray] = []

    t = 0.0

    def record(step_idx: int, t: float):
        feats, scores, cands = pd.analyze_substrate(substrate)

        times.append(t)

        p_id = cands.proton_id
        e_id = cands.electron_id

        proton_ids.append(p_id)
        electron_ids.append(e_id)
        proton_scores.append(cands.proton_score)
        electron_scores.append(cands.electron_score)

        # Distances & photon_score
        d_pe = graph_distance(neighbors, p_id, e_id, max_radius=max_graph_radius)
        dist_pe.append(d_pe)

        photon_score = scores.photon_score  # (N,)
        photon_scores_all.append(np.asarray(photon_score, dtype=float))

        # BFS from proton & electron to define shells
        dist_from_proton = bfs_distances(neighbors, p_id, max_radius=max_shell_radius)
        dist_from_electron = bfs_distances(neighbors, e_id, max_radius=max_shell_radius)

        shell_p = np.zeros(max_shell_radius + 1, dtype=float)
        shell_e = np.zeros(max_shell_radius + 1, dtype=float)

        for nid in range(n_nodes):
            d_p = dist_from_proton[nid]
            d_e = dist_from_electron[nid]

            if d_p <= max_shell_radius:
                shell_p[d_p] += photon_score[nid]
            if d_e <= max_shell_radius:
                shell_e[d_e] += photon_score[nid]

        photon_shell_proton.append(shell_p)
        photon_shell_electron.append(shell_e)

        print(
            f"[t={t:.2f}] "
            f"p_id={p_id} (score={cands.proton_score:.2f}), "
            f"e_id={e_id} (score={cands.electron_score:.2f}), "
            f"dist_p-e={d_pe}"
        )

    # Initial snapshot
    record(step_idx=0, t=t)

    # Main evolution loop
    for step in range(1, total_steps + 1):
        substrate.evolve(n_steps=1)
        t += dt

        if (step % record_stride == 0) or (step == total_steps):
            record(step_idx=step, t=t)

    # Convert to arrays
    times_arr = np.asarray(times, dtype=float)
    proton_ids_arr = np.asarray(proton_ids, dtype=int)
    electron_ids_arr = np.asarray(electron_ids, dtype=int)
    proton_scores_arr = np.asarray(proton_scores, dtype=float)
    electron_scores_arr = np.asarray(electron_scores, dtype=float)
    dist_pe_arr = np.asarray(dist_pe, dtype=int)

    photon_scores_all_arr = np.stack(photon_scores_all, axis=0)  # (T, N)
    photon_shell_proton_arr = np.stack(photon_shell_proton, axis=0)  # (T, R+1)
    photon_shell_electron_arr = np.stack(photon_shell_electron, axis=0)  # (T, R+1)

    # Summary stats for quick sanity check
    mean_pe_distance = float(np.mean(dist_pe_arr))
    frac_bind1 = float(np.mean(dist_pe_arr <= 1))
    frac_bind2 = float(np.mean(dist_pe_arr <= 2))

    print("------------------------------------------------------------")
    print(f"Total frames recorded:         {len(times_arr)}")
    print(f"Mean proton/electron distance: {mean_pe_distance:.3f}")
    print(f"Fraction time dist<=1:         {frac_bind1:.3f}")
    print(f"Fraction time dist<=2:         {frac_bind2:.3f}")
    print("------------------------------------------------------------")

    meta: Dict[str, object] = {
        "config": asdict(config),
        "burn_in_steps": int(burn_in_steps),
        "total_steps": int(total_steps),
        "record_stride": int(record_stride),
        "max_graph_radius": int(max_graph_radius),
        "max_shell_radius": int(max_shell_radius),
        "mean_pe_distance": mean_pe_distance,
        "fraction_bind1": frac_bind1,
        "fraction_bind2": frac_bind2,
        "n_frames": int(len(times_arr)),
    }

    fname = "photon_event_probe.npz"
    np.savez(
        fname,
        times=times_arr,
        proton_ids=proton_ids_arr,
        electron_ids=electron_ids_arr,
        proton_scores=proton_scores_arr,
        electron_scores=electron_scores_arr,
        dist_pe=dist_pe_arr,
        photon_scores_all=photon_scores_all_arr,
        photon_shell_proton=photon_shell_proton_arr,
        photon_shell_electron=photon_shell_electron_arr,
        meta_json=json.dumps(meta, indent=2),
    )

    print(f"Saved {fname} (T={len(times_arr)})")


def main():
    config = Config(
        n_nodes=64,
        internal_dim=3,
        monogamy_budget=1.0,
        defrag_rate=0.1,
        seed=987,
    )
    if not hasattr(config, "dt"):
        config.dt = 0.1  # type: ignore[attr-defined]

    print("============================================================")
    print("  Photon event probe")
    print("============================================================")
    print(
        f"Config: n_nodes={config.n_nodes}, d={config.internal_dim}, "
        f"monogamy={config.monogamy_budget}, defrag={config.defrag_rate}, "
        f"dt={getattr(config, 'dt', 'N/A')}"
    )
    print("============================================================\n")

    run_photon_event_probe(
        config=config,
        burn_in_steps=300,
        total_steps=3000,
        record_stride=10,
        max_graph_radius=10,
        max_shell_radius=5,
    )

    print("\nPhoton event probe complete.")


if __name__ == "__main__":
    main()
