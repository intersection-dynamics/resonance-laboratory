#!/usr/bin/env python3
"""
pattern_coexistence_probe.py

Experiment:
  Let the batched Substrate evolve and, at intervals, run pattern detection
  to see whether proton-like, electron-like, and photon-like patterns
  coexist, and how proton-like and electron-like patterns sit relative
  to each other in the graph.

Ontology:
  - Substrate is the Hilbert sea with local DOFs + entanglement.
  - "Proton / electron / photon" are emergent patterns detected by
    pattern_detector.analyze_substrate; nothing is fundamentally pinned.

Outputs:
  Saves pattern_coexistence_probe.npz with:
    - times
    - best candidate ids over time
    - graph distance between proton-like and electron-like candidates
    - per-species best scores
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List

import numpy as np

from substrate import Config, Substrate  # type: ignore
import pattern_detector as pd  # type: ignore


def graph_distance(substrate: Substrate, a_id: int, b_id: int, max_radius: int = 10) -> int:
    """
    BFS distance between nodes a and b in the substrate graph.
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


def run_pattern_coexistence_probe(
    config: Config,
    burn_in_steps: int = 300,
    total_steps: int = 3000,
    record_stride: int = 10,
    max_dist_radius: int = 10,
) -> None:
    print("------------------------------------------------------------")
    print("Pattern coexistence probe")
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

    t = 0.0

    # Initial snapshot
    def record(step_idx: int, t: float):
        feats, scores, cands = pd.analyze_substrate(substrate)

        times.append(t)

        p_ids.append(cands.proton_id)
        e_ids.append(cands.electron_id)
        ph_ids.append(cands.photon_id)

        p_scores.append(cands.proton_score)
        e_scores.append(cands.electron_score)
        ph_scores.append(cands.photon_score)

        d_pe = graph_distance(substrate, cands.proton_id, cands.electron_id, max_radius=max_dist_radius)
        dist_pe.append(d_pe)

        print(
            f"[t={t:.2f}] "
            f"p_id={cands.proton_id} (score={cands.proton_score:.2f}), "
            f"e_id={cands.electron_id} (score={cands.electron_score:.2f}), "
            f"ph_id={cands.photon_id} (score={cands.photon_score:.2f}), "
            f"dist_p-e={d_pe}"
        )

    record(step_idx=0, t=t)

    # Main evolution loop
    for step in range(1, total_steps + 1):
        substrate.evolve(n_steps=1)
        t += dt

        if (step % record_stride == 0) or (step == total_steps):
            record(step_idx=step, t=t)

    # Convert to arrays
    times_arr = np.asarray(times, dtype=float)
    p_ids_arr = np.asarray(p_ids, dtype=int)
    e_ids_arr = np.asarray(e_ids, dtype=int)
    ph_ids_arr = np.asarray(ph_ids, dtype=int)

    p_scores_arr = np.asarray(p_scores, dtype=float)
    e_scores_arr = np.asarray(e_scores, dtype=float)
    ph_scores_arr = np.asarray(ph_scores, dtype=float)

    dist_pe_arr = np.asarray(dist_pe, dtype=int)

    # Simple coexistence stats
    mean_dist = float(np.mean(dist_pe_arr))
    frac_bind1 = float(np.mean(dist_pe_arr <= 1))
    frac_bind2 = float(np.mean(dist_pe_arr <= 2))

    print(f"  Mean proton-like / electron-like distance: {mean_dist:.3f}")
    print(f"  Fraction of time dist<=1 (neighbors):      {frac_bind1:.3f}")
    print(f"  Fraction of time dist<=2:                  {frac_bind2:.3f}")

    meta: Dict[str, object] = {
        "config": asdict(config),
        "burn_in_steps": int(burn_in_steps),
        "total_steps": int(total_steps),
        "record_stride": int(record_stride),
        "max_dist_radius": int(max_dist_radius),
        "mean_pe_distance": mean_dist,
        "fraction_bind1": frac_bind1,
        "fraction_bind2": frac_bind2,
    }

    fname = "pattern_coexistence_probe.npz"
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
        meta_json=json.dumps(meta, indent=2),
    )

    print(f"Saved {fname} (T={len(times_arr)})")


def main():
    config = Config(
        n_nodes=64,
        internal_dim=3,
        monogamy_budget=1.0,
        defrag_rate=0.1,
        seed=321,
    )
    if not hasattr(config, "dt"):
        config.dt = 0.1  # type: ignore[attr-defined]

    print("============================================================")
    print("  Pattern coexistence probe")
    print("============================================================")
    print(
        f"Config: n_nodes={config.n_nodes}, d={config.internal_dim}, "
        f"monogamy={config.monogamy_budget}, defrag={config.defrag_rate}, "
        f"dt={getattr(config, 'dt', 'N/A')}"
    )
    print("============================================================\n")

    run_pattern_coexistence_probe(
        config=config,
        burn_in_steps=300,
        total_steps=3000,
        record_stride=10,
        max_dist_radius=10,
    )

    print("\nPattern coexistence probe complete.")


if __name__ == "__main__":
    main()
