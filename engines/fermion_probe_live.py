#!/usr/bin/env python3
"""
fermion_probe_live.py

Live fermion probe on the Hilbert substrate.

This script:

  - Builds a Substrate(Config) from substrate.py
  - Evolves it for a number of steps with unitary + defrag
  - At the final snapshot, runs pattern_detector.analyze_substrate
    to compute per-node features and species scores
  - Treats high electron_score nodes as "fermion-like" candidates
    (electron-ish lumps) and writes them to CSV.

Outputs:

  outputs/fermion_probe/<run_id>/fermion_candidates.csv
  outputs/fermion_probe/<run_id>/summary.json

Usage (Windows example):

  python fermion_probe_live.py ^
      --n_nodes 64 ^
      --internal_dim 4 ^
      --steps 4000 ^
      --record-stride 100 ^
      --defrag-rate 0.5 ^
      --connectivity 0.4 ^
      --dt 0.1 ^
      --seed 42 ^
      --output-root outputs ^
      --tag fermion_scan

"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

from substrate import Config, Substrate  # type: ignore
from pattern_detector import (  # type: ignore
    NodeFeatures,
    SpeciesScores,
    SpeciesCandidates,
    analyze_substrate,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_run_id(tag: str | None) -> str:
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if tag:
        safe_tag = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in str(tag)
        )
        return f"{now}_{safe_tag}"
    return now


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live fermion probe on the Hilbert substrate."
    )
    p.add_argument("--n_nodes", type=int, default=64, help="Number of substrate nodes.")
    p.add_argument(
        "--internal_dim",
        type=int,
        default=4,
        help="Local Hilbert-space dimension per node.",
    )
    p.add_argument(
        "--steps", type=int, default=4000, help="Number of evolution steps."
    )
    p.add_argument(
        "--record-stride",
        type=int,
        default=0,
        help="If >0, print candidate summaries every this many steps.",
    )
    p.add_argument(
        "--defrag-rate",
        type=float,
        default=0.5,
        help="Defrag rate in substrate.evolve.",
    )
    p.add_argument(
        "--connectivity",
        type=float,
        default=0.4,
        help="Initial edge probability between node pairs.",
    )
    p.add_argument("--dt", type=float, default=0.1, help="Time step dt.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="Number of top electron-like nodes to record as fermion candidates.",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory for fermion_probe outputs.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="fermion_probe",
        help="Tag appended to run id.",
    )
    p.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_true",
        help="Force GPU backend if CuPy is available.",
    )
    p.add_argument(
        "--cpu",
        dest="use_gpu",
        action="store_false",
        help="Force CPU backend (NumPy).",
    )
    p.set_defaults(use_gpu=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build config and substrate
    cfg = Config(
        n_nodes=args.n_nodes,
        internal_dim=args.internal_dim,
        monogamy_budget=1.0,
        defrag_rate=float(args.defrag_rate),
        dt=float(args.dt),
        seed=int(args.seed),
        connectivity=float(args.connectivity),
        use_gpu=args.use_gpu,
    )
    sub = Substrate(cfg)

    run_id = _make_run_id(args.tag)
    run_root = os.path.join(args.output_root, "fermion_probe", run_id)
    data_dir = os.path.join(run_root, "data")
    _ensure_dir(run_root)
    _ensure_dir(data_dir)

    print("=" * 60)
    print("  Fermion Probe Live")
    print("=" * 60)
    print(f"n_nodes:       {args.n_nodes}")
    print(f"internal_dim:  {args.internal_dim}")
    print(f"steps:         {args.steps}")
    print(f"defrag_rate:   {args.defrag_rate}")
    print(f"connectivity:  {args.connectivity}")
    print(f"dt:            {args.dt}")
    print(f"seed:          {args.seed}")
    print(f"backend:       {'GPU' if sub.xp.__name__ == 'cupy' else 'CPU'}")
    print(f"output root:   {run_root}")
    print("=" * 60)

    # Optional periodic logging of candidates
    record_stride = int(args.record_stride)
    if record_stride > 0:
        for step in range(args.steps):
            sub.evolve(n_steps=1, defrag_rate=float(args.defrag_rate))
            if (step + 1) % record_stride == 0:
                feats, scores, cands = analyze_substrate(sub)
                print(
                    f"[step {step+1}] "
                    f"proton_id={cands.proton_id} "
                    f"electron_id={cands.electron_id} "
                    f"photon_id={cands.photon_id} "
                    f"(e_score={cands.electron_score:.3f})"
                )
    else:
        # Single block evolution
        sub.evolve(n_steps=int(args.steps), defrag_rate=float(args.defrag_rate))

    # Final snapshot analysis
    feats, scores, cands = analyze_substrate(sub)

    # Sort nodes by electron_score (fermion-like)
    electron_scores = scores.electron_score
    idx_sorted = np.argsort(electron_scores)[::-1]  # descending
    top_k = int(args.top_k)
    top_idx = idx_sorted[:top_k]

    # Write per-candidate CSV
    cand_csv_path = os.path.join(data_dir, "fermion_candidates.csv")
    with open(cand_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "node_id",
                "electron_score",
                "proton_score",
                "photon_score",
                "degree",
                "clustering",
                "total_entanglement",
                "localization",
                "coherence",
                "excitation",
                "redundancy",
            ]
        )
        for i in top_idx:
            writer.writerow(
                [
                    int(i),
                    float(scores.electron_score[i]),
                    float(scores.proton_score[i]),
                    float(scores.photon_score[i]),
                    float(feats.degree[i]),
                    float(feats.clustering[i]),
                    float(feats.total_entanglement[i]),
                    float(feats.localization[i]),
                    float(feats.coherence[i]),
                    float(feats.excitation[i]),
                    float(feats.redundancy[i]),
                ]
            )

    # Write summary.json with the single best candidates and some stats
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "n_nodes": args.n_nodes,
        "internal_dim": args.internal_dim,
        "steps": args.steps,
        "defrag_rate": args.defrag_rate,
        "connectivity": args.connectivity,
        "dt": args.dt,
        "seed": args.seed,
        "top_k": top_k,
        "best_candidates": {
            "proton": {
                "node_id": int(cands.proton_id),
                "score": float(cands.proton_score),
            },
            "electron": {
                "node_id": int(cands.electron_id),
                "score": float(cands.electron_score),
            },
            "photon": {
                "node_id": int(cands.photon_id),
                "score": float(cands.photon_score),
            },
        },
        "files": {
            "fermion_candidates_csv": cand_csv_path,
        },
    }
    with open(os.path.join(run_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("=" * 60)
    print("Fermion probe complete.")
    print(f"Fermion candidates CSV: {cand_csv_path}")
    print(f"Summary JSON:           {os.path.join(run_root, 'summary.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
