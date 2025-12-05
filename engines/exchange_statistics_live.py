#!/usr/bin/env python3
"""
exchange_statistics_live.py

Live exchange / exclusion probe for the Hilbert substrate.

This script:

  1) Builds a Substrate(Config) from substrate.py
  2) Evolves for a given number of steps
  3) Runs pattern_detector.analyze_substrate(sub) at the final snapshot
  4) Treats high electron_score nodes as "fermion-like" and
     high photon_score nodes as "boson-like"
  5) Uses substrate_diagnostics to compute:
       - triangle loop phase winding per node
       - internal mode occupancy stats per basis mode
  6) Writes results to:

       outputs/exchange_live/<run_id>/
         summary.json
         params.json
         data/winding_per_node.csv
         data/mode_occupancy.csv

Usage example (Windows):

  python exchange_statistics_live.py ^
      --n_nodes 64 ^
      --internal_dim 4 ^
      --steps 4000 ^
      --defrag-rate 0.5 ^
      --connectivity 0.4 ^
      --dt 0.1 ^
      --seed 42 ^
      --top-k 16 ^
      --prob-threshold 0.5 ^
      --tag exchange_live
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
from pattern_detector import (          # type: ignore
    NodeFeatures,
    SpeciesScores,
    SpeciesCandidates,
    analyze_substrate,
)
from substrate_diagnostics import (     # type: ignore
    compute_triangle_loop_phases,
    summarize_phase_winding,
    compute_mode_occupancy_stats,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


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
        description="Live exchange / exclusion probe on the Hilbert substrate."
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
    p.add_argument("--dt", type=float, default=0.1, help="Time step.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    p.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="Number of top electron/photon nodes to treat as fermion/boson sets.",
    )
    p.add_argument(
        "--prob-threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold p_i[m] >= prob_threshold to count node i "
            "as occupying internal basis mode m."
        ),
    )
    p.add_argument(
        "--pi-window",
        type=float,
        default=float(np.pi / 3.0),
        help="Angular window (radians) to classify phases as near π or near 0.",
    )

    p.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory for exchange_live outputs.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="exchange_live",
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Build config + substrate
    cfg = Config(
        n_nodes=int(args.n_nodes),
        internal_dim=int(args.internal_dim),
        monogamy_budget=1.0,
        defrag_rate=float(args.defrag_rate),
        dt=float(args.dt),
        seed=int(args.seed),
        connectivity=float(args.connectivity),
        use_gpu=args.use_gpu,
    )
    sub = Substrate(cfg)

    run_id = _make_run_id(args.tag)
    run_root = os.path.join(args.output_root, "exchange_live", run_id)
    data_dir = os.path.join(run_root, "data")
    _ensure_dir(run_root)
    _ensure_dir(data_dir)

    print("=" * 60)
    print("  Exchange Statistics (Live)")
    print("=" * 60)
    print(f"n_nodes:       {args.n_nodes}")
    print(f"internal_dim:  {args.internal_dim}")
    print(f"steps:         {args.steps}")
    print(f"defrag_rate:   {args.defrag_rate}")
    print(f"connectivity:  {args.connectivity}")
    print(f"dt:            {args.dt}")
    print(f"seed:          {args.seed}")
    print(f"backend:       {'GPU' if sub.xp.__name__ == 'cupy' else 'CPU'}")
    print(f"top_k:         {args.top_k}")
    print(f"prob_threshold:{args.prob_threshold}")
    print(f"pi_window:     {args.pi_window}")
    print(f"output root:   {run_root}")
    print("=" * 60)

    # Evolve
    sub.evolve(n_steps=int(args.steps), defrag_rate=float(args.defrag_rate))

    # Analyze final snapshot with pattern_detector
    feats, scores, cands = analyze_substrate(sub)

    electron_scores = scores.electron_score
    photon_scores = scores.photon_score

    # Fermion-like = top-k electron_score
    k = int(args.top_k)
    idx_e_sorted = np.argsort(electron_scores)[::-1]
    fermion_nodes = idx_e_sorted[:k]

    # Boson-like = top-k photon_score
    idx_ph_sorted = np.argsort(photon_scores)[::-1]
    boson_nodes = idx_ph_sorted[:k]

    print(f"# fermion-like nodes (top-k electrons): {len(fermion_nodes)}")
    print(f"# boson-like nodes (top-k photons):    {len(boson_nodes)}")

    # ----------------- Phase winding diagnostics -------------------

    tri_data = compute_triangle_loop_phases(sub)
    winding = summarize_phase_winding(tri_data, pi_window=float(args.pi_window))

    winding_csv_path = os.path.join(data_dir, "winding_per_node.csv")
    with open(winding_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "node_id",
                "is_fermion",
                "is_boson",
                "mean_phase",
                "phase_spread",
                "has_pi_winding",
                "has_zero_winding",
                "electron_score",
                "photon_score",
            ]
        )
        fermion_mask = np.zeros(cfg.n_nodes, dtype=bool)
        boson_mask = np.zeros(cfg.n_nodes, dtype=bool)
        fermion_mask[fermion_nodes] = True
        boson_mask[boson_nodes] = True

        for i in range(cfg.n_nodes):
            writer.writerow(
                [
                    i,
                    1 if fermion_mask[i] else 0,
                    1 if boson_mask[i] else 0,
                    float(winding.mean_phase[i])
                    if np.isfinite(winding.mean_phase[i])
                    else float("nan"),
                    float(winding.phase_spread[i])
                    if np.isfinite(winding.phase_spread[i])
                    else float("nan"),
                    1 if winding.has_pi_winding[i] else 0,
                    1 if winding.has_zero_winding[i] else 0,
                    float(electron_scores[i]),
                    float(photon_scores[i]),
                ]
            )

    # ----------------- Mode occupancy diagnostics -----------------

    occ_stats = compute_mode_occupancy_stats(
        sub,
        fermion_nodes=list(fermion_nodes),
        boson_nodes=list(boson_nodes),
        prob_threshold=float(args.prob_threshold),
    )

    occ_csv_path = os.path.join(data_dir, "mode_occupancy.csv")
    with open(occ_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode_index", "occ_fermion", "occ_boson", "occ_all"])
        for m in range(occ_stats.occ_all.shape[0]):
            writer.writerow(
                [
                    m,
                    int(occ_stats.occ_fermion[m]),
                    int(occ_stats.occ_boson[m]),
                    int(occ_stats.occ_all[m]),
                ]
            )

    # ----------------- High-level summary ------------------------

    def _frac(mask: np.ndarray, cond: np.ndarray) -> float:
        idx = np.where(mask)[0]
        if idx.size == 0:
            return float("nan")
        return float(np.sum(cond[idx])) / float(idx.size)

    fermion_mask = np.zeros(cfg.n_nodes, dtype=bool)
    boson_mask = np.zeros(cfg.n_nodes, dtype=bool)
    fermion_mask[fermion_nodes] = True
    boson_mask[boson_nodes] = True

    ferm_pi_frac = _frac(fermion_mask, winding.has_pi_winding)
    ferm_zero_frac = _frac(fermion_mask, winding.has_zero_winding)
    boson_pi_frac = _frac(boson_mask, winding.has_pi_winding)
    boson_zero_frac = _frac(boson_mask, winding.has_zero_winding)

    modes_f_2plus = int(np.sum(occ_stats.occ_fermion >= 2))
    modes_b_2plus = int(np.sum(occ_stats.occ_boson >= 2))
    modes_f_1plus = int(np.sum(occ_stats.occ_fermion >= 1))
    modes_b_1plus = int(np.sum(occ_stats.occ_boson >= 1))

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "n_nodes": cfg.n_nodes,
        "internal_dim": cfg.internal_dim,
        "steps": int(args.steps),
        "dt": float(args.dt),
        "defrag_rate": float(args.defrag_rate),
        "connectivity": float(args.connectivity),
        "seed": int(args.seed),
        "top_k": int(args.top_k),
        "prob_threshold": float(args.prob_threshold),
        "pi_window": float(args.pi_window),
        "fermion_nodes": [int(i) for i in fermion_nodes],
        "boson_nodes": [int(i) for i in boson_nodes],
        "phase_winding_summary": {
            "fermion_pi_fraction": ferm_pi_frac,
            "fermion_zero_fraction": ferm_zero_frac,
            "boson_pi_fraction": boson_pi_frac,
            "boson_zero_fraction": boson_zero_frac,
        },
        "mode_occupancy_summary": {
            "total_modes": int(occ_stats.occ_all.shape[0]),
            "modes_with_fermion_occupancy_ge_1": modes_f_1plus,
            "modes_with_fermion_occupancy_ge_2": modes_f_2plus,
            "modes_with_boson_occupancy_ge_1": modes_b_1plus,
            "modes_with_boson_occupancy_ge_2": modes_b_2plus,
            "total_fermions": int(occ_stats.total_fermions),
            "total_bosons": int(occ_stats.total_bosons),
            "total_nodes": int(occ_stats.total_nodes),
        },
    }

    with open(os.path.join(run_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with open(os.path.join(run_root, "params.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": cfg.__dict__,
                "args": vars(args),
                "run_id": run_id,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print("=" * 60)
    print("Exchange statistics (live) complete.")
    print(f"Winding CSV:      {winding_csv_path}")
    print(f"Mode occupancy:   {occ_csv_path}")
    print(f"Summary JSON:     {os.path.join(run_root, 'summary.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
