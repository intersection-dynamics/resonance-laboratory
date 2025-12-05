#!/usr/bin/env python3
"""
exchange_statistics_probe.py

Experiment: probe exchange-like statistics, exclusion, and indistinguishability
for emergent excitations in the Substrate.

This script combines:

  - pattern_detector run (for the dynamics config and time grid)
  - particle_cluster_analysis run (for node->cluster labeling)
  - substrate_diagnostics (for phase winding + mode occupancy)

At a chosen time step, it:

  1) Reconstructs the Substrate with the same Config as the pattern run.
  2) Evolves it up to the selected step.
  3) Uses the cluster assignments at that step to classify nodes as:
         fermion-like vs boson-like
     (either from explicit --fermion-clusters/--boson-clusters, or inferred
      from the cluster summary: electron-dominated vs photon-dominated).
  4) Runs diagnostics:
       - Triangle loop phases + phase winding summary per node.
       - Mode occupancy statistics per internal basis mode, separately
         for fermion and boson node sets.
  5) Writes:

       outputs/exchange_statistics_probe/<run_id>/
         params.json
         summary.json
         data/winding_per_node.csv
         data/mode_occupancy.csv

Usage example (Windows):

  python exchange_statistics_probe.py ^
      --pattern-run 20251201T190712Z_proton_electron_photon_scan ^
      --cluster-run 20251201T191609Z_quark_gluon_probe ^
      --step -1 ^
      --output-root outputs ^
      --tag exchange_stats

If --step is negative, the script uses the maximal step found in
cluster_assignments.csv (typically the final snapshot).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Sequence

import numpy as np

from substrate import Config, Substrate  # type: ignore
from substrate_diagnostics import (  # type: ignore
    compute_triangle_loop_phases,
    summarize_phase_winding,
    compute_mode_occupancy_stats,
)


# ---------------------------------------------------------------------
# Small helpers
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


def _write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------
# Cluster metadata loading
# ---------------------------------------------------------------------


@dataclass
class ClusterSpeciesMeta:
    fermion_clusters: List[int]
    boson_clusters: List[int]


def _load_cluster_species_meta(summary_path: str) -> ClusterSpeciesMeta:
    """
    Load cluster summary.json from particle_cluster_analysis and infer
    fermion-like vs boson-like cluster IDs.

    Heuristics:
      - fermion-like: electron_like_fraction >= 0.5 and photon_like_fraction <= 0.2
      - boson-like:   photon_like_fraction  >= 0.5
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    fermion_clusters: List[int] = []
    boson_clusters: List[int] = []

    clusters = summary.get("clusters", [])
    for c in clusters:
        cid = int(c.get("cluster_id"))
        species = c.get("species_domination", {})
        e_frac = float(species.get("electron_like_fraction", 0.0))
        ph_frac = float(species.get("photon_like_fraction", 0.0))

        if e_frac >= 0.5 and ph_frac <= 0.2:
            fermion_clusters.append(cid)
        if ph_frac >= 0.5:
            boson_clusters.append(cid)

    return ClusterSpeciesMeta(
        fermion_clusters=fermion_clusters,
        boson_clusters=boson_clusters,
    )


# ---------------------------------------------------------------------
# Cluster assignments loading
# ---------------------------------------------------------------------


@dataclass
class AssignmentData:
    steps: np.ndarray
    nodes: np.ndarray
    clusters: np.ndarray


def _load_cluster_assignments(csv_path: str) -> AssignmentData:
    """
    Load cluster_assignments.csv from particle_cluster_analysis.

    CSV columns:
      step,t,node_id,cluster_id,proton_score,electron_score,photon_score
    """
    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    steps = data["step"].astype(int)
    nodes = data["node_id"].astype(int)
    clusters = data["cluster_id"].astype(int)
    return AssignmentData(steps=steps, nodes=nodes, clusters=clusters)


def _parse_cluster_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(tok) for tok in s.split(",") if tok.strip()]


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Probe exchange-like statistics and exclusion for emergent "
            "fermion- vs boson-like excitations in the Substrate."
        )
    )

    p.add_argument(
        "--pattern-root",
        type=str,
        default="outputs/pattern_detector_scan",
        help="Root directory where pattern_detector runs are stored.",
    )
    p.add_argument(
        "--pattern-run",
        type=str,
        required=True,
        help="pattern_detector run ID (e.g. 20251201T190712Z_proton_electron_photon_scan).",
    )

    p.add_argument(
        "--cluster-root",
        type=str,
        default="outputs/particle_cluster_analysis",
        help="Root directory where particle_cluster_analysis runs are stored.",
    )
    p.add_argument(
        "--cluster-run",
        type=str,
        required=True,
        help="particle_cluster_analysis run ID containing summary.json and cluster_assignments.csv.",
    )

    p.add_argument(
        "--step",
        type=int,
        default=-1,
        help=(
            "Time step to analyze. If negative, use max step found in "
            "cluster_assignments.csv (typically the final snapshot)."
        ),
    )

    p.add_argument(
        "--fermion-clusters",
        type=str,
        default="",
        help=(
            "Comma-separated cluster IDs to treat as fermion-like. "
            "If empty, infer from cluster summary (electron-dominated)."
        ),
    )
    p.add_argument(
        "--boson-clusters",
        type=str,
        default="",
        help=(
            "Comma-separated cluster IDs to treat as boson-like. "
            "If empty, infer from cluster summary (photon-dominated)."
        ),
    )

    p.add_argument(
        "--prob-threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold p_i[m] >= prob_threshold to count node i "
            "as 'occupying' internal basis mode m."
        ),
    )

    p.add_argument(
        "--pi-window",
        type=float,
        default=float(np.pi / 3.0),
        help="Angular window (radians) to treat phases as 'near π' or 'near 0'.",
    )

    p.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory for exchange_statistics_probe outputs.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="exchange_statistics_probe",
        help="Tag for this analysis run.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ----------------- locate pattern run & params -----------------
    pattern_run_dir = os.path.join(args.pattern_root, args.pattern_run)
    pattern_params_path = os.path.join(pattern_run_dir, "params.json")
    if not os.path.isfile(pattern_params_path):
        raise FileNotFoundError(f"pattern params.json not found at: {pattern_params_path}")

    with open(pattern_params_path, "r", encoding="utf-8") as f:
        pattern_params = json.load(f)

    # ----------------- locate cluster run & artifacts --------------
    cluster_run_dir = os.path.join(args.cluster_root, args.cluster_run)
    cluster_summary_path = os.path.join(cluster_run_dir, "summary.json")
    cluster_csv_path = os.path.join(cluster_run_dir, "data", "cluster_assignments.csv")
    if not os.path.isfile(cluster_summary_path):
        raise FileNotFoundError(f"cluster summary.json not found at: {cluster_summary_path}")
    if not os.path.isfile(cluster_csv_path):
        raise FileNotFoundError(f"cluster_assignments.csv not found at: {cluster_csv_path}")

    # ----------------- infer fermion/boson clusters ----------------
    species_meta = _load_cluster_species_meta(cluster_summary_path)
    fermion_clusters_auto = species_meta.fermion_clusters
    boson_clusters_auto = species_meta.boson_clusters

    fermion_clusters = _parse_cluster_list(args.fermion_clusters)
    boson_clusters = _parse_cluster_list(args.boson_clusters)

    if not fermion_clusters:
        fermion_clusters = fermion_clusters_auto
    if not boson_clusters:
        boson_clusters = boson_clusters_auto

    if not fermion_clusters:
        raise RuntimeError("No fermion-like clusters found or specified.")
    if not boson_clusters:
        raise RuntimeError("No boson-like clusters found or specified.")

    # ----------------- load cluster assignments --------------------
    assign = _load_cluster_assignments(cluster_csv_path)
    unique_steps = np.unique(assign.steps)
    if unique_steps.size == 0:
        raise RuntimeError("cluster_assignments.csv has no rows.")

    if args.step < 0:
        step_to_use = int(unique_steps.max())
    else:
        # Find the closest step <= requested one that exists
        candidate_steps = unique_steps[unique_steps <= args.step]
        if candidate_steps.size == 0:
            raise RuntimeError(
                f"Requested step {args.step} not found (no recorded step <= that)."
            )
        step_to_use = int(candidate_steps.max())

    step_mask = assign.steps == step_to_use
    if not np.any(step_mask):
        raise RuntimeError(f"No cluster assignments found for step {step_to_use}.")

    node_ids = assign.nodes[step_mask]
    cluster_ids = assign.clusters[step_mask]

    # ----------------- build node -> cluster map -------------------
    n_nodes = int(pattern_params["n_nodes"])
    node_cluster = -np.ones(n_nodes, dtype=int)
    for nid, cid in zip(node_ids, cluster_ids):
        nid_i = int(nid)
        if 0 <= nid_i < n_nodes:
            node_cluster[nid_i] = int(cid)

    fermion_mask = np.isin(node_cluster, fermion_clusters)
    boson_mask = np.isin(node_cluster, boson_clusters)

    fermion_nodes = np.where(fermion_mask)[0]
    boson_nodes = np.where(boson_mask)[0]

    # ----------------- set up output dirs --------------------------
    run_id = _make_run_id(args.tag)
    run_root = os.path.join(args.output_root, "exchange_statistics_probe", run_id)
    data_dir = os.path.join(run_root, "data")
    _ensure_dir(run_root)
    _ensure_dir(data_dir)

    print("=" * 60)
    print("  Exchange Statistics Probe")
    print("=" * 60)
    print(f"pattern run:   {pattern_run_dir}")
    print(f"cluster run:   {cluster_run_dir}")
    print(f"step to use:   {step_to_use}")
    print(f"fermion clusters: {fermion_clusters}")
    print(f"boson clusters:   {boson_clusters}")
    print(f"# fermion nodes:  {len(fermion_nodes)}")
    print(f"# boson nodes:    {len(boson_nodes)}")
    print(f"prob_threshold:   {args.prob_threshold}")
    print(f"pi_window:        {args.pi_window}")
    print(f"output:           {run_root}")
    print("=" * 60)

    # ----------------- reconstruct and evolve substrate -----------
    cfg = Config(
        n_nodes=int(pattern_params["n_nodes"]),
        internal_dim=int(pattern_params["internal_dim"]),
        monogamy_budget=float(pattern_params["monogamy_budget"]),
        defrag_rate=float(pattern_params["defrag_rate"]),
        dt=float(pattern_params["dt"]),
        seed=int(pattern_params["seed"]),
        connectivity=float(pattern_params["connectivity"]),
        use_gpu=bool(pattern_params.get("use_gpu", False)),
    )
    sub = Substrate(cfg)

    dt = float(pattern_params["dt"])
    if step_to_use > 0:
        sub.evolve(n_steps=step_to_use)
    t = step_to_use * dt

    # ----------------- phase winding diagnostics ------------------
    tri_data = compute_triangle_loop_phases(sub)
    winding = summarize_phase_winding(tri_data, pi_window=args.pi_window)

    # ----------------- mode occupancy diagnostics -----------------
    occ_stats = compute_mode_occupancy_stats(
        sub,
        fermion_nodes=fermion_nodes,
        boson_nodes=boson_nodes,
        prob_threshold=args.prob_threshold,
    )

    # ----------------- write per-node winding CSV -----------------
    winding_csv_path = os.path.join(data_dir, "winding_per_node.csv")
    with open(winding_csv_path, "w", encoding="utf-8") as f:
        f.write(
            "node_id,cluster_id,is_fermion,is_boson,"
            "mean_phase,phase_spread,has_pi_winding,has_zero_winding\n"
        )
        for i in range(n_nodes):
            f.write(
                f"{i},"
                f"{node_cluster[i]},"
                f"{1 if fermion_mask[i] else 0},"
                f"{1 if boson_mask[i] else 0},"
                f"{winding.mean_phase[i] if np.isfinite(winding.mean_phase[i]) else np.nan},"
                f"{winding.phase_spread[i] if np.isfinite(winding.phase_spread[i]) else np.nan},"
                f"{1 if winding.has_pi_winding[i] else 0},"
                f"{1 if winding.has_zero_winding[i] else 0}\n"
            )

    # ----------------- write mode occupancy CSV -------------------
    occ_csv_path = os.path.join(data_dir, "mode_occupancy.csv")
    d = occ_stats.occ_all.shape[0]
    with open(occ_csv_path, "w", encoding="utf-8") as f:
        f.write(
            "mode_index,occ_fermion,occ_boson,occ_all\n"
        )
        for m in range(d):
            f.write(
                f"{m},"
                f"{occ_stats.occ_fermion[m]},"
                f"{occ_stats.occ_boson[m]},"
                f"{occ_stats.occ_all[m]}\n"
            )

    # ----------------- summary-level stats ------------------------
    def _frac(mask: np.ndarray, cond: np.ndarray) -> float:
        """Fraction of mask nodes that satisfy cond."""
        idx = np.where(mask)[0]
        if idx.size == 0:
            return float("nan")
        return float(np.sum(cond[idx])) / float(idx.size)

    ferm_pi_frac = _frac(fermion_mask, winding.has_pi_winding)
    ferm_zero_frac = _frac(fermion_mask, winding.has_zero_winding)
    boson_pi_frac = _frac(boson_mask, winding.has_pi_winding)
    boson_zero_frac = _frac(boson_mask, winding.has_zero_winding)

    # How many modes have >= 2 fermion/boson occupancies?
    modes_f_2plus = int(np.sum(occ_stats.occ_fermion >= 2))
    modes_b_2plus = int(np.sum(occ_stats.occ_boson >= 2))
    modes_f_1plus = int(np.sum(occ_stats.occ_fermion >= 1))
    modes_b_1plus = int(np.sum(occ_stats.occ_boson >= 1))

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "pattern_run": args.pattern_run,
        "cluster_run": args.cluster_run,
        "step": step_to_use,
        "time": t,
        "fermion_clusters": fermion_clusters,
        "boson_clusters": boson_clusters,
        "n_nodes": n_nodes,
        "n_fermion_nodes": int(len(fermion_nodes)),
        "n_boson_nodes": int(len(boson_nodes)),
        "prob_threshold": args.prob_threshold,
        "pi_window": args.pi_window,
        "phase_winding_summary": {
            "fermion_pi_fraction": ferm_pi_frac,
            "fermion_zero_fraction": ferm_zero_frac,
            "boson_pi_fraction": boson_pi_frac,
            "boson_zero_fraction": boson_zero_frac,
        },
        "mode_occupancy_summary": {
            "total_modes": int(d),
            "modes_with_fermion_occupancy_ge_1": modes_f_1plus,
            "modes_with_fermion_occupancy_ge_2": modes_f_2plus,
            "modes_with_boson_occupancy_ge_1": modes_b_1plus,
            "modes_with_boson_occupancy_ge_2": modes_b_2plus,
            "total_fermions": occ_stats.total_fermions,
            "total_bosons": occ_stats.total_bosons,
            "total_nodes": occ_stats.total_nodes,
        },
    }

    _write_json(os.path.join(run_root, "summary.json"), summary)
    _write_json(os.path.join(run_root, "params.json"), {
        "run_id": run_id,
        "pattern_run": args.pattern_run,
        "cluster_run": args.cluster_run,
        "pattern_root": args.pattern_root,
        "cluster_root": args.cluster_root,
        "step": args.step,
        "resolved_step": step_to_use,
        "fermion_clusters": fermion_clusters,
        "boson_clusters": boson_clusters,
        "prob_threshold": args.prob_threshold,
        "pi_window": args.pi_window,
    })

    print("=" * 60)
    print("Exchange statistics probe complete.")
    print(f"Winding CSV:      {winding_csv_path}")
    print(f"Mode occupancy:   {occ_csv_path}")
    print(f"Summary JSON:     {os.path.join(run_root, 'summary.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
