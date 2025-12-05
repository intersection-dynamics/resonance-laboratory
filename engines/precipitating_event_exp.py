"""
precipitating_event_exp.py

Experiment script for the Hilbert-substrate Precipitating Event on an
emergent geometry.

Responsibilities:
- Parse CLI arguments (output-root, tag, seed, physics params, geom file).
- Load the emergent geometry asset (e.g. lr_embedding_3d.npz).
- Call the geom_precipitation_engine.run_experiment(...) engine.
- Save params.json, metadata.json, summary.json, and raw arrays under outputs/.
- Log a human-readable summary to console and logs/run.log.

This script follows the Substrate Project Software Guide conventions.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from typing import Any, Dict

import numpy as np

# Import the engine (adjust path if your repo layout differs)
from engines.geom_precipitation_engine import run_experiment, PrecipitationParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precipitating Event on emergent geometry (Hilbert-substrate)."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory for all experiment outputs.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag appended to the run_id.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for the initial state.",
    )
    parser.add_argument(
        "--geom-file",
        type=str,
        default="configs/lr_embedding_3d.npz",
        help="Path to emergent geometry asset (npz with graph_dist, etc.).",
    )
    # Physics / schedule parameters
    parser.add_argument(
        "--t-total",
        type=float,
        default=10.0,
        help="Total evolution time.",
    )
    parser.add_argument(
        "--t-quench",
        type=float,
        default=4.0,
        help="Quench time (0 < t_quench < t_total).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=101,
        help="Number of time steps / snapshots.",
    )
    parser.add_argument(
        "--J-coupling",
        type=float,
        default=1.0,
        help="Heisenberg coupling strength on edges.",
    )
    parser.add_argument(
        "--h-field",
        type=float,
        default=0.2,
        help="On-site Z field strength.",
    )
    parser.add_argument(
        "--defrag-hot",
        type=float,
        default=0.3,
        help="Defrag strength before quench.",
    )
    parser.add_argument(
        "--defrag-cold",
        type=float,
        default=1.0,
        help="Defrag strength after quench.",
    )
    return parser.parse_args()


def make_run_id(tag: str) -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if tag:
        return f"{ts}_{tag}"
    return ts


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=False)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()

    # Resolve script name and experiment name
    script_name = os.path.basename(__file__)
    experiment_name = os.path.splitext(script_name)[0]

    # Construct run_id and output directories
    run_id = make_run_id(args.tag)
    run_root = os.path.join(args.output_root, experiment_name, run_id)

    data_dir = os.path.join(run_root, "data")
    figures_dir = os.path.join(run_root, "figures")
    logs_dir = os.path.join(run_root, "logs")

    try:
        ensure_dir(run_root)
        ensure_dir(data_dir)
        ensure_dir(figures_dir)
        ensure_dir(logs_dir)
    except FileExistsError:
        print(
            f"[ERROR] Run directory already exists: {run_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Set up logging to file
    log_path = os.path.join(logs_dir, "run.log")
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        print(msg, file=log_file, flush=True)

    # Header
    log("=" * 60)
    log("  Hilbert-Substrate Precipitating Event")
    log("=" * 60)
    log(f"Script:         {script_name}")
    log(f"Experiment:     {experiment_name}")
    log(f"Run ID:         {run_id}")
    log(f"Output root:    {run_root}")
    log(f"Seed:           {args.seed}")
    log(f"Geometry file:  {args.geom_file}")
    log("-" * 60)

    # Load geometry asset
    if not os.path.exists(args.geom_file):
        log(f"[ERROR] Geometry file not found: {args.geom_file}")
        log_file.close()
        sys.exit(1)

    geom_npz = np.load(args.geom_file)
    # We expect at least graph_dist; others are optional but nice to have
    if "graph_dist" not in geom_npz:
        log("[ERROR] geom-file is missing 'graph_dist' array.")
        log_file.close()
        sys.exit(1)

    geometry: Dict[str, np.ndarray] = {
        key: geom_npz[key] for key in geom_npz.files
    }
    n_sites = geometry["graph_dist"].shape[0]
    log(f"Loaded geometry with n_sites = {n_sites}")
    log("-" * 60)

    # Build params for the engine
    params = PrecipitationParams(
        n_sites=n_sites,
        local_dim=2,
        J_coupling=args.J_coupling,
        h_field=args.h_field,
        defrag_hot=args.defrag_hot,
        defrag_cold=args.defrag_cold,
        t_total=args.t_total,
        t_quench=args.t_quench,
        n_steps=args.n_steps,
    )

    # Serialize params and metadata
    params_json = {
        "framework_version": "0.1.0",
        "script": script_name,
        "experiment": experiment_name,
        "run_id": run_id,
        "timestamp": _dt.datetime.now().isoformat(),
        "params": vars(params),
    }
    write_json(os.path.join(run_root, "params.json"), params_json)

    metadata = {
        "seed": args.seed,
        "geom_file": os.path.abspath(args.geom_file),
        "args": vars(args),
        "run_root": os.path.abspath(run_root),
    }
    write_json(os.path.join(run_root, "metadata.json"), metadata)

    # Echo parameter summary to log
    log("Parameter summary:")
    for k, v in vars(params).items():
        log(f"  {k:16s} = {v}")
    log("-" * 60)
    log("Running engine...")
    log("-" * 60)

    # Run the engine
    results = run_experiment(
        geometry=geometry,
        seed=int(args.seed),
        params_dict=vars(params),
    )

    log("Engine completed.")
    log("-" * 60)

    # Save summary.json
    summary = {
        "framework_version": "0.1.0",
        "script": script_name,
        "experiment": experiment_name,
        "run_id": run_id,
        "timestamp": _dt.datetime.now().isoformat(),
        "params": results["params"],
        "metrics": results["metrics"],
        "exchange": results["exchange"],
    }
    write_json(os.path.join(run_root, "summary.json"), summary)

    # Save raw arrays to data/
    np.savez_compressed(
        os.path.join(data_dir, "time_series.npz"),
        times=results["times"],
        local_z_t=results["local_z_t"],
        lump_counts=results["lump_counts"],
    )

    # Lump history is ragged; save as JSON
    write_json(os.path.join(data_dir, "lump_hist.json"), results["lump_hist"])

    # Console verdicts
    m = results["metrics"]
    log("==== Precipitating Event Summary ====")
    log(f"t_quench_effective: {m['t_quench_effective']:.6f}")
    log(f"final_n_lumps:      {m['final_n_lumps']}")
    log(f"final_lump_sizes:   {m['final_lump_sizes']}")
    log(f"mean_lump_count:    {m['mean_lump_count']:.3f}")
    log(f"has_candidates:     {m['has_particle_candidates']}")
    log("--------------------------------------")
    log("Exchange-signature placeholder:")
    log(f"  implemented: {results['exchange']['implemented']}")
    log("======================================")

    log_file.close()


if __name__ == "__main__":
    main()
