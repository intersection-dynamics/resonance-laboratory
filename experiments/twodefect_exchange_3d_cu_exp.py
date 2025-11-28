"""
experiments/twodefect_exchange_3d_cu_exp.py

3D CuPy two-defect exchange-stability experiment using:

    engines/twodefect_exchange_3d_cu_engine.py

Follows the Substrate Project Software Guide conventions【turn1file5†L23-L33】【turn1file8†L110-L120】:

- Outputs are placed in:
      outputs/<experiment_name>/<run_id>/
- Writes:
      params.json
      metadata.json
      summary.json
      logs/run.log
      data/*.npz
      figures/*.png

Example (Windows):
    python experiments\\twodefect_exchange_3d_cu_exp.py ^
        --Lx 4 --Ly 4 --Lz 4 ^
        --T 16.0 --dt 0.1 ^
        --phases 0 1.57079632679 3.14159265359 ^
        --seed 42 --tag 3d_cu_demo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt

# Import engine from repo root
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from engines.twodefect_exchange_3d_cu_engine import (  # type: ignore
    run_experiment as engine_run_experiment,
)


FRAMEWORK_VERSION = "0.1.0"
SCRIPT_NAME = os.path.basename(__file__)
EXPERIMENT_NAME = os.path.splitext(SCRIPT_NAME)[0]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D CuPy two-defect exchange stability experiment.",
    )

    # Standard project args
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag.")
    parser.add_argument("--seed", type=int, default=1234)

    # Lattice / physics
    parser.add_argument("--Lx", type=int, default=4, help="Lattice size in x.")
    parser.add_argument("--Ly", type=int, default=4, help="Lattice size in y.")
    parser.add_argument("--Lz", type=int, default=4, help="Lattice size in z.")
    parser.add_argument("--J", type=float, default=1.0, help="Hopping amplitude.")
    parser.add_argument("--T", type=float, default=16.0, help="Total evolution time.")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step.")
    parser.add_argument(
        "--phases",
        type=float,
        nargs="+",
        default=[0.0, 0.5 * np.pi, np.pi],
        help="List of exchange phases to probe.",
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        default="random",
        help='Initial state mode in H_phys (default: "random").',
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Output dirs / logging
# ---------------------------------------------------------------------------

def generate_run_id(tag: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{tag}" if tag else ts


def ensure_unique_run_dir(base_dir: str) -> str:
    """
    Ensure we do not overwrite an existing run folder.
    If base_dir exists, append _1, _2, ... until free.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=False)
        return base_dir

    suffix = 1
    while True:
        candidate = f"{base_dir}_{suffix}"
        if not os.path.exists(candidate):
            os.makedirs(candidate, exist_ok=False)
            return candidate
        suffix += 1


def setup_output_dirs(output_root: str, run_id: str) -> Dict[str, str]:
    """
    Create:

    outputs/<experiment_name>/<run_id>/
        params.json
        metadata.json
        summary.json
        logs/run.log
        data/
        figures/
    """
    experiment_root = os.path.join(output_root, EXPERIMENT_NAME)
    os.makedirs(experiment_root, exist_ok=True)

    base_run_dir = os.path.join(experiment_root, run_id)
    run_dir = ensure_unique_run_dir(base_run_dir)

    data_dir = os.path.join(run_dir, "data")
    figures_dir = os.path.join(run_dir, "figures")
    logs_dir = os.path.join(run_dir, "logs")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    return {
        "run_dir": run_dir,
        "data_dir": data_dir,
        "figures_dir": figures_dir,
        "logs_dir": logs_dir,
    }


def setup_logging(logs_dir: str) -> None:
    log_path = os.path.join(logs_dir, "run.log")

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fh_fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh_fmt = logging.Formatter("%(message)s")
    sh.setFormatter(sh_fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Metadata / summary
# ---------------------------------------------------------------------------

def build_params_dict(args: argparse.Namespace, run_id: str) -> Dict[str, Any]:
    return {
        "framework_version": FRAMEWORK_VERSION,
        "script": SCRIPT_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "run_id": run_id,
        "Lx": args.Lx,
        "Ly": args.Ly,
        "Lz": args.Lz,
        "J": args.J,
        "T": args.T,
        "dt": args.dt,
        "phases": args.phases,
        "seed": args.seed,
        "init_mode": args.init_mode,
    }


def build_metadata_dict(args: argparse.Namespace, run_id: str) -> Dict[str, Any]:
    return {
        "script": SCRIPT_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "cli_args": vars(args),
        "git_commit": None,
        "notes": "3D CuPy two-defect exchange stability experiment.",
    }


def summarize_results(engine_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build summary.json capturing:
      - metrics: S(phi), best phases
      - diagnostics: dim, eigenvalue range, runtime, etc.
      - verdicts: whether 0 and pi are among the most stable phases.
    """
    stabilities = engine_results["stabilities"]
    diagnostics = engine_results["diagnostics"]

    phi_vals: List[float] = []
    S_vals: List[float] = []
    for k, v in stabilities.items():
        phi_vals.append(float(k))
        S_vals.append(float(v))

    phi_array = np.array(phi_vals, dtype=float)
    S_array = np.array(S_vals, dtype=float)

    S_max = float(S_array.max())
    best_indices = np.where(np.isclose(S_array, S_max))[0].tolist()
    best_phases = [float(phi_array[i]) for i in best_indices]

    metrics: Dict[str, Any] = {
        "S_by_phase": {str(phi_array[i]): float(S_array[i]) for i in range(len(phi_array))},
        "S_max": S_max,
        "best_phases": best_phases,
    }

    has_zero = any(np.isclose(phi, 0.0) for phi in best_phases)
    has_pi = any(np.isclose(phi, np.pi) for phi in best_phases)
    boson_fermion_only = has_zero and has_pi and len(best_phases) <= 2

    verdicts: Dict[str, Any] = {
        "has_phase_zero_stable": has_zero,
        "has_phase_pi_stable": has_pi,
        "boson_fermion_only": boson_fermion_only,
    }

    summary: Dict[str, Any] = {
        "framework_version": FRAMEWORK_VERSION,
        "script": SCRIPT_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "verdicts": verdicts,
    }

    return summary


# ---------------------------------------------------------------------------
# Data / figures
# ---------------------------------------------------------------------------

def save_data_arrays(data_dir: str, engine_results: Dict[str, Any]) -> None:
    """
    Save raw arrays and stabilities to data/exchange_3d_data.npz.
    """
    times_list = engine_results["times"]
    fid_list = engine_results["fidelities"]
    stabilities = engine_results["stabilities"]

    times_array = np.array(times_list, dtype=object)
    fidelities_array = np.array(fid_list, dtype=object)
    stab_items = np.array(list(stabilities.items()), dtype=object)

    out_path = os.path.join(data_dir, "exchange_3d_data.npz")
    np.savez(
        out_path,
        times=times_array,
        fidelities=fidelities_array,
        stabilities=stab_items,
    )


def save_stability_figure(figures_dir: str, engine_results: Dict[str, Any]) -> None:
    """
    Plot S(phi) vs phi and save as stability_vs_phase_3d.png.
    """
    stabilities = engine_results["stabilities"]
    phi_vals: List[float] = []
    S_vals: List[float] = []

    for k, v in stabilities.items():
        phi_vals.append(float(k))
        S_vals.append(float(v))

    phi_vals = np.array(phi_vals, dtype=float)
    S_vals = np.array(S_vals, dtype=float)

    order = np.argsort(phi_vals)
    phi_vals = phi_vals[order]
    S_vals = S_vals[order]

    plt.figure()
    plt.plot(phi_vals / np.pi, S_vals, marker="o")
    plt.xlabel("Exchange phase ϕ / π")
    plt.ylabel("Stability S(ϕ)")
    plt.title("3D CuPy Two-Defect Exchange Stability")
    plt.grid(True)

    fig_path = os.path.join(figures_dir, "stability_vs_phase_3d.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    run_id = generate_run_id(args.tag)
    dirs = setup_output_dirs(args.output_root, run_id)
    setup_logging(dirs["logs_dir"])

    logging.info("=" * 72)
    logging.info(f" Experiment: {EXPERIMENT_NAME}")
    logging.info(f" Script    : {SCRIPT_NAME}")
    logging.info(f" Run ID    : {run_id}")
    logging.info("=" * 72)

    params = build_params_dict(args, run_id)
    metadata = build_metadata_dict(args, run_id)

    # Save params & metadata upfront
    write_json(os.path.join(dirs["run_dir"], "params.json"), params)
    write_json(os.path.join(dirs["run_dir"], "metadata.json"), metadata)

    engine_params: Dict[str, Any] = {
        "Lx": args.Lx,
        "Ly": args.Ly,
        "Lz": args.Lz,
        "J": args.J,
        "T": args.T,
        "dt": args.dt,
        "phases": args.phases,
        "seed": args.seed,
        "init_mode": args.init_mode,
    }

    logging.info("Engine parameters:")
    for k, v in engine_params.items():
        logging.info(f"  {k}: {v}")

    try:
        engine_results = engine_run_experiment(engine_params)
        summary = summarize_results(engine_results)
        summary["run_id"] = run_id
        summary["params"] = params

        write_json(os.path.join(dirs["run_dir"], "summary.json"), summary)
        save_data_arrays(dirs["data_dir"], engine_results)
        save_stability_figure(dirs["figures_dir"], engine_results)

        logging.info("Run completed successfully.")
        logging.info(f"  Output root : {args.output_root}")
        logging.info(f"  Run dir     : {dirs['run_dir']}")

    except Exception as exc:
        logging.exception("Run failed with exception:")
        error_summary = {
            "framework_version": FRAMEWORK_VERSION,
            "script": SCRIPT_NAME,
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run_id,
            "error": str(exc),
        }
        write_json(os.path.join(dirs["run_dir"], "summary.json"), error_summary)
        sys.exit(1)


if __name__ == "__main__":
    main()
