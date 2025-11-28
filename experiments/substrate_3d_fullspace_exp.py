"""
experiments/substrate_3d_fullspace_exp.py

Experiment wrapper for the full-Hilbert-space 3D substrate engine:

    engines/substrate_3d_fullspace_cu.py

This script:
  - calls run_experiment(...) on the full-space engine,
  - writes results to a JSON file in a run-specific folder, AND
  - writes a "latest" results JSON at:
        outputs/fullspace_results_latest.json

Workflow:
    python experiments\\substrate_3d_fullspace_exp.py --Lx 2 --Ly 2 --Lz 2 ...
    python experiments\\substrate_3d_sector_analysis.py

The analysis script defaults to using outputs/fullspace_results_latest.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict

# Repo root / imports
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from engines.substrate_3d_fullspace_cu import run_experiment as engine_run_experiment  # type: ignore

FRAMEWORK_VERSION = "0.1.0"
SCRIPT_NAME = os.path.basename(__file__)
EXPERIMENT_NAME = os.path.splitext(SCRIPT_NAME)[0]


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-Hilbert-space 3D substrate experiment (CuPy).",
    )

    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag.")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--Lx", type=int, default=2, help="Lattice size in x.")
    parser.add_argument("--Ly", type=int, default=2, help="Lattice size in y.")
    parser.add_argument("--Lz", type=int, default=2, help="Lattice size in z.")
    parser.add_argument("--J", type=float, default=1.0, help="XY coupling strength.")
    parser.add_argument("--hz", type=float, default=0.0, help="Longitudinal field hz.")
    parser.add_argument("--T", type=float, default=10.0, help="Total evolution time.")
    parser.add_argument("--dt", type=float, default=0.1, help="Sampling time step.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Output dirs / logging
# ---------------------------------------------------------------------------

def generate_run_id(tag: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{tag}" if tag else ts


def ensure_unique_run_dir(base_dir: str) -> str:
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
    experiment_root = os.path.join(output_root, EXPERIMENT_NAME)
    os.makedirs(experiment_root, exist_ok=True)

    base_run_dir = os.path.join(experiment_root, run_id)
    run_dir = ensure_unique_run_dir(base_run_dir)

    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    return {
        "run_dir": run_dir,
        "logs_dir": logs_dir,
        "output_root": output_root,
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
# Metadata builders
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
        "hz": args.hz,
        "T": args.T,
        "dt": args.dt,
        "seed": args.seed,
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
        "notes": "Full-Hilbert-space 3D substrate experiment (CuPy).",
    }


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

    # Save params & metadata
    write_json(os.path.join(dirs["run_dir"], "params.json"), params)
    write_json(os.path.join(dirs["run_dir"], "metadata.json"), metadata)

    engine_params: Dict[str, Any] = {
        "Lx": args.Lx,
        "Ly": args.Ly,
        "Lz": args.Lz,
        "J": args.J,
        "hz": args.hz,
        "T": args.T,
        "dt": args.dt,
        "seed": args.seed,
    }

    logging.info("Engine parameters:")
    for k, v in engine_params.items():
        logging.info(f"  {k}: {v}")

    try:
        results = engine_run_experiment(engine_params)

        # Run-specific results JSON
        run_json = os.path.join(dirs["run_dir"], "fullspace_results.json")
        write_json(run_json, results)

        # Stable "latest" copy for downstream analysis
        latest_json = os.path.join(dirs["output_root"], "fullspace_results_latest.json")
        write_json(latest_json, results)

        logging.info("Run completed successfully.")
        logging.info(f"  Output root        : {args.output_root}")
        logging.info(f"  Run dir            : {dirs['run_dir']}")
        logging.info(f"  Run-specific JSON  : {run_json}")
        logging.info(f"  Latest-results JSON: {latest_json}")

    except Exception as exc:
        logging.exception("Run failed with exception:")
        error_summary = {
            "framework_version": FRAMEWORK_VERSION,
            "script": SCRIPT_NAME,
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run_id,
            "error": str(exc),
        }
        run_json = os.path.join(dirs["run_dir"], "fullspace_results.json")
        latest_json = os.path.join(dirs["output_root"], "fullspace_results_latest.json")
        write_json(run_json, error_summary)
        write_json(latest_json, error_summary)
        sys.exit(1)


if __name__ == "__main__":
    main()
