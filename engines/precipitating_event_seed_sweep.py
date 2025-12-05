#!/usr/bin/env python3
"""
precipitating_event_seed_sweep.py
=================================

Seed sweep driver for precipitating_event.py.

For a fixed emergent geometry (asset_dir) and fixed physics parameters,
this script runs the precipitating-event quench+cooling simulation for a
range of random seeds and records the resulting chiral excess.

For each seed s in [seed_start, seed_end]:

  - Build a PrecipConfig with that seed.
  - Use output_dir = base_output_dir + "_seed{seed}" so runs do not collide.
  - Call run_simulation(cfg, return_data=True).
  - Extract:
      * Q_final        = Q_net[-1]
      * N_plus_final   = N_plus[-1]
      * N_minus_final  = N_minus[-1]
      * Q_quench_end   = Q_net[idx_quench]
        where idx_quench is the time index closest to t_quench.

Results are saved to:

  asset_dir / sweep_output_dir / seed_sweep_results.npz
  asset_dir / sweep_output_dir / summary.txt

The npz file contains:
  seeds, Q_final, N_plus_final, N_minus_final,
  Q_quench_end, t_quench, t_final

The summary.txt file prints a small table and basic stats over seeds.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, List

import numpy as np

from precipitating_event import PrecipConfig, run_simulation


def run_seed_sweep(args: argparse.Namespace) -> None:
    # Base config (seed will be overridden in loop)
    base_cfg = PrecipConfig(
        asset_dir=args.asset_dir,
        metrics_file=args.metrics_file,
        embedding_smooth_file=args.embedding_smooth_file,
        embedding_file=args.embedding_file,
        hot_strength=args.hot_strength,
        J_order=args.J_order,
        rot_strength=args.rot_strength,
        Omega_x=args.Omega_x,
        Omega_y=args.Omega_y,
        Omega_z=args.Omega_z,
        t_max=args.t_max,
        n_steps=args.n_steps,
        quench_fraction=args.quench_fraction,
        seed=args.seed_start,  # will be replaced
        init_random_phase=not args.no_random_phase,
        max_cycle_len=args.max_cycle_len,
        output_dir=args.base_output_dir,  # will get suffix
    )

    seed_start = args.seed_start
    seed_end = args.seed_end
    seeds: List[int] = list(range(seed_start, seed_end + 1))

    print("Seed sweep config:")
    print("  asset_dir       :", base_cfg.asset_dir)
    print("  metrics_file    :", base_cfg.metrics_file)
    print("  embedding_smooth:", base_cfg.embedding_smooth_file)
    print("  embedding_file  :", base_cfg.embedding_file)
    print("  hot_strength    :", base_cfg.hot_strength)
    print("  J_order         :", base_cfg.J_order)
    print("  rot_strength    :", base_cfg.rot_strength)
    print("  Omega           :", (base_cfg.Omega_x, base_cfg.Omega_y, base_cfg.Omega_z))
    print("  t_max, n_steps  :", base_cfg.t_max, base_cfg.n_steps)
    print("  quench_fraction :", base_cfg.quench_fraction)
    print("  max_cycle_len   :", base_cfg.max_cycle_len)
    print("  base_output_dir :", base_cfg.output_dir)
    print("  seeds           :", seeds)
    print()

    n_seeds = len(seeds)
    Q_final = np.zeros(n_seeds, dtype=int)
    N_plus_final = np.zeros(n_seeds, dtype=int)
    N_minus_final = np.zeros(n_seeds, dtype=int)
    Q_quench_end = np.zeros(n_seeds, dtype=int)

    t_quench = base_cfg.quench_fraction * base_cfg.t_max

    for idx, seed in enumerate(seeds):
        print("=" * 72)
        print(f"Running seed {seed} ({idx+1}/{n_seeds})")
        print("=" * 72)

        cfg = base_cfg
        cfg.seed = seed
        cfg.output_dir = f"{args.base_output_dir}_seed{seed}"

        result = run_simulation(cfg, return_data=True)
        assert result is not None

        times = result["times"]
        Q_net = result["Q_net"]
        N_plus = result["N_plus"]
        N_minus = result["N_minus"]

        # Final values
        Q_final[idx] = int(Q_net[-1])
        N_plus_final[idx] = int(N_plus[-1])
        N_minus_final[idx] = int(N_minus[-1])

        # Value at end of quench (closest time to t_quench)
        idx_quench = int(np.argmin(np.abs(times - t_quench)))
        Q_quench_end[idx] = int(Q_net[idx_quench])

        print(
            f"  seed {seed}: Q_quench_end={Q_quench_end[idx]}, "
            f"Q_final={Q_final[idx]}, "
            f"N_plus_final={N_plus_final[idx]}, "
            f"N_minus_final={N_minus_final[idx]}"
        )
        print()

    # Prepare sweep output dir
    sweep_dir = os.path.join(base_cfg.asset_dir, args.sweep_output_dir)
    os.makedirs(sweep_dir, exist_ok=True)

    # Save npz
    npz_path = os.path.join(sweep_dir, "seed_sweep_results.npz")
    np.savez(
        npz_path,
        seeds=np.array(seeds, dtype=int),
        Q_final=Q_final,
        N_plus_final=N_plus_final,
        N_minus_final=N_minus_final,
        Q_quench_end=Q_quench_end,
        t_quench=float(t_quench),
        t_final=float(base_cfg.t_max),
    )

    # Save human-readable summary
    summary_path = os.path.join(sweep_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Precipitating Event Seed Sweep\n")
        f.write("================================\n\n")
        f.write(f"asset_dir        : {base_cfg.asset_dir}\n")
        f.write(f"metrics_file     : {base_cfg.metrics_file}\n")
        f.write(f"embedding_smooth : {base_cfg.embedding_smooth_file}\n")
        f.write(f"embedding_file   : {base_cfg.embedding_file}\n")
        f.write(f"hot_strength     : {base_cfg.hot_strength}\n")
        f.write(f"J_order          : {base_cfg.J_order}\n")
        f.write(f"rot_strength     : {base_cfg.rot_strength}\n")
        f.write(f"Omega            : ({base_cfg.Omega_x}, {base_cfg.Omega_y}, {base_cfg.Omega_z})\n")
        f.write(f"t_max, n_steps   : {base_cfg.t_max}, {base_cfg.n_steps}\n")
        f.write(f"quench_fraction  : {base_cfg.quench_fraction}\n")
        f.write(f"max_cycle_len    : {base_cfg.max_cycle_len}\n")
        f.write(f"base_output_dir  : {base_cfg.output_dir}\n\n")
        f.write(f"seed range       : {seed_start} .. {seed_end}\n\n")

        f.write("Per-seed results:\n")
        f.write("  seed | Q_quench_end | Q_final | N_plus_final | N_minus_final\n")
        f.write("  ------------------------------------------------------------\n")
        for i, seed in enumerate(seeds):
            f.write(
                f"  {seed:4d} | {Q_quench_end[i]:12d} | {Q_final[i]:7d} | "
                f"{N_plus_final[i]:13d} | {N_minus_final[i]:14d}\n"
            )

        f.write("\nStatistics over seeds:\n")
        f.write(f"  mean Q_final          = {float(np.mean(Q_final)):.3f}\n")
        f.write(f"  std  Q_final          = {float(np.std(Q_final)):.3f}\n")
        f.write(f"  fraction Q_final > 0  = {float(np.mean(Q_final > 0)):.3f}\n")
        f.write(f"  fraction Q_final < 0  = {float(np.mean(Q_final < 0)):.3f}\n")
        f.write(f"  fraction Q_final == 0 = {float(np.mean(Q_final == 0)):.3f}\n")

        # Also stats for Q_quench_end
        f.write("\n  mean Q_quench_end     = {0:.3f}\n".format(float(np.mean(Q_quench_end))))
        f.write("  std  Q_quench_end     = {0:.3f}\n".format(float(np.std(Q_quench_end))))
        f.write(
            "  fraction Q_quench_end > 0  = {0:.3f}\n".format(
                float(np.mean(Q_quench_end > 0))
            )
        )
        f.write(
            "  fraction Q_quench_end < 0  = {0:.3f}\n".format(
                float(np.mean(Q_quench_end < 0))
            )
        )
        f.write(
            "  fraction Q_quench_end == 0 = {0:.3f}\n".format(
                float(np.mean(Q_quench_end == 0))
            )
        )

    print("Seed sweep complete.")
    print("  Saved seed_sweep_results.npz to", npz_path)
    print("  Saved summary.txt to", summary_path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed sweep driver for precipitating_event (quench+cooling)."
    )
    parser.add_argument(
        "--asset-dir",
        type=str,
        default="cubic_universe_L3",
        help="Directory with lr_metrics.npz and embeddings.",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="lr_metrics.npz",
        help="Metrics file name inside asset-dir.",
    )
    parser.add_argument(
        "--embedding-smooth-file",
        type=str,
        default="lr_embedding_3d_smooth.npz",
        help="Preferred smoothed embedding file name.",
    )
    parser.add_argument(
        "--embedding-file",
        type=str,
        default="lr_embedding_3d.npz",
        help="Fallback embedding file name.",
    )
    parser.add_argument(
        "--hot-strength",
        type=float,
        default=1.0,
        help="Strength for disordered H_hot.",
    )
    parser.add_argument(
        "--J-order",
        type=float,
        default=1.0,
        help="Coupling for ordered H_order.",
    )
    parser.add_argument(
        "--rot-strength",
        type=float,
        default=0.3,
        help="Overall scale for rotation bias H_rot.",
    )
    parser.add_argument(
        "--Omega-x",
        type=float,
        default=0.0,
        help="Rotation vector Omega_x.",
    )
    parser.add_argument(
        "--Omega-y",
        type=float,
        default=0.0,
        help="Rotation vector Omega_y.",
    )
    parser.add_argument(
        "--Omega-z",
        type=float,
        default=1.0,
        help="Rotation vector Omega_z. Use 1.0 for the stronger spin run.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=10.0,
        help="Total evolution time.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=4000,
        help="Number of RK4 time steps.",
    )
    parser.add_argument(
        "--quench-fraction",
        type=float,
        default=0.4,
        help="Fraction of t_max for lambda(t) to ramp from 0 to 1.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=100,
        help="First seed (inclusive).",
    )
    parser.add_argument(
        "--seed-end",
        type=int,
        default=119,
        help="Last seed (inclusive).",
    )
    parser.add_argument(
        "--no-random-phase",
        action="store_true",
        help="If set, start from localized |0> instead of random phases.",
    )
    parser.add_argument(
        "--max-cycle-len",
        type=int,
        default=4,
        help="Maximum cycle length to consider for vorticity (min 3).",
    )
    parser.add_argument(
        "--base-output-dir",
        type=str,
        default="precipitating_event_L3_Omega1",
        help="Base output-dir prefix for individual seed runs (suffix '_seedX' added).",
    )
    parser.add_argument(
        "--sweep-output-dir",
        type=str,
        default="precipitating_event_seed_sweep_L3_Omega1",
        help="Directory under asset_dir to store sweep summary/npz.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    run_seed_sweep(args)


if __name__ == "__main__":
    main()
