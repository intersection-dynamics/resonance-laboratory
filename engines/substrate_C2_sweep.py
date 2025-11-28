#!/usr/bin/env python3
"""
experiments/substrate_C2_sweep.py

Automate a parameter sweep for substrate_C2_analysis.py.

It runs substrate_C2_analysis.py for a grid of:
  - n_nodes ∈ {16, 36}
  - connectivity ∈ {0.2, 0.3, 0.4}

All runs use:
  - internal_dim = 2
  - n_evolution_steps = 50
  - dt = 0.1
  - excitation_threshold = 0.05
  - max_defects = 12
  - seed = 123

Each run gets a tag like: n16_c02, n36_c04, etc.
"""

import subprocess
import sys


def run_sweep():
    # Fixed parameters
    internal_dim = 2
    n_evolution_steps = 50
    dt = 0.1
    excitation_threshold = 0.05
    max_defects = 12
    seed = 123

    # Sweep grid
    n_nodes_list = [16, 36]
    connectivity_list = [0.2, 0.3, 0.4]

    for n_nodes in n_nodes_list:
        for connectivity in connectivity_list:
            # Make a short tag like n16_c02, n36_c04, etc.
            c_tag = int(round(connectivity * 100))
            tag = f"n{n_nodes}_c{c_tag:02d}"

            cmd = [
                sys.executable,
                "substrate_C2_analysis.py",
                "--n-nodes", str(n_nodes),
                "--internal-dim", str(internal_dim),
                "--connectivity", str(connectivity),
                "--n-evolution-steps", str(n_evolution_steps),
                "--dt", str(dt),
                "--excitation-threshold", str(excitation_threshold),
                "--max-defects", str(max_defects),
                "--seed", str(seed),
                "--tag", tag,
            ]

            print("\n============================================================")
            print("Running:", " ".join(cmd))
            print("============================================================")

            # Run the command and make sure errors propagate
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"\nERROR: Command failed with return code {result.returncode}")
                sys.exit(result.returncode)

    print("\nAll sweep runs completed successfully.")


if __name__ == "__main__":
    run_sweep()
