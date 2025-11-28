#!/usr/bin/env python3
"""
generate_stability_vs_phase_data.py

Generates a long-time stability profile S(phi) vs exchange phase phi,
and writes it as a .dat file for PGFPlots.

Output:
    ../figures/fig_stability_vs_phi.dat

Columns:
    phi   S
"""

import os
import numpy as np

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
N_PHI = 200   # resolution over phi in [0, pi]
PHI_MIN = 0.0
PHI_MAX = np.pi


# ---------------------------------------------------------------------
# SYNTHETIC STABILITY PROFILE (REPLACE WITH REAL DATA LATER)
# ---------------------------------------------------------------------

def synthetic_stability_profile(phi: np.ndarray) -> np.ndarray:
    """
    Construct a synthetic S(phi) with sharp peaks at phi=0 and phi=pi
    and suppression in between. Here we use a sum of two narrow Gaussians.

    This is purely illustrative; plug in real S(phi) from simulations if available.
    """
    sigma = 0.2  # width of peaks

    # Two Gaussians centered at 0 and pi
    peak0 = np.exp(-0.5 * (phi / sigma)**2)
    peakpi = np.exp(-0.5 * ((phi - np.pi) / sigma)**2)

    # Normalize so maxima are 1
    S = peak0 + peakpi
    S /= S.max()

    return S


# ---------------------------------------------------------------------
# IO HELPERS
# ---------------------------------------------------------------------

def ensure_output_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def write_dat(filename: str, phi: np.ndarray, S: np.ndarray) -> None:
    """
    Write two-column ASCII file:
        phi   S
    """
    data = np.column_stack([phi, S])
    header = "phi\tS\n"
    with open(filename, "w") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.8f\t%.8f")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    ensure_output_dir(OUTPUT_DIR)

    phi = np.linspace(PHI_MIN, PHI_MAX, N_PHI)
    S = synthetic_stability_profile(phi)

    out_path = os.path.join(OUTPUT_DIR, "fig_stability_vs_phi.dat")
    write_dat(out_path, phi, S)

    print(f"Wrote {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
