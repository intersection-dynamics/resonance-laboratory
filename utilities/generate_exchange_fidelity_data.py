#!/usr/bin/env python3
"""
generate_exchange_fidelity_data.py

Generate exchange-fidelity data from a simple, honest 2-level hopping model.

We model a single-excitation subspace spanned by:
    |10> = (1, 0)^T
    |01> = (0, 1)^T

Initial state:
    |ψ_φ(0)> = (|10> + e^{i φ} |01>)/sqrt(2)

Hamiltonian (tight-binding / hopping):
    H = -t ( |10><01| + |01><10| ) = -t * [[0, 1],
                                           [1, 0]]

Time evolution:
    |ψ_φ(t)> = exp(-i H t) |ψ_φ(0)>

Fidelity:
    F_φ(t) = |<ψ_φ(0) | ψ_φ(t)>|

This produces real quantum dynamics (no synthetic curves), but does not depend
on substrate.py internals. The resulting .dat files can be used directly by
PGFPlots in the LaTeX paper.

Outputs (relative to repo root, assuming this is in utilities/):
    figures/fig_fidelity_phi0.dat
    figures/fig_fidelity_phipi.dat
    figures/fig_fidelity_phihalf.dat

Each file has two columns:
    t   F
"""

import os
import numpy as np
from scipy.linalg import expm

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Where to write .dat files (relative to this script)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# Time grid
T_MAX = 10.0
N_STEPS = 200

# Hopping strength
HOPPING_COUPLING = 1.0


# ---------------------------------------------------------------------
# CORE SIMULATION
# ---------------------------------------------------------------------

def run_exchange_simulation(phi: float):
    """
    Run a simple 2-level hopping simulation for a given exchange phase φ.

    Basis:
        |10> -> e1 = [1, 0]^T
        |01> -> e2 = [0, 1]^T

    Initial state:
        |ψ_φ(0)> = (|10> + e^{i φ} |01>)/sqrt(2)

    Hamiltonian:
        H = -t (|10><01| + |01><10|) = -t * [[0, 1],
                                             [1, 0]]

    Time evolution:
        |ψ_φ(t)> = exp(-i H t) |ψ_φ(0)>

    Return:
        times: np.ndarray shape (N_STEPS,)
        fidelities: np.ndarray shape (N_STEPS,)
    """
    # 2D Hilbert space
    dim = 2

    # Hamiltonian H
    H = -HOPPING_COUPLING * np.array([[0.0, 1.0],
                                      [1.0, 0.0]], dtype=complex)

    # Time grid
    times = np.linspace(0.0, T_MAX, N_STEPS)
    dt = times[1] - times[0]

    # Single-step propagator U = exp(-i H dt)
    U = expm(-1j * H * dt)

    # Initial state |ψ_φ(0)>
    phase = np.exp(1j * phi)
    psi0 = (np.array([1.0, 0.0], dtype=complex) +
            phase * np.array([0.0, 1.0], dtype=complex)) / np.sqrt(2.0)

    # Normalize (should already be normalized, but for safety)
    psi0 = psi0 / np.linalg.norm(psi0)

    # Evolve
    psi_t = psi0.copy()
    fidelities = []

    for _ in times:
        # Fidelity F_φ(t) = |<ψ_φ(0) | ψ_φ(t)>|
        overlap = np.vdot(psi0, psi_t)
        F = np.abs(overlap)
        fidelities.append(float(F))

        # Advance: |ψ(t+dt)> = U |ψ(t)>
        psi_t = U @ psi_t
        # Renormalize to mitigate numerical drift
        psi_t = psi_t / np.linalg.norm(psi_t)

    fidelities = np.array(fidelities, dtype=float)
    return times, fidelities


# ---------------------------------------------------------------------
# IO HELPERS
# ---------------------------------------------------------------------

def ensure_output_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def write_dat(filename: str, t: np.ndarray, F: np.ndarray) -> None:
    """
    Write two-column ASCII table:
        t   F
    """
    data = np.column_stack([t, F])
    header = "t\tF\n"
    with open(filename, "w") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.8f\t%.8f")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    ensure_output_dir(OUTPUT_DIR)

    phases = [
        (0.0, "phi0"),             # φ = 0
        (np.pi, "phipi"),          # φ = π
        (np.pi / 2.0, "phihalf"),  # φ = π/2
    ]

    for phi, label in phases:
        print(f"[generate_exchange_fidelity_data] phi = {phi:.6f} ({label})")
        t, F = run_exchange_simulation(phi)
        out_path = os.path.join(OUTPUT_DIR, f"fig_fidelity_{label}.dat")
        write_dat(out_path, t, F)
        print(f"  -> wrote {out_path}")

    print("All exchange fidelity datasets generated.")


if __name__ == "__main__":
    main()
