#!/usr/bin/env python3
"""
Fermion / Boson Measurement Experiment
======================================
Probe how symmetric ("boson-like") vs antisymmetric ("fermion-like")
two-site patterns respond to different measurement channels implemented
as unitaries coupling to an environment.

This version is tailored to the minimal substrate.py in this repo:
- Uses Substrate only (no detection_unitary, no partial trace helpers).
- Builds its own CNOT-like unitaries, phase-kick, and partial trace.

Usage:
    python tests/measurement_experiment.py
"""

import numpy as np
from substrate import Substrate  # should resolve to the project substrate.py


# ------------------------------------------------------------
# Helper: basis indexing and local unitaries
# ------------------------------------------------------------

def decode_bits(idx: int, n_sites: int):
    """Decode an integer basis index into a list of bits [b0,...,b_{n-1}],
    where b_site is 0 or 1 for that site.

    Convention: site s corresponds to bit at position (n_sites-1-s).
    This matches the usual mapping for 2^n-dimensional qubit Hilbert space.
    """
    return [(idx >> (n_sites - 1 - s)) & 1 for s in range(n_sites)]


def encode_bits(bits, n_sites: int) -> int:
    """Inverse of decode_bits: [b0,...,b_{n-1}] -> integer index."""
    idx = 0
    for s in range(n_sites):
        idx |= (bits[s] & 1) << (n_sites - 1 - s)
    return idx


def cnot_unitary(n_sites: int, control: int, target: int) -> np.ndarray:
    """Build a full 2^n x 2^n CNOT (control -> target) on n_sites qubits."""
    dim = 2 ** n_sites
    U = np.zeros((dim, dim), dtype=complex)
    for k in range(dim):
        bits = decode_bits(k, n_sites)
        new_bits = bits[:]
        if bits[control] == 1:
            new_bits[target] = 1 - new_bits[target]  # flip target
        new_idx = encode_bits(new_bits, n_sites)
        U[new_idx, k] = 1.0
    return U


def phase_kick_unitary(n_sites: int, system_sites, phi: float) -> np.ndarray:
    """Diagonal unitary U = exp(-i phi (n0 - n1)) on given system sites.

    n0, n1 are occupations (0 or 1) of the two system sites.
    """
    dim = 2 ** n_sites
    U = np.zeros((dim, dim), dtype=complex)
    s0, s1 = system_sites
    for k in range(dim):
        bits = decode_bits(k, n_sites)
        n0 = bits[s0]
        n1 = bits[s1]
        phase = np.exp(-1j * phi * (n0 - n1))
        U[k, k] = phase
    return U


# ------------------------------------------------------------
# Helper: partial trace over environment
# ------------------------------------------------------------

def bits_from_subindex(idx: int, sub_len: int):
    """For a subsystem of sub_len qubits, map integer idx -> list of bits
    [b0,...,b_{sub_len-1}] with the same MSB convention.
    """
    return [(idx >> (sub_len - 1 - i)) & 1 for i in range(sub_len)]


def reduced_density_matrix(rho_full: np.ndarray, system_sites, n_sites: int) -> np.ndarray:
    """Trace out all sites not in system_sites from rho_full.

    - n_sites: total number of qubits (dim=2 each).
    - system_sites: list of site indices to keep (e.g. [0,1]).

    Returns a 2^len(system_sites) × 2^len(system_sites) density matrix.
    """
    system_sites = sorted(system_sites)
    env_sites = [s for s in range(n_sites) if s not in system_sites]

    dim_sys = 2 ** len(system_sites)
    dim_env = 2 ** len(env_sites)

    # Precompute mapping from (sys_index, env_index) -> global index
    global_index = np.zeros((dim_sys, dim_env), dtype=int)

    for i_sys in range(dim_sys):
        sys_bits_local = bits_from_subindex(i_sys, len(system_sites))
        for i_env in range(dim_env):
            env_bits_local = bits_from_subindex(i_env, len(env_sites))

            bits_global = [0] * n_sites
            # place system bits
            for k, site in enumerate(system_sites):
                bits_global[site] = sys_bits_local[k]
            # place env bits
            for k, site in enumerate(env_sites):
                bits_global[site] = env_bits_local[k]

            global_index[i_sys, i_env] = encode_bits(bits_global, n_sites)

    rho_sys = np.zeros((dim_sys, dim_sys), dtype=complex)

    # ρ_sys(i,j) = Σ_env ρ_full(global(i,e), global(j,e))
    for i_sys in range(dim_sys):
        for j_sys in range(dim_sys):
            acc = 0.0 + 0.0j
            for i_env in range(dim_env):
                gi = global_index[i_sys, i_env]
                gj = global_index[j_sys, i_env]
                acc += rho_full[gi, gj]
            rho_sys[i_sys, j_sys] = acc

    return rho_sys


# ------------------------------------------------------------
# Build boson-like and fermion-like patterns
# ------------------------------------------------------------

def make_patterns(sub: Substrate, system_sites):
    """
    Build boson-like (symmetric) and fermion-like (antisymmetric)
    2-site patterns on given system_sites, environment in vacuum.

    We work in occupation basis |n0 n1 n2 n3>.
    """
    # Excitation at first system site, others vacuum-like
    psi0 = sub.excitation(system_sites[0])  # e.g. |1000>
    psi1 = sub.excitation(system_sites[1])  # e.g. |0100>

    # Boson-like: symmetric superposition
    psi_B = (psi0 + psi1) / np.sqrt(2.0)

    # Fermion-like: antisymmetric superposition
    psi_F = (psi0 - psi1) / np.sqrt(2.0)

    # Normalize (should already be normalized)
    psi_B = psi_B / np.linalg.norm(psi_B)
    psi_F = psi_F / np.linalg.norm(psi_F)

    return psi_B, psi_F


# ------------------------------------------------------------
# Run measurement protocols
# ------------------------------------------------------------

def run_protocol(sub: Substrate,
                 psi_init: np.ndarray,
                 system_sites,
                 env_sites,
                 protocol: str,
                 n_steps: int = 6):
    """
    Apply a given measurement protocol repeatedly and track:

      - purity(t) of the reduced state on system_sites
      - fidelity(t) with initial reduced state on system_sites

    protocol ∈ {"copy_0", "copy_1", "alt_copy", "phase_kick"}.
    """
    n_sites = sub.n_sites

    # Initial reduced density matrix on system
    rho0_full = np.outer(psi_init, psi_init.conj())
    rho0 = reduced_density_matrix(rho0_full, system_sites, n_sites)

    def measure_state(psi):
        rho_full = np.outer(psi, psi.conj())
        rho = reduced_density_matrix(rho_full, system_sites, n_sites)
        purity = float(np.real(np.trace(rho @ rho)))
        fidelity = float(np.real(np.trace(rho0 @ rho)))
        return purity, fidelity

    purity_seq = []
    fidelity_seq = []

    psi = psi_init.copy()

    # t = 0
    p0, f0 = measure_state(psi)
    purity_seq.append(p0)
    fidelity_seq.append(f0)

    for step in range(1, n_steps + 1):
        # Choose a unitary according to protocol
        if protocol == "copy_0":
            # CNOT from system_sites[0] into alternating env sites
            env_site = env_sites[(step - 1) % len(env_sites)]
            U = cnot_unitary(n_sites, control=system_sites[0], target=env_site)

        elif protocol == "copy_1":
            # CNOT from system_sites[1] into env
            env_site = env_sites[(step - 1) % len(env_sites)]
            U = cnot_unitary(n_sites, control=system_sites[1], target=env_site)

        elif protocol == "alt_copy":
            # Alternate which system site we "measure"
            sys_site = system_sites[(step - 1) % len(system_sites)]
            env_site = env_sites[(step - 1) % len(env_sites)]
            U = cnot_unitary(n_sites, control=sys_site, target=env_site)

        elif protocol == "phase_kick":
            # No env coupling, just opposite phase kicks on the two system sites
            phi = 0.4 * step
            U = phase_kick_unitary(n_sites, system_sites, phi)

        else:
            raise ValueError(f"Unknown protocol: {protocol}")

        # Apply unitary and renormalize
        psi = U @ psi
        psi = psi / np.linalg.norm(psi)

        # Measure after this step
        p, f = measure_state(psi)
        purity_seq.append(p)
        fidelity_seq.append(f)

    return {
        "purity": [float(x) for x in purity_seq],
        "fidelity": [float(x) for x in fidelity_seq],
    }


def print_results(label, result):
    print(f"    {label}:")
    print(f"      purity:   {['%.3f' % v for v in result['purity']]}")
    print(f"      fidelity: {['%.3f' % v for v in result['fidelity']]}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("=" * 60)
    print("FERMION / BOSON MEASUREMENT EXPERIMENT")
    print("=" * 60)

    # 4 sites: 2 system + 2 environment
    sub = Substrate(n_sites=4)
    system_sites = [0, 1]
    env_sites = [2, 3]

    psi_B, psi_F = make_patterns(sub, system_sites)

    protocols = ["copy_0", "copy_1", "alt_copy", "phase_kick"]

    for proto in protocols:
        print(f"\n[PROTOCOL: {proto}]")
        print("-" * 40)

        res_B = run_protocol(sub, psi_B, system_sites, env_sites, proto, n_steps=6)
        res_F = run_protocol(sub, psi_F, system_sites, env_sites, proto, n_steps=6)

        print_results("boson-like (symmetric)", res_B)
        print_results("fermion-like (antisymmetric)", res_F)

        # crude difference metric at final time
        d_final = abs(res_B["fidelity"][-1] - res_F["fidelity"][-1])
        print(f"    |Δ fidelity_final| = {d_final:.3e}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
