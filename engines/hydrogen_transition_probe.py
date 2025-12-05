#!/usr/bin/env python3
"""
Hydrogen Transition Probe
=========================

Goal
----
Use the merged Hilbert Substrate engine (CPU, lattice mode) to set up a
minimal "hydrogen-like" scenario:

- A designated proton "core" at the center of an L×L lattice.
- Electron-like bound modes (ground and excited) identified from the
  substrate's own coupling matrix.
- A photon-like excitation encoded as local gauge flux / phase structure
  in a ring around the core.
- Pure unitary evolution using Substrate.evolve(...).
- Diagnostics tracking:
    * electron ground/excited mode occupations,
    * radial density profiles relative to the core,
    * "photon energy" proxy from gauge curvature around the core.

This is deliberately an *experiment* script:
- It imports the substrate engine from substrate_merged.py.
- It writes outputs under outputs/hydrogen_transition_probe/<run_id>/.
- It does not modify the engine.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from substrate import Config as SubConfig, Substrate


# =============================================================================
# Experiment configuration
# =============================================================================


@dataclass
class ExperimentConfig:
    # Lattice parameters
    L: int = 8
    internal_dim: int = 8

    # Evolution
    steps: int = 4000
    record_every: int = 10

    # Substrate dynamics
    defrag_rate: float = 0.0
    gauge_phase_rate: float = 0.05
    gauge_noise: float = 0.01
    dt: float = 0.05

    # Random seed
    seed: int = 42

    # Output
    output_root: str = "outputs"
    tag: str = "hydrogen_probe"


# =============================================================================
# Helpers: filesystem and JSON
# =============================================================================


def make_run_dir(exp_cfg: ExperimentConfig) -> Tuple[Path, str]:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{timestamp}_{exp_cfg.tag}"
    root = Path(exp_cfg.output_root).resolve()
    run_dir = root / "hydrogen_transition_probe" / run_id

    if run_dir.exists():
        raise RuntimeError(f"Run directory already exists: {run_dir}")

    (run_dir / "data").mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=False)
    return run_dir, run_id


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


# =============================================================================
# Physics helpers
# =============================================================================


def build_substrate(exp: ExperimentConfig) -> Substrate:
    """
    Build an L×L lattice substrate suitable for a hydrogen-like toy model.
    """
    cfg = SubConfig(
        geometry="lattice",
        L=exp.L,
        internal_dim=exp.internal_dim,
        defrag_rate=exp.defrag_rate,
        gauge_phase_rate=exp.gauge_phase_rate,
        gauge_noise=exp.gauge_noise,
        dt=exp.dt,
        seed=exp.seed,
    )
    sub = Substrate(cfg)
    return sub


def core_index(sub: Substrate) -> int:
    """
    Return the index of the central lattice site, to be treated as the "proton core".
    """
    if sub.geometry != "lattice":
        raise ValueError("core_index only defined for lattice geometry")

    L = sub.L
    cx = L // 2
    cy = L // 2
    return sub._idx(cx, cy)  # using engine's own index helper


def lattice_positions(sub: Substrate) -> np.ndarray:
    """
    Return an array of shape (n_nodes, 2) with lattice (x, y) positions.
    """
    if sub.geometry != "lattice":
        raise ValueError("lattice_positions only defined for lattice geometry")
    coords = np.zeros((sub.n_nodes, 2), dtype=int)
    for i in range(sub.n_nodes):
        x, y = sub._positions[i]
        coords[i, 0] = x
        coords[i, 1] = y
    return coords


def radial_distances(sub: Substrate, center_idx: int) -> np.ndarray:
    """
    Euclidean distance on the lattice between each node and the center node.
    """
    coords = lattice_positions(sub)
    cx, cy = coords[center_idx]
    dx = coords[:, 0] - cx
    dy = coords[:, 1] - cy
    return np.sqrt(dx**2 + dy**2)


def compute_electron_modes(sub: Substrate, core_idx: int, n_modes: int = 2) -> np.ndarray:
    """
    Compute a few "electron-like" bound modes from the substrate's own couplings.

    We use the single-particle hopping matrix H_hop = J_mag * exp(i A), which
    lives on the lattice sites (not the internal_dim factor). We then:

    - Diagonalize H_hop.
    - Sort eigenmodes by how strongly they are localized near the core.
    - Return the n_modes most core-localized eigenvectors as columns of shape
      (n_nodes, n_modes).

    These define effective "orbitals" living on nodes; we project the Substrate
    state onto these using the first two internal components as the electron
    sector.
    """
    H_hop = sub._J_mag * np.exp(1j * sub._A)
    eigvals, eigvecs = np.linalg.eigh(H_hop)

    core_weights = np.abs(eigvecs[core_idx, :]) ** 2
    idx_sorted = np.argsort(core_weights)[::-1]  # descending
    chosen = idx_sorted[:n_modes]

    modes = eigvecs[:, chosen]  # shape (n_nodes, n_modes)
    norms = np.linalg.norm(modes, axis=0, keepdims=True)
    norms[norms == 0.0] = 1.0
    modes /= norms
    return modes


def project_electron_modes(sub: Substrate, modes: np.ndarray) -> np.ndarray:
    """
    Project the current Substrate state onto the supplied node-modes using
    internal components 0 and 1 as the electron "spinor sector".

    Returns an array of shape (n_modes,) of mode amplitudes.
    """
    psi = sub.states  # (n_nodes, internal_dim)
    n_nodes, d = psi.shape
    n_modes = modes.shape[1]

    if d < 2:
        raise ValueError("internal_dim must be >= 2 for electron pseudo-spin sector")

    # Electron vector per node: simple combination of components 0 and 1
    electron_vec = psi[:, 0] + psi[:, 1]

    amps = np.zeros(n_modes, dtype=np.complex128)
    for k in range(n_modes):
        mode_k = modes[:, k]
        amps[k] = np.vdot(mode_k, electron_vec)
    return amps


def inject_photon_like_flux(sub: Substrate, core_idx: int, inner_r: float, outer_r: float) -> None:
    """
    Modify the gauge field A_ij to create a ring-like flux structure around the core.

    - For edges whose endpoints lie in a radial band [inner_r, outer_r] from the core,
      we add a phase bias (small circulation).
    - This acts as a photon-like excitation in the gauge sector, not as a new rule.
    """
    if sub.geometry != "lattice":
        raise ValueError("inject_photon_like_flux only defined for lattice geometry")

    r = radial_distances(sub, core_idx)
    A = sub._A.copy()
    J_mag = sub._J_mag

    phase_bump = 0.3  # radians; small compared to π

    for i in range(sub.n_nodes):
        if not (inner_r <= r[i] <= outer_r):
            continue
        for j in sub._neighbors[i]:
            if not (inner_r <= r[j] <= outer_r):
                continue
            if J_mag[i, j] <= 0.0:
                continue
            A[i, j] += phase_bump
            A[j, i] -= phase_bump

    A = (A + np.pi) % (2.0 * np.pi) - np.pi

    sub._A = A
    sub._update_couplings_from_mag_and_phase()


def photon_energy_proxy(sub: Substrate, core_idx: int, inner_r: float, outer_r: float) -> float:
    """
    Crude "photon energy" proxy: sum of squared plaquette flux magnitudes
    in an annulus around the core.

    - Uses Substrate.all_plaquette_fluxes() which returns an array of shape
      (L-1, L-1) of flux values (real phases).
    - We map each plaquette to the node at its lower-left corner (x, y)
      to estimate its radial position.
    """
    if sub.geometry != "lattice":
        raise ValueError("photon_energy_proxy only defined for lattice geometry")

    r = radial_distances(sub, core_idx)
    fluxes = sub.all_plaquette_fluxes()  # shape (L-1, L-1)
    L = sub.L

    total = 0.0
    for x in range(L - 1):
        for y in range(L - 1):
            # Representative node for this plaquette: lower-left corner (x, y)
            idx = sub._idx(x, y)
            rr = r[idx]
            if inner_r <= rr <= outer_r:
                phi = fluxes[x, y]
                total += float(phi**2)
    return total


def radial_density_profile(sub: Substrate, core_idx: int, n_bins: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute binned radial electron density profile:

    - Electron density per node = |psi_0|^2 + |psi_1|^2.
    - Bin by distance from core into n_bins between 0 and r_max.
    """
    psi = sub.states
    r = radial_distances(sub, core_idx)

    electron_density = np.abs(psi[:, 0])**2 + np.abs(psi[:, 1])**2

    r_max = float(r.max())
    if r_max == 0.0:
        r_max = 1.0
    edges = np.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profile = np.zeros(n_bins, dtype=float)

    for b in range(n_bins):
        mask = (r >= edges[b]) & (r < edges[b + 1])
        if np.any(mask):
            profile[b] = float(np.sum(electron_density[mask]))
        else:
            profile[b] = 0.0

    return centers, profile


# =============================================================================
# Main experiment
# =============================================================================


def run_experiment(exp: ExperimentConfig) -> Dict[str, Any]:
    # Build run directory
    run_dir, run_id = make_run_dir(exp)
    log_path = run_dir / "logs" / "run.log"

    header = [
        "=" * 70,
        "HYDROGEN TRANSITION PROBE",
        "=" * 70,
        f"Run ID: {run_id}",
        f"Lattice size L: {exp.L}",
        f"internal_dim: {exp.internal_dim}",
        f"steps: {exp.steps}, record_every: {exp.record_every}",
        f"dt: {exp.dt}",
        f"defrag_rate: {exp.defrag_rate}",
        f"gauge_phase_rate: {exp.gauge_phase_rate}",
        f"gauge_noise: {exp.gauge_noise}",
        f"seed: {exp.seed}",
        "",
    ]
    log_text = "\n".join(header)
    print(log_text)
    log_path.write_text(log_text + "\n")

    # Build substrate
    sub = build_substrate(exp)
    core_idx = core_index(sub)
    coords = lattice_positions(sub)
    cx, cy = coords[core_idx]
    print(f"Core (proton) at lattice site ({cx}, {cy}), node index {core_idx}")

    # Compute electron modes (ground and excited)
    modes = compute_electron_modes(sub, core_idx, n_modes=2)
    mode_labels = ["ground", "excited"]

    # Align initial state so electron lives mostly in the ground mode
    ground_mode = modes[:, 0]
    psi0 = sub.states.copy()
    psi0[:, 0] = ground_mode
    psi0[:, 1] = 0.0
    norms = np.linalg.norm(psi0, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    psi0 /= norms
    sub.states = psi0

    # Inject photon-like flux ring around the core
    r = radial_distances(sub, core_idx)
    r_nonzero = r[r > 0.0]
    r_med = float(np.median(r_nonzero)) if r_nonzero.size > 0 else 1.0
    inner_r = 0.75 * r_med
    outer_r = 1.25 * r_med
    print(f"Injecting photon-like flux in annulus [{inner_r:.2f}, {outer_r:.2f}]")
    inject_photon_like_flux(sub, core_idx, inner_r=inner_r, outer_r=outer_r)

    # Time evolution and diagnostics
    n_records = exp.steps // exp.record_every + 1
    times = np.zeros(n_records, dtype=float)
    mode_occupancies = np.zeros((n_records, 2), dtype=float)  # ground, excited
    photon_energy = np.zeros(n_records, dtype=float)
    radial_bins = None
    radial_profiles = []

    record_idx = 0
    for step in range(exp.steps + 1):
        if step % exp.record_every == 0:
            t = step * exp.dt
            times[record_idx] = t

            # Project onto electron modes
            amps = project_electron_modes(sub, modes)
            occ = np.abs(amps) ** 2
            mode_occupancies[record_idx, :] = occ

            # Photon energy proxy
            photon_energy[record_idx] = photon_energy_proxy(sub, core_idx, inner_r, outer_r)

            # Radial profile
            rbins, rprof = radial_density_profile(sub, core_idx, n_bins=6)
            if radial_bins is None:
                radial_bins = rbins
            radial_profiles.append(rprof)

            record_idx += 1

        if step < exp.steps:
            sub.evolve(n_steps=1, defrag_rate=exp.defrag_rate)

    radial_profiles = np.asarray(radial_profiles, dtype=float)  # (n_records, n_bins)

    # Build summary + save data
    summary: Dict[str, Any] = {
        "framework_version": "0.1.0",
        "script": "hydrogen_transition_probe.py",
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": asdict(exp),
        "metrics": {
            "final_ground_occupancy": float(mode_occupancies[-1, 0]),
            "final_excited_occupancy": float(mode_occupancies[-1, 1]),
            "max_excited_occupancy": float(mode_occupancies[:, 1].max()),
            "max_photon_energy_proxy": float(photon_energy.max()),
        },
        "diagnostics": {
            "n_records": int(n_records),
            "core_index": int(core_idx),
            "core_position": [int(cx), int(cy)],
            "mode_labels": mode_labels,
        },
        "verdicts": {
            "has_nontrivial_photon_activity": bool(photon_energy.max() > 1e-4),
            "has_excited_mode_activity": bool(mode_occupancies[:, 1].max() > 1e-3),
        },
    }

    data_dir = run_dir / "data"
    np.save(data_dir / "times.npy", times)
    np.save(data_dir / "mode_occupancies.npy", mode_occupancies)
    np.save(data_dir / "photon_energy_proxy.npy", photon_energy)
    np.save(data_dir / "radial_bins.npy", radial_bins)
    np.save(data_dir / "radial_profiles.npy", radial_profiles)

    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "params.json", asdict(exp))

    print("=" * 70)
    print("Run complete.")
    print(f"Run directory: {run_dir}")
    print("Key metrics:")
    print(f"  max excited occupancy: {summary['metrics']['max_excited_occupancy']:.4f}")
    print(f"  max photon energy proxy: {summary['metrics']['max_photon_energy_proxy']:.4f}")
    print("=" * 70)

    return summary


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser(description="Hydrogen transition probe on Substrate lattice.")
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--internal-dim", type=int, default=8)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--record-every", type=int, default=10)
    p.add_argument("--defrag-rate", type=float, default=0.0)
    p.add_argument("--gauge-phase-rate", type=float, default=0.05)
    p.add_argument("--gauge-noise", type=float, default=0.01)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", type=str, default="outputs")
    p.add_argument("--tag", type=str, default="hydrogen_probe")

    args = p.parse_args()

    return ExperimentConfig(
        L=args.L,
        internal_dim=args.internal_dim,
        steps=args.steps,
        record_every=args.record_every,
        defrag_rate=args.defrag_rate,
        gauge_phase_rate=args.gauge_phase_rate,
        gauge_noise=args.gauge_noise,
        dt=args.dt,
        seed=args.seed,
        output_root=args.output_root,
        tag=args.tag,
    )


def main() -> None:
    exp_cfg = parse_args()
    run_experiment(exp_cfg)


if __name__ == "__main__":
    main()
