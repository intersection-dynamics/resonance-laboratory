#!/usr/bin/env python3
"""
===============================================================================
Emergent Hydrogen Pointer Substrate (GPU + Progress + Built-in Analysis)
===============================================================================

One global Hilbert space ψ[i, α, e]:

    i   = lattice node (emergent spacetime site)
    α   = pointer index (0=vac, 1=proton, 2=electron, 3=photon)
    e   = environment index (local environment for decoherence)

Dynamics (unitary only):

  - Graph-local kinetic term on pointer modes (emergent metric).
  - Local emission/absorption (electron <-> photon) in a radial shell.
  - Pointer-dependent environment phases (decoherence) that entangle
    pointer and env but never collapse the state.

Emergent story:

  - "Particles" are decohered pointer patterns in this Hilbert substrate.
  - Hydrogen nucleus = proton pointer lump at origin.
  - Valence electron = excited shell of electron pointer pattern.
  - Photons = pointer pattern in the photon direction, emitted via
    Jaynes–Cummings-like local couplings.

This version:

  - Optional GPU backend via CuPy (`--use-gpu`).
  - Progress reporting (step, %, elapsed, ETA).
  - Safe TeeLogger (no "I/O operation on closed file" at shutdown).
  - Built-in analysis:
      * Prints metrics after the run.
      * Generates:
          figures/radii_vs_time.png
          figures/decoherence_vs_time.png
          figures/radial_profiles.png
          figures/electron_slice_xy.png   <-- NEW: slice of electron environment
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

# Try to import CuPy for GPU support
try:
    import cupy as cp  # type: ignore
    HAVE_CUPY = True
except Exception:
    cp = None  # type: ignore
    HAVE_CUPY = False


# =============================================================================
# Parameter bundle
# =============================================================================


@dataclass
class PointerHydrogenParams:
    # Lattice
    Lx: int = 16
    Ly: int = 16
    Lz: int = 16
    lattice_spacing: float = 1.0  # arbitrary length unit

    # Pointer and environment dimensions
    d_pointer: int = 4   # 0=vac, 1=proton, 2=electron, 3=photon
    d_env: int = 4       # environment subspace per node

    # Time stepping
    dt: float = 0.02
    total_steps: int = 5000
    record_stride: int = 10

    # Graph / kinetic scales
    electron_hopping: float = 1.0
    photon_hopping: float = 1.5   # photons move faster
    proton_hopping: float = 0.0   # pinned nucleus (can make tiny but nonzero)

    # Decoherence / environment
    env_phase_strength: float = 0.3   # phase amplitude per step
    env_phase_correlation_time: int = 1  # redraw phases every N steps

    # Hydrogen structure
    orbital_radius: float = 4.0
    orbital_width: float = 1.0

    # Emission/absorption coupling (local JC)
    coupling_strength: float = 0.15
    coupling_radius: float = 4.0
    coupling_width: float = 1.0

    # Diagnostics
    max_radius: float = 10.0
    n_radial_bins: int = 32

    # RNG
    seed: int = 12345

    # Execution
    use_gpu: bool = False


# =============================================================================
# Utility: array module, run directory, JSON, logging
# =============================================================================


def get_xp(use_gpu: bool):
    if use_gpu and HAVE_CUPY:
        return cp
    return np


def to_numpy(x, xp):
    if xp is np:
        return x
    else:
        return cp.asnumpy(x)  # type: ignore[attr-defined]


def make_run_dir(output_root: str, tag: str | None) -> Tuple[str, str]:
    script_base = "emergent_pointer_hydrogen"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}" + (f"_{tag}" if tag else "")
    run_dir = os.path.join(output_root, script_base, run_id)
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
    return run_id, run_dir


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


class TeeLogger:
    """
    Tee stdout to both console and a log file.

    Safe against interpreter shutdown:
      - Tracks `_closed` flag.
      - After close(), write/flush still forward to console but skip file.
    """

    def __init__(self, path: str):
        self._f = open(path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        self._closed = False

    def write(self, msg: str) -> None:
        self._stdout.write(msg)
        if not self._closed and self._f is not None:
            self._f.write(msg)
            self._f.flush()

    def flush(self) -> None:
        self._stdout.flush()
        if not self._closed and self._f is not None:
            try:
                self._f.flush()
            except ValueError:
                self._closed = True

    def close(self) -> None:
        if not self._closed and self._f is not None:
            try:
                self._f.flush()
                self._f.close()
            except ValueError:
                pass
            self._closed = True
            self._f = None


# =============================================================================
# Emergent spacetime lattice (CPU-side geometric data)
# =============================================================================


class EmergentLattice:
    def __init__(self, Lx: int, Ly: int, Lz: int, a: float):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.a = float(a)
        self.N = Lx * Ly * Lz

        self.positions = self._build_positions()
        self.L = self._build_laplacian()

    def _build_positions(self) -> np.ndarray:
        coords = np.zeros((self.N, 3), dtype=float)
        idx = 0
        for ix in range(self.Lx):
            for iy in range(self.Ly):
                for iz in range(self.Lz):
                    coords[idx] = [ix * self.a, iy * self.a, iz * self.a]
                    idx += 1
        center = coords.mean(axis=0, keepdims=True)
        coords -= center
        return coords

    def _build_laplacian(self) -> np.ndarray:
        N = self.N
        Lx, Ly, Lz = self.Lx, self.Ly, self.Lz
        a2 = self.a ** 2

        Lmat = np.zeros((N, N), dtype=float)

        def idx(ix: int, iy: int, iz: int) -> int:
            return (ix * Ly + iy) * Lz + iz

        for ix in range(Lx):
            for iy in range(Ly):
                for iz in range(Lz):
                    i = idx(ix, iy, iz)
                    diag = 0.0
                    for dx, dy, dz in [
                        (1, 0, 0),
                        (-1, 0, 0),
                        (0, 1, 0),
                        (0, -1, 0),
                        (0, 0, 1),
                        (0, 0, -1),
                    ]:
                        jx, jy, jz = ix + dx, iy + dy, iz + dz
                        if 0 <= jx < Lx and 0 <= jy < Ly and 0 <= jz < Lz:
                            j = idx(jx, jy, jz)
                            Lmat[i, j] = 1.0 / a2
                            diag -= 1.0 / a2
                    Lmat[i, i] = diag
        return Lmat

    def origin_index(self) -> int:
        r2 = np.sum(self.positions ** 2, axis=1)
        return int(np.argmin(r2))


# =============================================================================
# Global Hilbert substrate: pointer + env, GPU-aware
# =============================================================================


class PointerSubstrate:
    """
    Single global Hilbert state ψ[i, α, e] on CPU or GPU.

    i: lattice node
    α: pointer index (vac/proton/electron/photon)
    e: environment index
    """

    def __init__(self, lattice: EmergentLattice, params: PointerHydrogenParams):
        self.lat = lattice
        self.params = params

        self.xp = get_xp(params.use_gpu)

        self.N = lattice.N
        self.d_p = params.d_pointer
        self.d_env = params.d_env

        # Geometry on CPU (for diagnostics)
        self.positions = lattice.positions
        self.nucleus_index = lattice.origin_index()
        self.r = np.linalg.norm(
            self.positions - self.positions[self.nucleus_index],
            axis=1,
        )

        # Pointer indices
        self.IDX_VAC = 0
        self.IDX_PROTON = 1
        self.IDX_ELECTRON = 2
        self.IDX_PHOTON = 3

        # Laplacian on xp
        self.L = self.xp.asarray(lattice.L, dtype=self.xp.float64)

        # Global state: ψ[i, α, e] on xp
        self.psi = self.xp.zeros(
            (self.N, self.d_p, self.d_env),
            dtype=self.xp.complex128,
        )

        # Decoherence phases: θ[i, α, e] on xp
        self.env_phases = self.xp.zeros_like(self.psi, dtype=self.xp.float64)

        # Emission/absorption profile (CPU)
        self.coupling_profile = self._build_coupling_profile()

        # Initialize ψ
        self._initialize_state()

        self.t = 0.0
        self._phase_counter = 0

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------

    def _build_coupling_profile(self) -> np.ndarray:
        p = self.params
        r = self.r
        rc = p.coupling_radius
        w = max(p.coupling_width, 1e-6)
        return np.exp(-0.5 * ((r - rc) / w) ** 2) * p.coupling_strength

    def _initialize_state(self) -> None:
        p = self.params
        rng = np.random.default_rng(p.seed)

        # Proton pointer at nucleus, env e=0
        self.psi[self.nucleus_index, self.IDX_PROTON, 0] = 1.0

        # Electron shell
        r = self.r
        shell_r = p.orbital_radius
        shell_w = max(p.orbital_width, 1e-6)
        envelope = np.exp(-0.5 * ((r - shell_r) / shell_w) ** 2)

        phase_spatial = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=self.N))
        env_state = rng.normal(size=self.d_env) + 1j * rng.normal(size=self.d_env)
        env_state /= np.linalg.norm(env_state)

        psi_e_cpu = envelope * phase_spatial  # (N,)
        psi_e_full_cpu = psi_e_cpu[:, None] * env_state[None, :]  # (N, d_env)

        self.psi[:, self.IDX_ELECTRON, :] += self.xp.asarray(
            psi_e_full_cpu, dtype=self.xp.complex128
        )

        self._renormalize_global()

        if self.xp is not np and hasattr(self.xp.random, "seed"):
            self.xp.random.seed(p.seed)

    def _renormalize_global(self) -> None:
        xp = self.xp
        norm = xp.linalg.norm(self.psi.ravel())
        if norm > 0:
            self.psi /= norm

    # ---------------------------------------------------------------------
    # Time evolution
    # ---------------------------------------------------------------------

    def step(self, dt: float) -> None:
        self._kinetic_step(dt)
        self._emission_absorption_step(dt)
        self._env_decoherence_step()
        self._renormalize_global()
        self.t += dt

    def _kinetic_step(self, dt: float) -> None:
        xp = self.xp
        L = self.L
        p = self.params
        iunit = 1j

        psi = self.psi

        psi_vac = psi[:, self.IDX_VAC, :]
        psi_p = psi[:, self.IDX_PROTON, :]
        psi_e = psi[:, self.IDX_ELECTRON, :]
        psi_ph = psi[:, self.IDX_PHOTON, :]

        k_proton = p.proton_hopping
        k_e = p.electron_hopping
        k_ph = p.photon_hopping

        def laplacian_act(psi_comp, k: float):
            if k == 0.0:
                return psi_comp
            return psi_comp - iunit * dt * (-k * (L @ psi_comp))

        psi_p = laplacian_act(psi_p, k_proton)
        psi_e = laplacian_act(psi_e, k_e)
        psi_ph = laplacian_act(psi_ph, k_ph)

        psi[:, self.IDX_VAC, :] = psi_vac
        psi[:, self.IDX_PROTON, :] = psi_p
        psi[:, self.IDX_ELECTRON, :] = psi_e
        psi[:, self.IDX_PHOTON, :] = psi_ph

        self.psi = psi

    def _emission_absorption_step(self, dt: float) -> None:
        xp = self.xp
        psi = self.psi

        e0, e1 = 0, 1
        profile = xp.asarray(self.coupling_profile, dtype=xp.float64)
        theta = profile * dt

        if xp.all(theta == 0):
            return

        ce = xp.cos(theta)
        se = xp.sin(theta)
        iunit = 1j

        ae = psi[:, self.IDX_ELECTRON, e0]
        ag = psi[:, self.IDX_PHOTON, e1]

        new_ae = ce * ae - iunit * se * ag
        new_ag = -iunit * se * ae + ce * ag

        psi[:, self.IDX_ELECTRON, e0] = new_ae
        psi[:, self.IDX_PHOTON, e1] = new_ag

        self.psi = psi

    def _refresh_env_phases(self) -> None:
        xp = self.xp
        p = self.params
        phi_max = p.env_phase_strength
        shape = self.psi.shape

        if xp is np:
            rng = np.random.default_rng(p.seed + self._phase_counter)
            self.env_phases = rng.uniform(low=-phi_max, high=phi_max, size=shape)
        else:
            self.env_phases = xp.random.uniform(low=-phi_max, high=phi_max, size=shape)

        self._phase_counter += 1

    def _env_decoherence_step(self) -> None:
        xp = self.xp
        p = self.params

        if self._phase_counter % max(1, p.env_phase_correlation_time) == 0:
            self._refresh_env_phases()

        phase_factor = xp.exp(1j * self.env_phases)
        self.psi *= phase_factor

    # ---------------------------------------------------------------------
    # Diagnostics (NumPy side)
    # ---------------------------------------------------------------------

    def pointer_populations_np(self) -> np.ndarray:
        xp = self.xp
        pop = xp.sum(xp.abs(self.psi) ** 2, axis=2)
        return to_numpy(pop, xp)

    def reduced_pointer_density_np(self, i: int) -> np.ndarray:
        xp = self.xp
        v = self.psi[i]  # (d_p, d_env)
        v_np = to_numpy(v, xp)
        return v_np @ v_np.conj().T

    def pointer_coherence_measure(self) -> float:
        N = self.N
        total = 0.0
        for i in range(N):
            rho = self.reduced_pointer_density_np(i)
            off_diag = rho - np.diag(np.diag(rho))
            total += float(np.sum(np.abs(off_diag)))
        return total / N

    def pointer_mean_radius(self, idx: int) -> float:
        pop = self.pointer_populations_np()
        dens = pop[:, idx]
        total = dens.sum()
        if total <= 0:
            return 0.0
        return float((dens * self.r).sum() / total)

    def pointer_radial_profile(
        self, idx: int, r_max: float, n_bins: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        pop = self.pointer_populations_np()
        dens = pop[:, idx]
        r = self.r

        bins = np.linspace(0.0, r_max, n_bins + 1)
        counts = np.zeros(n_bins, dtype=float)
        mass = np.zeros(n_bins, dtype=float)

        for i in range(self.N):
            ri = r[i]
            if ri < 0.0 or ri >= r_max:
                continue
            k = int((ri / r_max) * n_bins)
            if k < 0:
                k = 0
            if k >= n_bins:
                k = n_bins - 1
            counts[k] += 1.0
            mass[k] += dens[i]

        with np.errstate(divide="ignore", invalid="ignore"):
            profile = np.where(counts > 0, mass / counts, 0.0)

        centers = 0.5 * (bins[:-1] + bins[1:])
        return centers, profile


# =============================================================================
# Experiment loop (with progress reporting)
# =============================================================================


def run_experiment(params: PointerHydrogenParams, progress: bool = True) -> Dict[str, Any]:
    np.random.seed(params.seed)

    lattice = EmergentLattice(params.Lx, params.Ly, params.Lz, params.lattice_spacing)
    model = PointerSubstrate(lattice, params)

    steps = params.total_steps
    dt = params.dt
    stride = max(1, params.record_stride)
    n_frames = (steps // stride) + 1

    times = np.zeros(n_frames, dtype=float)
    e_radius = np.zeros(n_frames, dtype=float)
    ph_radius = np.zeros(n_frames, dtype=float)
    proton_radius = np.zeros(n_frames, dtype=float)
    decoherence_C = np.zeros(n_frames, dtype=float)

    r_centers, e_prof0 = model.pointer_radial_profile(
        model.IDX_ELECTRON, params.max_radius, params.n_radial_bins
    )
    _, ph_prof0 = model.pointer_radial_profile(
        model.IDX_PHOTON, params.max_radius, params.n_radial_bins
    )
    _, pr_prof0 = model.pointer_radial_profile(
        model.IDX_PROTON, params.max_radius, params.n_radial_bins
    )

    e_prof_accum = np.zeros_like(e_prof0)
    ph_prof_accum = np.zeros_like(ph_prof0)
    pr_prof_accum = np.zeros_like(pr_prof0)
    prof_count = 0

    frame = 0
    t_start = time.time()
    progress_interval = max(1, steps // 20)  # ~5% chunks

    for step in range(steps + 1):
        if step % stride == 0:
            t = step * dt
            times[frame] = t

            e_radius[frame] = model.pointer_mean_radius(model.IDX_ELECTRON)
            ph_radius[frame] = model.pointer_mean_radius(model.IDX_PHOTON)
            proton_radius[frame] = model.pointer_mean_radius(model.IDX_PROTON)
            decoherence_C[frame] = model.pointer_coherence_measure()

            r_centers, e_prof = model.pointer_radial_profile(
                model.IDX_ELECTRON, params.max_radius, params.n_radial_bins
            )
            _, ph_prof = model.pointer_radial_profile(
                model.IDX_PHOTON, params.max_radius, params.n_radial_bins
            )
            _, pr_prof = model.pointer_radial_profile(
                model.IDX_PROTON, params.max_radius, params.n_radial_bins
            )

            e_prof_accum += e_prof
            ph_prof_accum += ph_prof
            pr_prof_accum += pr_prof
            prof_count += 1

            frame += 1

        if progress and (step % progress_interval == 0 or step == steps):
            elapsed = time.time() - t_start
            frac = step / max(1, steps)
            percent = 100.0 * frac
            eta = elapsed * (1.0 - frac) / max(frac, 1e-8)
            print(
                f"[progress] step {step}/{steps} "
                f"({percent:5.1f}%)  elapsed={elapsed:7.2f}s  ETA~{eta:7.2f}s"
            )

        if step >= steps:
            break

        model.step(dt)

    times = times[:frame]
    e_radius = e_radius[:frame]
    ph_radius = ph_radius[:frame]
    proton_radius = proton_radius[:frame]
    decoherence_C = decoherence_C[:frame]

    if prof_count > 0:
        e_prof_mean = e_prof_accum / prof_count
        ph_prof_mean = ph_prof_accum / prof_count
        pr_prof_mean = pr_prof_accum / prof_count
    else:
        e_prof_mean = e_prof_accum
        ph_prof_mean = ph_prof_accum
        pr_prof_mean = pr_prof_accum

    # NEW: full final electron pointer density (per node)
    final_pop = model.pointer_populations_np()  # shape (N, d_p)
    electron_pop_final = final_pop[:, model.IDX_ELECTRON]  # (N,)

    results: Dict[str, Any] = {
        "positions": lattice.positions,
        "nucleus_index": lattice.origin_index(),
        "times": times,
        "electron_radius": e_radius,
        "photon_radius": ph_radius,
        "proton_radius": proton_radius,
        "decoherence_C": decoherence_C,
        "r_centers": r_centers,
        "electron_radial_mean": e_prof_mean,
        "photon_radial_mean": ph_prof_mean,
        "proton_radial_mean": pr_prof_mean,
        "electron_pop_final": electron_pop_final,  # NEW
    }

    results["final_electron_radius"] = float(e_radius[-1]) if len(e_radius) else 0.0
    results["final_photon_radius"] = float(ph_radius[-1]) if len(ph_radius) else 0.0
    results["final_proton_radius"] = float(proton_radius[-1]) if len(proton_radius) else 0.0
    results["final_decoherence_C"] = float(decoherence_C[-1]) if len(decoherence_C) else 0.0

    return results


# =============================================================================
# Built-in analysis (plots)
# =============================================================================


def generate_plots(run_dir: str, params: PointerHydrogenParams, results: Dict[str, Any]) -> None:
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    times = results["times"]
    e_r = results["electron_radius"]
    ph_r = results["photon_radius"]
    pr_r = results["proton_radius"]
    C = results["decoherence_C"]
    r = results["r_centers"]
    e_prof = results["electron_radial_mean"]
    ph_prof = results["photon_radial_mean"]
    pr_prof = results["proton_radial_mean"]

    # Radii vs time
    plt.figure(figsize=(8, 5))
    plt.plot(times, e_r, label="electron ⟨r⟩")
    plt.plot(times, ph_r, label="photon ⟨r⟩")
    plt.plot(times, pr_r, label="proton ⟨r⟩")
    plt.xlabel("time")
    plt.ylabel("mean radius ⟨r⟩")
    plt.title("Pointer mean radii vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "radii_vs_time.png"), dpi=150)
    plt.close()

    # Decoherence vs time (log if possible)
    plt.figure(figsize=(8, 5))
    if np.any(C > 0):
        plt.semilogy(times, C, label="pointer coherence C")
    else:
        plt.plot(times, C, label="pointer coherence C")
    plt.xlabel("time")
    plt.ylabel("C (avg off-diagonal magnitude)")
    plt.title("Decoherence in pointer basis vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "decoherence_vs_time.png"), dpi=150)
    plt.close()

    # Radial profiles
    plt.figure(figsize=(8, 5))
    plt.plot(r, pr_prof, label="proton", linewidth=2)
    plt.plot(r, e_prof, label="electron", linewidth=2)
    plt.plot(r, ph_prof, label="photon", linewidth=2)
    plt.xlabel("radius r")
    plt.ylabel("avg pointer density")
    plt.title("Time-averaged radial pointer profiles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "radial_profiles.png"), dpi=150)
    plt.close()

    # NEW: Electron environment slice (XY plane at mid Z)
    e_flat = results["electron_pop_final"]  # shape (N,)
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    e_grid = e_flat.reshape((Lx, Ly, Lz))  # matches construction order

    z_mid = Lz // 2
    slice_xy = e_grid[:, :, z_mid].T  # transpose so y is vertical

    plt.figure(figsize=(6, 5))
    plt.imshow(
        slice_xy,
        origin="lower",
        interpolation="nearest",
        extent=[0, Lx * params.lattice_spacing, 0, Ly * params.lattice_spacing],
    )
    plt.colorbar(label="electron pointer density")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Electron pointer density (z-slice at index {z_mid})")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "electron_slice_xy.png"), dpi=150)
    plt.close()


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emergent hydrogen pointer substrate (global Hilbert space + decoherence, GPU-aware)."
    )
    parser.add_argument("--output-root", type=str, default="outputs", help="Output root directory.")
    parser.add_argument("--tag", type=str, default=None, help="Optional tag appended to run_id.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--steps", type=int, default=5000, help="Total number of time steps.")
    parser.add_argument("--record-stride", type=int, default=10, help="Record diagnostics every N steps.")
    parser.add_argument("--L", type=int, default=16, help="Cubic lattice size (Lx=Ly=Lz=L).")
    parser.add_argument("--dt", type=float, default=0.02, help="Time step.")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use CuPy GPU backend if available.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    params = PointerHydrogenParams(
        Lx=args.L,
        Ly=args.L,
        Lz=args.L,
        dt=args.dt,
        total_steps=args.steps,
        record_stride=args.record_stride,
        seed=args.seed,
        use_gpu=args.use_gpu,
    )

    run_id, run_dir = make_run_dir(args.output_root, args.tag)
    log_path = os.path.join(run_dir, "logs", "run.log")

    logger = TeeLogger(log_path)
    original_stdout = logger._stdout
    sys.stdout = logger  # type: ignore[assignment]

    print("=" * 76)
    print("Emergent Hydrogen Pointer Substrate (GPU-aware, Built-in Analysis)")
    print("=" * 76)
    print(f"Run ID   : {run_id}")
    print(f"Run dir  : {run_dir}")
    print(f"Seed     : {args.seed}")
    print(f"Steps    : {args.steps}")
    print(f"Stride   : {args.record_stride}")
    print(f"Lattice  : {args.L} x {args.L} x {args.L}")
    print(f"dt       : {args.dt}")
    print(f"use_gpu  : {args.use_gpu} (CuPy available: {HAVE_CUPY})")
    print("=" * 76)
    print()

    params_path = os.path.join(run_dir, "params.json")
    metadata_path = os.path.join(run_dir, "metadata.json")
    summary_path = os.path.join(run_dir, "summary.json")
    data_path = os.path.join(run_dir, "data", "timeseries.npz")

    save_json(params_path, asdict(params))
    metadata = {
        "script": "emergent_hydrogen_pointer_substrate.py",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "argv": sys.argv,
        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
    }
    save_json(metadata_path, metadata)

    t0 = time.time()
    try:
        results = run_experiment(params, progress=True)
        ok = True
        warnings: List[str] = []
    except Exception as exc:
        ok = False
        warnings = [f"Exception during run: {repr(exc)}"]
        print("ERROR during run:")
        print(repr(exc))
        results = {}
    t1 = time.time()

    if results:
        np.savez_compressed(data_path, **results)

    metrics: Dict[str, Any] = {}
    if results:
        metrics["final_electron_radius"] = results.get("final_electron_radius", 0.0)
        metrics["final_photon_radius"] = results.get("final_photon_radius", 0.0)
        metrics["final_proton_radius"] = results.get("final_proton_radius", 0.0)
        metrics["final_decoherence_C"] = results.get("final_decoherence_C", 0.0)

        e_r0 = float(results["electron_radius"][0])
        e_rf = float(results["final_electron_radius"])
        ph_r0 = float(results["photon_radius"][0])
        ph_rf = float(results["final_photon_radius"])
        C0 = float(results["decoherence_C"][0])
        Cf = float(results["final_decoherence_C"])

        emitted = (ph_rf > ph_r0 + 0.5) and (e_rf < e_r0)
        more_decohered = Cf < C0

        metrics["emission_observed"] = emitted
        metrics["decoherence_in_pointer_basis"] = more_decohered

    diagnostics = {
        "converged": ok,
        "runtime_seconds": t1 - t0,
        "warnings": warnings,
    }

    summary = {
        "script": "emergent_hydrogen_pointer_substrate.py",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "params": asdict(params),
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
    save_json(summary_path, summary)

    print()
    print("---- Run summary ----")
    print(f"Converged            : {ok}")
    print(f"Runtime (s)          : {t1 - t0:0.3f}")
    if results:
        print(f"final_electron_radius: {metrics['final_electron_radius']}")
        print(f"final_photon_radius  : {metrics['final_photon_radius']}")
        print(f"final_proton_radius  : {metrics['final_proton_radius']}")
        print(f"final_decoherence_C  : {metrics['final_decoherence_C']}")
        print(f"emission_observed    : {metrics['emission_observed']}")
        print(f"decoherence_in_basis : {metrics['decoherence_in_pointer_basis']}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
    print("----------------------")

    if results:
        print()
        print("Generating plots into figures/ ...")
        generate_plots(run_dir, params, results)
        print("Plots written: radii_vs_time.png, decoherence_vs_time.png, "
              "radial_profiles.png, electron_slice_xy.png")

    print()
    print("=" * 76)
    print("Run complete.")
    print("=" * 76)

    sys.stdout = original_stdout
    logger.close()


if __name__ == "__main__":
    main()
