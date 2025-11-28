#!/usr/bin/env python3
"""
HILBERT VISUALIZER LAYER
========================
Resonance Labs, 2025

3D visualization + wavelet analysis + decoherence microscope
for the Hilbert Substrate framework.

Dependencies:
  - numpy
  - scipy
  - matplotlib
  - pywt  (PyWavelets)  -> pip install pywt

Usage examples (Windows, from repo root):

  # 1) Simple 3D amplitude surface for a hopping excitation
  python hilbert_visualizer.py --mode amplitude3d --n_sites 12 --t_max 10.0 --steps 200 --output amp_surface.png

  # 2) Wavelet scalogram of the density vs site
  python hilbert_visualizer.py --mode wavelet --n_sites 12 --t_max 10.0 --steps 200 --output wavelet.png

  # 3) Decoherence microscope scan on a localized excitation
  python hilbert_visualizer.py --mode decomicro --n_sites 12 --output deco_scan.json
"""

import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import pywt
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from substrate import Substrate  # uses Ben's Substrate class


# ---------------------------------------------------------------------
# Core helper: simulate dynamics on the substrate
# ---------------------------------------------------------------------

@dataclass
class SimulationResult:
    times: np.ndarray          # shape (T,)
    states: np.ndarray         # shape (T, dim) complex
    densities: np.ndarray      # shape (T, n_sites) real


class HilbertVisualizer:
    """
    Visualization + analysis layer sitting on top of the Substrate.

    Design assumptions:
      - For clarity, we default to the single-excitation regime
        (Substrate(..., use_full_space=False)) so that the Hilbert
        space is directly n_sites-dimensional and "site density"
        matches |psi[site]|^2.
      - You *can* use full_space, but some operations (like the
        decoherence microscope) will then be approximations.
    """

    def __init__(self, substrate: Substrate):
        self.sub = substrate

    # -----------------------------------------------------------------
    # 1. Time evolution on the substrate
    # -----------------------------------------------------------------

    def simulate_hopping(
        self,
        t_max: float = 10.0,
        n_steps: int = 200,
        source: int = None,
        hopping: float = 1.0,
    ) -> SimulationResult:
        """
        Simulate a single excitation hopping on the substrate.

        Returns densities(t, site), and full state vectors.
        """
        if source is None:
            source = self.sub.n_sites // 2

        H = self.sub.hopping_hamiltonian(t=hopping)
        dt = t_max / (n_steps - 1)
        U = expm(-1j * H * dt)

        times = np.linspace(0.0, t_max, n_steps)
        psi = self.sub.excitation(source)

        states = np.zeros((n_steps, self.sub.dim), dtype=complex)
        densities = np.zeros((n_steps, self.sub.n_sites), dtype=float)

        for i, _ in enumerate(times):
            states[i] = psi
            densities[i] = self.sub.density(psi)
            psi = U @ psi
            psi = psi / np.linalg.norm(psi)

        return SimulationResult(times=times, states=states, densities=densities)

    # -----------------------------------------------------------------
    # 2. 3D visualizations
    # -----------------------------------------------------------------

    def plot_amplitude_surface(
        self,
        sim: SimulationResult,
        output_path: str = "amplitude_surface.png",
    ) -> None:
        """
        Plot |psi(site, t)|^2 as a 3D surface: x=site, y=time, z=density.
        """
        times = sim.times
        densities = sim.densities  # (T, n_sites)
        n_steps, n_sites = densities.shape

        X, Y = np.meshgrid(np.arange(n_sites), times)
        Z = densities

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z)  # use default colormap
        ax.set_xlabel("Site index")
        ax.set_ylabel("Time")
        ax.set_zlabel("Density |ψ|^2")
        ax.set_title("Substrate excitation density over time")

        plt.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    def plot_state_curve3d(
        self,
        psi: np.ndarray,
        output_path: str = "state_curve3d.png",
    ) -> None:
        """
        Plot a single state as a 3D curve:
          x = site index
          y = Re(psi[site])
          z = Im(psi[site])
          color (implicitly) encodes |psi[site]| via marker size
        """
        n = self.sub.n_sites
        if psi.shape[0] != n:
            raise ValueError("For curve3d, expected single-excitation representation (dim = n_sites).")

        x = np.arange(n)
        y = np.real(psi)
        z = np.imag(psi)
        mag = np.abs(psi)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x, y, z)
        ax.scatter(x, y, z, s=50 * mag / (mag.max() + 1e-12))

        ax.set_xlabel("Site index")
        ax.set_ylabel("Re(ψ)")
        ax.set_zlabel("Im(ψ)")
        ax.set_title("Single state embedding into 3D")

        plt.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    # -----------------------------------------------------------------
    # 3. Wavelet decomposition (1D along the lattice)
    # -----------------------------------------------------------------

    def wavelet_decompose_densities(
        self,
        densities: np.ndarray,
        wavelet: str = "db4",
        level: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a discrete wavelet transform on the density profile
        along the lattice, for each time slice.

        Returns:
          coeff_powers: array shape (n_scales, T) with energy per scale.
          scales: array of scale indices (0..n_scales-1)
        """
        T, n_sites = densities.shape
        w = pywt.Wavelet(wavelet)

        # Use one representative time slice to determine max level
        if level is None:
            max_level = pywt.dwt_max_level(n_sites, w.dec_len)
        else:
            max_level = level

        # We’ll treat each time slice separately and then stack.
        coeff_powers_list = []

        for t_idx in range(T):
            row = densities[t_idx]
            coeffs = pywt.wavedec(row, w, level=max_level)
            # coeffs[0] = approximation, coeffs[1:] = detail at different scales
            # We'll collect total power per scale
            powers = []
            for c in coeffs:
                powers.append(np.sum(np.abs(c) ** 2))
            coeff_powers_list.append(powers)

        coeff_powers = np.array(coeff_powers_list).T  # (n_scales, T)
        scales = np.arange(coeff_powers.shape[0])

        return coeff_powers, scales

    def plot_wavelet_scalogram(
        self,
        sim: SimulationResult,
        output_path: str = "wavelet_scalogram.png",
        wavelet: str = "db4",
        level: int = None,
    ) -> None:
        """
        Plot a "scalogram" of wavelet band power vs time.

        x-axis: time
        y-axis: wavelet scale index (0 = coarsest, larger = finer)
        color: power
        """
        coeff_powers, scales = self.wavelet_decompose_densities(
            sim.densities, wavelet=wavelet, level=level
        )
        times = sim.times

        fig, ax = plt.subplots()
        im = ax.imshow(
            coeff_powers,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], scales[0], scales[-1]],
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Wavelet scale index")
        ax.set_title(f"Wavelet scalogram ({wavelet})")
        fig.colorbar(im, ax=ax, label="Power")

        plt.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    # -----------------------------------------------------------------
    # 4. Decoherence microscope ("deco-micro")
    # -----------------------------------------------------------------

    @staticmethod
    def decoherence_kernel(
        n_sites: int,
        gamma: float,
        t: float,
        distance_power: float = 1.0,
    ) -> np.ndarray:
        """
        Build a simple site-basis decoherence kernel D_ij:

          D_ij = exp(-gamma * t * |i - j|^distance_power)

        When applied as:
          ρ_ij(t) = D_ij * ρ_ij(0)

        this damps off-diagonal coherences, especially for far-separated sites.
        """
        idx = np.arange(n_sites)
        dist = np.abs(idx[:, None] - idx[None, :])  # |i - j|
        D = np.exp(-gamma * t * dist ** distance_power)
        return D

    def deco_micro_scan(
        self,
        psi: np.ndarray,
        gamma_list: List[float],
        t_list: List[float],
        distance_power: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Decoherence microscope:

        - Start from pure state psi (dim = n_sites).
        - Build ρ0 = |psi><psi|.
        - For each (gamma, t) in gamma_list × t_list, apply a
          decoherence kernel in the site basis,
        - Track:
            * total coherence = sum_{i != j} |ρ_ij|
            * diagonal purity  = sum_i (ρ_ii^2)
            * full purity      = Tr(ρ^2)

        Returns:
          {
            "gammas": [...],
            "times": [...],
            "metrics": {
                (gamma, t): { "coherence": ..., "diag_purity": ..., "purity": ... },
                ...
            }
          }
        """
        if psi.shape[0] != self.sub.n_sites:
            raise ValueError("deco_micro_scan expects dim = n_sites (single-excitation representation).")

        rho0 = np.outer(psi, psi.conj())
        metrics: Dict[str, Dict[str, float]] = {}

        for g in gamma_list:
            for t in t_list:
                D = self.decoherence_kernel(self.sub.n_sites, g, t, distance_power=distance_power)
                rho = D * rho0

                # metrics
                off_diag = rho.copy()
                np.fill_diagonal(off_diag, 0.0)
                coherence = float(np.sum(np.abs(off_diag)))

                diag = np.real(np.diag(rho))
                diag_purity = float(np.sum(diag ** 2))

                purity = float(np.real(np.trace(rho @ rho)))

                key = f"g={g:.3g},t={t:.3g}"
                metrics[key] = {
                    "coherence": coherence,
                    "diag_purity": diag_purity,
                    "purity": purity,
                }

        return {
            "gammas": gamma_list,
            "times": t_list,
            "metrics": metrics,
        }

    def deco_micro_profile_for_state(
        self,
        psi: np.ndarray,
        gamma_list: List[float],
        t_list: List[float],
        output_json: str = "deco_scan.json",
    ) -> None:
        """
        Convenience wrapper: run deco_micro_scan and dump to JSON.
        """
        res = self.deco_micro_scan(psi, gamma_list, t_list)
        with open(output_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Wrote decoherence microscope scan to {output_json}")


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hilbert visualizer (3D, wavelets, deco-micro).")
    parser.add_argument("--mode", choices=["amplitude3d", "wavelet", "decomicro"], default="amplitude3d",
                        help="Which visualization / analysis to run.")
    parser.add_argument("--n_sites", type=int, default=12, help="Number of sites in the substrate lattice.")
    parser.add_argument("--t_max", type=float, default=10.0, help="Max simulation time for dynamics.")
    parser.add_argument("--steps", type=int, default=200, help="Number of time steps.")
    parser.add_argument("--output", type=str, default=None, help="Output file (PNG for plots, JSON for deco).")

    args = parser.parse_args()

    # For visualization & deco-micro, we prefer single-excitation representation.
    sub = Substrate(args.n_sites, use_full_space=False)
    vis = HilbertVisualizer(sub)

    if args.mode in ("amplitude3d", "wavelet"):
        sim = vis.simulate_hopping(t_max=args.t_max, n_steps=args.steps)
        if args.mode == "amplitude3d":
            out = args.output or "amplitude_surface.png"
            vis.plot_amplitude_surface(sim, output_path=out)
            print(f"Wrote 3D amplitude surface to {out}")
        else:
            out = args.output or "wavelet_scalogram.png"
            vis.plot_wavelet_scalogram(sim, output_path=out)
            print(f"Wrote wavelet scalogram to {out}")

    elif args.mode == "decomicro":
        # use a localized excitation as the probe object
        psi0 = sub.excitation(sub.n_sites // 2)
        gamma_list = [0.0, 0.1, 0.3, 0.5, 1.0]
        t_list = [0.5, 1.0, 2.0, 5.0]
        out = args.output or "deco_scan.json"
        vis.deco_micro_profile_for_state(psi0, gamma_list, t_list, output_json=out)


if __name__ == "__main__":
    main()
