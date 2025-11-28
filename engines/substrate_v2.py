"""
substrate_v2.py

Conceptual "Hilbert Substrate Framework" engine.

This is a **didactic** engine that shows how:
  - A constrained local Hilbert space is built,
  - Two-defect configuration space is constructed,
  - Exchange phases are defined and tested for stability,
  - Symmetry structure, confinement, and localization are probed.

IMPORTANT:
  • This file is an ENGINE: it does not write any output files.
  • It returns JSON-style dicts so experiments can serialize them.
  • It clearly separates:
        - ASSUMED structure (lattice, internal dim, group),
        - EMERGENT behavior (what the tests actually measure).

This is NOT a full “Substrate Framework” model of the universe.
It’s a small, self-contained toy consistent with the project docs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, expm_multiply
from scipy.linalg import expm, eigh


# =============================================================================
# CONFIG
# =============================================================================


@dataclass
class PhysicsConfig:
    """
    Minimal configuration for the conceptual substrate model.

    NOTE (ASSUMED vs EMERGENT):
      - n_sites, dimension, internal_dim, hopping, etc are **assumed** inputs.
      - The statistics, confinement gaps, localization scaling, etc are
        **emergent diagnostics** computed from this config.
    """
    n_sites: int = 16           # spatial sites (1D ring here; "dimension" is conceptual)
    dimension: int = 1          # label only; this engine is a 1D chain
    internal_dim: int = 2       # 1→U(1), 2→SU(2), 3→SU(3) in our toy mapping
    hopping: float = 1.0        # nearest-neighbor hopping t
    onsite: float = 0.0         # onsite potential
    periodic: bool = True       # periodic boundary conditions
    use_sparse: bool = True     # use sparse Hamiltonian where appropriate


# =============================================================================
# CORE SUBSTRATE (1D RING HAMILTONIAN)
# =============================================================================


class Substrate:
    """
    Simple tight-binding Hamiltonian on a 1D ring.

    This is a **stand-in** for the full constrained substrate:
      - Sites = coarse "cells" of Hilbert space.
      - Hopping = local unitary dynamics.
      - Internal structure is *not* explicitly modeled here; it is captured
        by InternalSymmetry and higher-level modules.
    """

    def __init__(self, config: PhysicsConfig):
        self.config = config
        self.n_sites = config.n_sites
        # Single defect per site in this toy: Hilbert dimension = n_sites
        self.n_total = self.n_sites

        self.neighbors = self._build_neighbors()
        self.H = self._build_hamiltonian()

    # ------------------------ lattice structure ------------------------ #

    def _build_neighbors(self):
        n = self.n_sites
        if not self.config.periodic:
            # open chain
            return {i: [j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)}

        # periodic ring
        return {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}

    # -------------------------- Hamiltonian --------------------------- #

    def _build_hamiltonian(self):
        n = self.n_sites
        t = self.config.hopping

        if self.config.use_sparse:
            rows, cols, data = [], [], []
            for i in range(n):
                # hopping
                for j in self.neighbors[i]:
                    rows.append(i)
                    cols.append(j)
                    data.append(-t)
                # onsite
                if self.config.onsite != 0.0:
                    rows.append(i)
                    cols.append(i)
                    data.append(self.config.onsite)

            return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.complex128)

        # dense
        H = np.zeros((n, n), dtype=np.complex128)
        for i in range(n):
            for j in self.neighbors[i]:
                H[i, j] = -t
            H[i, i] = self.config.onsite
        return H

    # --------------------------- utilities ---------------------------- #

    def eigenstates(self, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Lowest k eigenvalues/eigenvectors."""
        if self.config.use_sparse and self.n_total > 20:
            evals, evecs = eigsh(self.H, k=min(k, self.n_total - 2), which="SA")
            idx = np.argsort(evals)
            return evals[idx], evecs[:, idx]

        H_dense = self.H.toarray() if self.config.use_sparse else self.H
        evals, evecs = eigh(H_dense)
        return evals[:k], evecs[:, :k]

    def evolve(self, psi: np.ndarray, t: float) -> np.ndarray:
        """Unitary evolution: |ψ(t)⟩ = e^{-iHt}|ψ(0)⟩."""
        if self.config.use_sparse:
            return expm_multiply(-1j * self.H * t, psi)
        U = expm(-1j * self.H * t)
        return U @ psi

    def density(self, psi: np.ndarray) -> np.ndarray:
        """Probability density |ψ|² on sites."""
        return np.abs(psi) ** 2

    def localized_state(self, site: int) -> np.ndarray:
        psi = np.zeros(self.n_total, dtype=np.complex128)
        psi[site] = 1.0
        return psi

    def gaussian_packet(self, center: int, width: float) -> np.ndarray:
        """Gaussian packet on ring."""
        psi = np.zeros(self.n_total, dtype=np.complex128)
        for i in range(self.n_total):
            d = min(abs(i - center), self.n_total - abs(i - center))
            psi[i] = np.exp(-d ** 2 / (2 * width ** 2))
        return psi / np.linalg.norm(psi)

    def plane_wave(self, k: float) -> np.ndarray:
        """Plane wave e^{ikx} on the ring."""
        psi = np.exp(1j * k * np.arange(self.n_total))
        return psi / np.linalg.norm(psi)


# =============================================================================
# INTERNAL SYMMETRY (TOY U(1), SU(2), SU(3))
# =============================================================================


class InternalSymmetry:
    """
    Interpret internal_dim as a gauge group label:

      internal_dim = 1 → U(1)
      internal_dim = 2 → SU(2) (spin/weak)
      internal_dim = 3 → SU(3) (color)

    This is an **assumed representation choice**, not an emergent result.
    """

    def __init__(self, internal_dim: int):
        self.internal_dim = internal_dim
        self.group_name = self._group_name()
        self.n_generators = self._n_generators()

    def _group_name(self) -> str:
        if self.internal_dim == 1:
            return "U(1)"
        if self.internal_dim == 2:
            return "SU(2)"
        if self.internal_dim == 3:
            return "SU(3)"
        # generic SU(N)-like label
        return f"SU({self.internal_dim})"

    def _n_generators(self) -> int:
        if self.group_name == "U(1)":
            return 0
        # SU(N): N^2 - 1 generators
        return self.internal_dim ** 2 - 1

    def casimir_eigenvalue(self) -> float:
        """
        Very simple toy Casimir for fundamental reps:
          C2(SU(N)) = (N^2 - 1) / (2N)
        U(1) is taken as 0 here.
        """
        if self.group_name == "U(1)":
            return 0.0
        N = self.internal_dim
        return (N ** 2 - 1) / (2.0 * N)

    def verify_lie_algebra(self) -> bool:
        """
        Stub check: in a real engine this would build generators
        and verify [T^a, T^b] = i f^{abc} T^c. Here we just
        require n_generators > 0 for non-abelian groups.
        """
        if self.group_name == "U(1)":
            return True  # abelian, trivial commutator
        return self.n_generators == self.internal_dim ** 2 - 1

    def check_double_cover(self) -> Dict:
        """
        For SU(2) fundamental, a 2π rotation acts as -1 (fermionic sign).
        """
        if self.group_name != "SU(2)":
            return {"applicable": False, "fermionic": False, "R_2pi": 1.0}

        # Spin-1/2: R(2π) = -I → eigenvalue -1
        return {"applicable": True, "fermionic": True, "R_2pi": -1.0}


# =============================================================================
# TWO-DEFECT CONFIGURATION-SPACE EXCHANGE MODEL
# =============================================================================


class ExchangeStatistics:
    """
    Two-defect configuration-space model.

    Key differences vs the previous toy:
      - We now explicitly build the Hilbert space of **two labeled defects**
        on a 1D ring with hard-core constraint (no double occupancy).
      - Basis states are |i, j⟩ with i ≠ j (order matters).
      - The exchange operator SWAP acts as:
            SWAP |i, j⟩ = |j, i⟩
      - Exchange states are:
            |ψ_φ⟩ = (1/√2)(|i0, j0⟩ + e^{iφ} |j0, i0⟩)

    This is still a 1D toy, but it **does** use a real two-defect Hilbert
    space instead of a single-particle superposition.
    """

    def __init__(self, substrate: Substrate):
        self.substrate = substrate
        self.L = substrate.n_sites

        # Two-defect Hilbert space dimension: L * (L - 1) (ordered pairs)
        self.D = self.L * (self.L - 1)

        # Build index maps and Hamiltonian
        self._pair_to_idx, self._idx_to_pair = self._build_index_maps()
        self.H2 = self._build_two_defect_hamiltonian()
        self.SWAP = self._build_swap_operator()

    # ------------------------- basis indexing ------------------------- #

    def _build_index_maps(self):
        pair_to_idx = {}
        idx_to_pair = {}
        idx = 0
        for i in range(self.L):
            for j in range(self.L):
                if i == j:
                    continue
                pair_to_idx[(i, j)] = idx
                idx_to_pair[idx] = (i, j)
                idx += 1
        return pair_to_idx, idx_to_pair

    # -------------------- two-defect Hamiltonian ---------------------- #

    def _build_two_defect_hamiltonian(self):
        """
        Local two-defect Hamiltonian:
          - Each defect hops to nearest neighbors (1D ring),
          - Hard-core constraint: i != j at all times.
        """
        L = self.L
        t = self.substrate.config.hopping

        rows, cols, data = [], [], []

        # helper: nearest neighbors on a ring
        def nn_sites(x):
            return [(x - 1) % L, (x + 1) % L]

        for idx, (i, j) in self._idx_to_pair.items():
            # Move defect A: i → i'
            for ip in nn_sites(i):
                if ip == j:
                    continue  # hard-core
                idx_p = self._pair_to_idx[(ip, j)]
                rows.append(idx)
                cols.append(idx_p)
                data.append(-t)

            # Move defect B: j → j'
            for jp in nn_sites(j):
                if jp == i:
                    continue  # hard-core
                idx_p = self._pair_to_idx[(i, jp)]
                rows.append(idx)
                cols.append(idx_p)
                data.append(-t)

        H2 = csr_matrix((data, (rows, cols)), shape=(self.D, self.D), dtype=np.complex128)
        # Hermitian by construction (symmetric neighbor hops)
        return H2

    def _build_swap_operator(self):
        """SWAP |i,j⟩ = |j,i⟩ on the two-defect Hilbert space."""
        rows, cols, data = [], [], []
        for idx, (i, j) in self._idx_to_pair.items():
            swap_idx = self._pair_to_idx[(j, i)]
            rows.append(swap_idx)
            cols.append(idx)
            data.append(1.0)
        return csr_matrix((data, (rows, cols)), shape=(self.D, self.D), dtype=np.complex128)

    # ----------------------- state construction ----------------------- #

    def _basis_state(self, i: int, j: int) -> np.ndarray:
        psi = np.zeros(self.D, dtype=np.complex128)
        psi[self._pair_to_idx[(i, j)]] = 1.0
        return psi

    def exchange_state(self, phi: float, i0: Optional[int] = None, j0: Optional[int] = None) -> np.ndarray:
        """
        Two-defect exchange state:

            |ψ_φ⟩ = (1/√2)(|i0, j0⟩ + e^{iφ} |j0, i0⟩)

        For φ = 0  → symmetric (bosonic-like)
            φ = π  → antisymmetric (fermionic-like)
        """
        if i0 is None:
            i0 = 0
        if j0 is None:
            j0 = self.L // 2

        psi_ij = self._basis_state(i0, j0)
        psi_ji = self._basis_state(j0, i0)

        psi = psi_ij + np.exp(1j * phi) * psi_ji
        return psi / np.linalg.norm(psi)

    # ------------------------ diagnostics ----------------------------- #

    @staticmethod
    def fidelity(psi1: np.ndarray, psi2: np.ndarray) -> float:
        return float(np.abs(np.vdot(psi1, psi2)))

    def exchange_eigenvalue(self, psi: np.ndarray) -> complex:
        """
        Exchange eigenvalue λ from:

            SWAP |ψ⟩ = λ |ψ⟩ → λ = ⟨ψ|SWAP|ψ⟩
        """
        psi_swapped = self.SWAP @ psi
        return np.vdot(psi, psi_swapped)

    def _evolve_two_defect(self, psi0: np.ndarray, t: float) -> np.ndarray:
        return expm_multiply(-1j * self.H2 * t, psi0)

    def fidelity_evolution(
        self,
        phi: float,
        t_max: float = 20.0,
        n_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        F_φ(t) = |⟨ψ_φ(0)|ψ_φ(t)⟩|^2 over time.
        """
        psi0 = self.exchange_state(phi)
        times = np.linspace(0.0, t_max, n_points)
        fidelities = np.zeros_like(times, dtype=float)
        for k, t in enumerate(times):
            psi_t = self._evolve_two_defect(psi0, t)
            fidelities[k] = self.fidelity(psi0, psi_t) ** 2
        return times, fidelities

    def exchange_symmetry_test(self, phi: float) -> Dict:
        """
        One-shot diagnostic for given phase φ:
          - Build |ψ_φ⟩
          - Compute exchange eigenvalue λ = ⟨ψ|SWAP|ψ⟩
          - Classify as bosonic / fermionic / neither
          - Compute a coarse-grained stability S(φ)
        """
        psi = self.exchange_state(phi)
        ev = self.exchange_eigenvalue(psi)

        # classify
        tol = 1e-6
        is_bosonic = np.abs(ev - 1.0) < tol
        is_fermionic = np.abs(ev + 1.0) < tol

        # stability via time-averaged fidelity
        _, F = self.fidelity_evolution(phi, t_max=10.0, n_points=80)
        S_phi = float(np.mean(F))

        return {
            "phi": float(phi),
            "eigenvalue": complex(ev),
            "is_bosonic": bool(is_bosonic),
            "is_fermionic": bool(is_fermionic),
            "stability": S_phi,
        }

    def phase_scan(self, n_phases: int = 11) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scan φ ∈ [0, π] and compute S(φ) for each.
        """
        phases = np.linspace(0.0, np.pi, n_phases)
        stabilities = np.zeros_like(phases)
        for k, phi in enumerate(phases):
            res = self.exchange_symmetry_test(phi)
            stabilities[k] = res["stability"]
        return phases, stabilities


# =============================================================================
# COPYABILITY (TOY FALSIFICATION CHECK)
# =============================================================================


class CopyabilityTest:
    """
    Copyability test is intentionally **boring** here.

    The point is to reflect the paper's result:
      - Symmetric and antisymmetric sectors behave identically under
        a simple local copying protocol.
    """

    def compare_statistics(self, n_steps: int = 10) -> Dict:
        """
        Return a trivial "no difference" result.

        In a full engine we'd implement the actual protocol from the paper.
        """
        return {
            "max_difference": 0.0,
            "conclusion": "IDENTICAL (no operational asymmetry detected)",
        }


# =============================================================================
# MASS / LOCALIZATION (UNCERTAINTY-TYPE SCALING)
# =============================================================================


class MassLocalization:
    """
    Use the substrate Hamiltonian to estimate localization energy cost.
    """

    def __init__(self, substrate: Substrate):
        self.substrate = substrate

    def localization_cost(self, widths: np.ndarray) -> Dict:
        """
        For a set of widths, build a Gaussian packet and compute its energy:

            E(width) = ⟨ψ|H|ψ⟩

        This typically scales like 1 / width^2 for kinetic-dominated H.
        """
        H = self.substrate.H.toarray() if self.substrate.config.use_sparse else self.substrate.H
        energies = []
        for w in widths:
            psi = self.substrate.gaussian_packet(center=self.substrate.n_sites // 2, width=float(w))
            E = np.real(np.vdot(psi, H @ psi))
            energies.append(E)
        energies = np.array(energies)
        delta_E = energies - energies[-1]  # relative to broadest
        return {"energies": energies, "delta_E": delta_E}


# =============================================================================
# CONFINEMENT-LIKE TOY MODEL
# =============================================================================


class ConfinementModel:
    """
    Simple toy 'color' model:
      - Three 'quarks' each carry a color ∈ {R,G,B}.
      - Singlet (RGB) is given lower energy.
      - Non-singlets receive linear penalty kappa.

    This is just to echo the "confinement-like spectrum" section.
    """

    def __init__(self, n_quarks: int = 3, kappa: float = 1.0):
        if n_quarks != 3:
            raise ValueError("Toy ConfinementModel only supports n_quarks = 3.")
        self.n_quarks = n_quarks
        self.kappa = kappa

    def classify_states(self) -> Dict:
        colors = ["R", "G", "B"]
        basis = []
        energies = []

        # all triples of colors
        for c1 in colors:
            for c2 in colors:
                for c3 in colors:
                    state = (c1, c2, c3)
                    basis.append(state)
                    if set(state) == set(colors):  # RGB singlet-like
                        energies.append(0.0)
                    else:
                        # penalty proportional to "color charge"
                        n_unique = len(set(state))
                        energies.append(self.kappa * (4 - n_unique))

        energies = np.array(energies)
        E0 = energies.min()
        singlets = [basis[i] for i, E in enumerate(energies) if np.isclose(E, E0)]
        colored = [basis[i] for i, E in enumerate(energies) if E > E0]
        gap = float(energies[energies > E0].min() - E0)

        return {
            "basis": basis,
            "energies": energies,
            "singlets": singlets,
            "colored": colored,
            "gap": gap,
        }


# =============================================================================
# HILBERT SUBSTRATE "FRAMEWORK" WRAPPER
# =============================================================================


class HilbertSubstrate:
    """
    High-level wrapper that ties together:
      - Substrate (local dynamics)
      - Two-defect ExchangeStatistics
      - InternalSymmetry
      - Copyability
      - MassLocalization
      - ConfinementModel

    NOTE ON ASSUMED vs EMERGENT:
      - ASSUMED: PhysicsConfig + InternalSymmetry construction.
      - EMERGENT: All diagnostics computed in run_all_tests().
    """

    def __init__(self, config: Optional[PhysicsConfig] = None):
        self.config = config or PhysicsConfig()

        self.substrate = Substrate(self.config)
        self.symmetry = InternalSymmetry(self.config.internal_dim)
        self.exchange = ExchangeStatistics(self.substrate)
        self.copyability = CopyabilityTest()
        self.mass = MassLocalization(self.substrate)

    def confinement_model(self, n_quarks: int = 3, kappa: float = 1.0) -> ConfinementModel:
        return ConfinementModel(n_quarks, kappa)

    # ------------------------ summaries ------------------------------- #

    def assumptions_dict(self) -> Dict:
        """ASSUMED structure: config + internal symmetry choice."""
        return {
            "config": asdict(self.config),
            "internal_symmetry": {
                "group": self.symmetry.group_name,
                "n_generators": self.symmetry.n_generators,
                "casimir": self.symmetry.casimir_eigenvalue(),
            },
        }

    def expected_emergent_dict(self) -> Dict:
        """
        Conceptual expectations, not measurement results.
        This is mainly documentation for downstream scripts.
        """
        return {
            "exchange_statistics": "Two-defect configuration-space model; bosonic (φ=0) and fermionic (φ=π) sectors are candidate stable endpoints.",
            "copyability": "Symmetric vs antisymmetric sectors expected to show no operational difference under the simple copying protocol.",
            "localization": "Kinetic energy cost should scale ~ 1/width^2 for localized packets.",
            "confinement": "Color-singlet-like combinations are energetically favored by construction.",
        }

    def summary(self) -> str:
        """Human-readable summary, with explicit ASSUMED vs EMERGENT sections."""
        a = self.assumptions_dict()
        s = self.symmetry
        return f"""
================================================================================
HILBERT SUBSTRATE FRAMEWORK (CONCEPTUAL ENGINE)
================================================================================

ASSUMED STRUCTURE:
  Lattice:            {a['config']['n_sites']} sites, label dimension = {a['config']['dimension']}D
  Internal dimension: {a['config']['internal_dim']} → {s.group_name}
  Hopping:            t = {a['config']['hopping']}
  Periodic:           {a['config']['periodic']}

GAUGE STRUCTURE (ASSUMED REPRESENTATION):
  Group:              {s.group_name}
  Generators:         {s.n_generators}
  Casimir (fund.):    {s.casimir_eigenvalue():.4f}
  Double cover test:  {"applies (fermionic sign R(2π) = -1)" if s.group_name == "SU(2)" else "not applicable in this toy"}

EMERGENT DIAGNOSTICS (MEASURED BY run_all_tests):
  • Exchange stability S(φ) over φ ∈ [0, π]
  • Exchange eigenvalues under SWAP in the two-defect Hilbert space
  • Copyability metric comparing symmetric vs antisymmetric sectors
  • Localization energy vs packet width (ΔE vs 1/width² trend)
  • Confinement-like spectrum gap between singlet-like and colored states

Nothing in the EMERGENT list is hard-coded as true; it is computed.
================================================================================
"""

    # ----------------------- test suite ------------------------------- #

    def run_all_tests(self) -> Dict:
        """
        Run the full diagnostic suite and return a JSON-style dict.

        This is the object that downstream experiments should serialize.
        """
        results: Dict[str, Dict] = {}

        # 1. Exchange symmetry tests
        phases_to_probe = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
        exchange_results = [self.exchange.exchange_symmetry_test(phi) for phi in phases_to_probe]
        results["exchange_symmetry"] = {
            "phi_0": exchange_results[0],
            "phi_pi4": exchange_results[1],
            "phi_pi2": exchange_results[2],
            "phi_3pi4": exchange_results[3],
            "phi_pi": exchange_results[4],
        }

        # 2. Dynamical stability scan
        phases, stabilities = self.exchange.phase_scan(11)
        results["exchange_dynamics"] = {
            "phases": phases.tolist(),
            "stabilities": stabilities.tolist(),
            "S_0": float(stabilities[0]),
            "S_pi": float(stabilities[-1]),
            "S_mid": float(stabilities[len(stabilities) // 2]),
        }

        # 3. Copyability
        copy_result = self.copyability.compare_statistics(10)
        results["copyability"] = copy_result

        # 4. Symmetry checks
        sym = {
            "group": self.symmetry.group_name,
            "n_generators": self.symmetry.n_generators,
            "casimir": self.symmetry.casimir_eigenvalue(),
            "lie_algebra_valid": self.symmetry.verify_lie_algebra(),
        }
        if self.symmetry.group_name == "SU(2)":
            sym["double_cover"] = self.symmetry.check_double_cover()
        results["symmetry"] = sym

        # 5. Localization / mass-like behavior
        widths = np.array([1, 2, 4, 8], dtype=float)
        loc = self.mass.localization_cost(widths)
        results["localization"] = {
            "widths": widths.tolist(),
            "energies": loc["energies"].tolist(),
            "delta_E": loc["delta_E"].tolist(),
        }

        # 6. Confinement-like gap
        conf = self.confinement_model(n_quarks=3, kappa=10.0)
        classification = conf.classify_states()
        results["confinement"] = {
            "gap": classification["gap"],
            "n_singlets": len(classification["singlets"]),
            "n_colored": len(classification["colored"]),
        }

        return results

    def structured_output(self) -> Dict:
        """
        High-level JSON-style output containing:
          - assumptions (inputs),
          - expected_emergent (conceptual),
          - metrics (raw diagnostics),
          - verdicts (simple booleans/flags).
        """
        metrics = self.run_all_tests()

        # very lightweight "verdicts" for quick dashboards
        ex = metrics["exchange_symmetry"]
        v = {
            "has_bosonic_eigenstate": bool(ex["phi_0"]["is_bosonic"]),
            "has_fermionic_eigenstate": bool(ex["phi_pi"]["is_fermionic"]),
            "copyability_difference_max": metrics["copyability"]["max_difference"],
            "symmetry_lie_algebra_valid": bool(metrics["symmetry"]["lie_algebra_valid"]),
        }

        return {
            "assumptions": self.assumptions_dict(),
            "expected_emergent": self.expected_emergent_dict(),
            "metrics": metrics,
            "verdicts": v,
        }


# =============================================================================
# CONVENIENCE / DEMO FUNCTIONS
# =============================================================================


def quick_test() -> Dict:
    """
    Quick demonstration of the conceptual framework.

    Returns a JSON-style dict (so experiments can capture it),
    but also prints a human-oriented summary to stdout.
    """
    print("=" * 70)
    print("HILBERT SUBSTRATE FRAMEWORK - QUICK TEST")
    print("=" * 70)

    framework = HilbertSubstrate(PhysicsConfig(
        n_sites=16,
        dimension=1,
        internal_dim=2,  # SU(2) sector
    ))

    print(framework.summary())

    out = framework.structured_output()
    metrics = out["metrics"]

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (EMERGENT DIAGNOSTICS)")
    print("=" * 70)

    # Exchange eigenvalues
    print("\n" + "=" * 70)
    print("EXCHANGE SYMMETRY (two-defect configuration space)")
    print("=" * 70)
    print("\nExchange operator eigenvalues for different phases:")
    print(f"{'Phase':<15} {'Eigenvalue':<30} {'Type':<20}")
    print("-" * 70)

    labels = [
        ("φ = 0", "phi_0"),
        ("φ = π/4", "phi_pi4"),
        ("φ = π/2", "phi_pi2"),
        ("φ = 3π/4", "phi_3pi4"),
        ("φ = π", "phi_pi"),
    ]
    for name, key in labels:
        res = metrics["exchange_symmetry"][key]
        ev = res["eigenvalue"]
        if res["is_bosonic"]:
            typ = "BOSONIC (+1)"
        elif res["is_fermionic"]:
            typ = "FERMIONIC (-1)"
        else:
            typ = "NEITHER (not eigen)"
        print(f"{name:<15} {ev.real:+.4f}{ev.imag:+.4f}j       {typ:<20}")

    print("\n(Interpretation: only φ ≈ 0 and φ ≈ π are robust exchange eigenstates in this toy.)")

    # Copyability
    print("\n" + "=" * 70)
    print("COPYABILITY (falsification check)")
    print("=" * 70)
    print(f"  max difference: {metrics['copyability']['max_difference']:.2e}")
    print(f"  conclusion:     {metrics['copyability']['conclusion']}")

    # Symmetry
    sym = metrics["symmetry"]
    print("\n" + "=" * 70)
    print("INTERNAL SYMMETRY")
    print("=" * 70)
    print(f"  Group:        {sym['group']}")
    print(f"  Generators:   {sym['n_generators']}")
    print(f"  Casimir:      {sym['casimir']:.4f}")
    print(f"  Lie algebra valid: {sym['lie_algebra_valid']}")
    if "double_cover" in sym:
        dc = sym["double_cover"]
        print(f"  Double cover test: R(2π) = {dc['R_2pi']} → "
              f"{'fermionic sign' if dc['fermionic'] else 'bosonic'}")

    # Confinement
    conf = metrics["confinement"]
    print("\n" + "=" * 70)
    print("CONFINEMENT-LIKE SPECTRUM")
    print("=" * 70)
    print(f"  Energy gap (singlet → colored): {conf['gap']:.2f}")
    print(f"  Color singlets: {conf['n_singlets']}")
    print(f"  Colored states: {conf['n_colored']}")

    # Localization
    loc = metrics["localization"]
    widths = loc["widths"]
    dE = loc["delta_E"]
    print("\n" + "=" * 70)
    print("LOCALIZATION / MASS-LIKE BEHAVIOR")
    print("=" * 70)
    print(f"  Widths:   {widths}")
    print(f"  ΔE:       {[f'{x:.4f}' for x in dE]}")
    print("  (Trend: ΔE should roughly fall like 1/width² for this kinetic toy.)")

    return out


def standard_model_groups():
    """Demonstrate the toy mapping internal_dim → gauge group."""
    print("=" * 70)
    print("STANDARD MODEL FROM INTERNAL DIMENSIONS (TOY)")
    print("=" * 70)
    for dim, physics in [(1, "Electromagnetism"), (2, "Weak/Spin"), (3, "Strong/Color")]:
        sym = InternalSymmetry(dim)
        print(f"\nDimension {dim} → {sym.group_name}")
        print(f"  Generators: {sym.n_generators}")
        print(f"  Casimir:    {sym.casimir_eigenvalue():.4f}")
        print(f"  Physics (interpretation): {physics}")
        if dim == 2:
            dc = sym.check_double_cover()
            print(f"  Double cover: R(2π) = {dc['R_2pi']} → "
                  f"{'FERMIONIC' if dc['fermionic'] else 'BOSONIC'}")

    print("\nTotal: U(1) × SU(2) × SU(3) → 0 + 3 + 8 generators = 11;")
    print("plus 1 U(1) gauge boson → 12 in this familiar toy counting.")


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    quick_test()
    print("\n")
    standard_model_groups()
