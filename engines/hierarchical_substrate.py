"""
engines/hierarchical_substrate.py

Hierarchical Hilbert substrate engine.

Idea:
  - We do NOT think of "particles" moving on a lattice.
  - Instead, we have a hierarchy of Hilbert spaces:

        Layer 0: H^(0)  (e.g. positions on a ring)
        Layer 1: H^(1)  (e.g. internal SU(2) spin)
        Layer 2: H^(2)  (e.g. "micro-substrate" for each spin state)
        ...

  - The total Hilbert space is built from these layers by tensor products:

        H_total = H^(0) ⊗ H^(1) ⊗ H^(2) ⊗ ...

    or, more generally, by mixing tensor products and direct sums.

  - Each layer carries its own local Hamiltonian H^(ℓ).
  - The global Hamiltonian is:

        H = H^(0) ⊗ I ⊗ I ... +
            I ⊗ H^(1) ⊗ I ... +
            ...

This module aims to give you a **structured way** to define such
hierarchies and to flatten them into actual matrices, while keeping
the "substrate all the way down" philosophy explicit in the code.

We keep it small and explicit, so you can swap parts out as the theory evolves.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple, Sequence, Union

import numpy as np
from numpy.linalg import eigh
from scipy.linalg import expm


# =============================================================================
# Basic layer specification
# =============================================================================


@dataclass
class LayerSpec:
    """
    Specification of a single Hilbert layer.

    Attributes
    ----------
    name : str
        Human-readable name ("position", "spin", "micro-substrate", ...).

    dim : int
        Dimension of the Hilbert space for this layer.

    type : str
        Either:
          - "tensor": this layer is tensor-producted with the previous ones.
          - "direct_sum": this layer is an alternative sector (not used in
            the simple example, but included for completeness).

        For now, we support only a straightforward tensor-product hierarchy.

    hamiltonian_builder : Optional[callable]
        Function that returns a (dim, dim) Hermitian matrix H^(ℓ) for this
        layer. If None, H^(ℓ) = 0.

        Signature:
            def builder(spec: LayerSpec) -> np.ndarray:
                ...
    """
    name: str
    dim: int
    type: str = "tensor"
    hamiltonian_builder: Optional[Any] = None


# =============================================================================
# Utility: Kronecker builds for hierarchical H
# =============================================================================


def kron_all(ops: Sequence[np.ndarray]) -> np.ndarray:
    """
    Compute Kronecker product of a sequence of operators:

        kron_all([A, B, C]) = A ⊗ B ⊗ C
    """
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def identity(dim: int) -> np.ndarray:
    return np.eye(dim, dtype=np.complex128)


# =============================================================================
# Example Hamiltonian builders for layers
# =============================================================================


def position_ring_builder(spec: LayerSpec, hopping: float = 1.0) -> np.ndarray:
    """
    Tight-binding ring on 'dim' sites:

        H = -t ∑_i ( |i><i+1| + |i+1><i| )

    with periodic boundary conditions.
    """
    n = spec.dim
    H = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        j = (i + 1) % n
        H[i, j] = -hopping
        H[j, i] = -hopping
    return H


def internal_spin_builder(spec: LayerSpec) -> np.ndarray:
    """
    Simple SU(2) "spin" layer, represented as a qubit.

    We choose a toy Hamiltonian:

        H = Δ σ^z

    with Δ = 1.0 by default. If dim != 2, we return zeros.
    """
    if spec.dim != 2:
        return np.zeros((spec.dim, spec.dim), dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    delta = 1.0
    return delta * sz


def micro_substrate_builder(spec: LayerSpec) -> np.ndarray:
    """
    Tiny "micro-substrate" Hamiltonian, here just a 2x2 toy.

        H = [[0, -g], [-g, 0]]

    If dim != 2, return zeros.
    """
    if spec.dim != 2:
        return np.zeros((spec.dim, spec.dim), dtype=np.complex128)
    g = 0.5
    H = np.array([[0, -g], [-g, 0]], dtype=np.complex128)
    return H


# =============================================================================
# Hierarchical substrate class
# =============================================================================


class HierarchicalSubstrate:
    """
    A stack of Hilbert layers:

        layer 0: position (dim L0)
        layer 1: spin (dim L1)
        layer 2: micro-substrate (dim L2)
        ...

    with a global Hilbert space:

        H_total = H^(0) ⊗ H^(1) ⊗ ... (for tensor layers)

    and global Hamiltonian:

        H = ∑_ℓ ( I ⊗ ... ⊗ H^(ℓ) ⊗ ... ⊗ I )

    This is **substrate all the way down**:
      - each layer is a "substrate" with its own local Hilbert space,
      - the total system is the combined substrate of all layers.

    We do not talk about "particles" or "fields" here; the only objects
    are Hilbert spaces and Hermitian generators.
    """

    def __init__(self, layers: List[LayerSpec]):
        if not layers:
            raise ValueError("At least one layer must be specified.")

        # Currently we only support tensor-product layers
        for layer in layers:
            if layer.type != "tensor":
                raise NotImplementedError("Only 'tensor' layers are currently supported.")

        self.layers = layers
        self.n_layers = len(layers)
        self.layer_dims = [layer.dim for layer in layers]

        # global dimension = product of layer dims
        self.dim_total = int(np.prod(self.layer_dims))

        # build local H^(ℓ)
        self.local_Hs: List[np.ndarray] = []
        for layer in self.layers:
            if layer.hamiltonian_builder is None:
                H_local = np.zeros((layer.dim, layer.dim), dtype=np.complex128)
            else:
                # If builder takes only spec, ignore extra args
                H_local = layer.hamiltonian_builder(layer)
            self.local_Hs.append(H_local)

        # build global Hamiltonian
        self.H_total = self._build_global_hamiltonian()

    # ------------------------ global H build -------------------------- #

    def _build_global_hamiltonian(self) -> np.ndarray:
        """
        H = ∑_ℓ ( I ⊗ ... ⊗ H^(ℓ) ⊗ ... ⊗ I )
        """
        H_total = np.zeros((self.dim_total, self.dim_total), dtype=np.complex128)

        for ell in range(self.n_layers):
            ops = []
            for m in range(self.n_layers):
                if m == ell:
                    ops.append(self.local_Hs[m])
                else:
                    ops.append(identity(self.layer_dims[m]))
            H_ell = kron_all(ops)
            H_total += H_ell

        # Hermitize explicitly to remove tiny numerical asymmetries
        H_total = 0.5 * (H_total + H_total.conj().T)
        return H_total

    # ---------------------- evolution utilities ---------------------- #

    def evolve_state(self, psi0: np.ndarray, t: float) -> np.ndarray:
        """
        Unitary evolution:

            |ψ(t)⟩ = e^{-i H t} |ψ(0)⟩

        This uses a dense expm, so only feasible for modest dim_total.
        In later versions you can swap in a more scalable integrator.
        """
        if psi0.shape[0] != self.dim_total:
            raise ValueError(f"psi0 dimension {psi0.shape[0]} does not match dim_total {self.dim_total}")
        U = expm(-1j * self.H_total * t)
        return U @ psi0

    def random_state(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a random normalized state in H_total.
        """
        if seed is not None:
            np.random.seed(seed)
        real = np.random.normal(size=self.dim_total)
        imag = np.random.normal(size=self.dim_total)
        psi = real + 1j * imag
        norm = np.linalg.norm(psi)
        if norm == 0:
            raise ValueError("Generated zero-norm random state.")
        return psi / norm

    def eigen_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dense eigendecomposition:

            H_total = V diag(E) V^†

        Returns:
            E (dim_total,), V (dim_total, dim_total)
        """
        E, V = eigh(self.H_total)
        return E, V

    # -------------------- measurement utilities ---------------------- #

    def layer_reduced_density(
        self,
        psi: np.ndarray,
        layer_index: int,
    ) -> np.ndarray:
        """
        Compute the reduced density matrix ρ^(ℓ) for a given layer ℓ,
        by tracing out all other layers.

        Implementation detail:
          - reshape |ψ⟩ into a tensor ψ[i0, i1, ..., i_{L-1}],
          - compute ρ = Tr_{others} |ψ⟩⟨ψ|.
        """
        if psi.shape[0] != self.dim_total:
            raise ValueError("psi has wrong dimension.")

        # reshape into multi-index tensor
        shape = self.layer_dims
        psi_tensor = psi.reshape(shape)
        rho_layer = None

        # Indices: (ℓ, others...) – we do partial trace by summing over others
        # ρ^(ℓ)_{a,b} = ∑_{rest} ψ[a,rest] ψ*[b,rest]
        dim_ell = self.layer_dims[layer_index]
        rho = np.zeros((dim_ell, dim_ell), dtype=np.complex128)

        # We do this the brute-force way; for small dims it's fine.
        # Index over all layer indices
        it = np.ndindex(*shape)
        for idx in it:
            a = idx[layer_index]
            amp = psi_tensor[idx]
            for jdx in it:
                if all(
                    (i == j or k == layer_index)
                    for (k, (i, j)) in enumerate(zip(idx, jdx))
                ):
                    # Only vary the traced-out indices identically
                    b = jdx[layer_index]
                    amp2 = psi_tensor[jdx]
                    rho[a, b] += amp * np.conjugate(amp2)

        return rho

    # ------------------------ structured info ------------------------ #

    def assumptions_dict(self) -> Dict[str, Any]:
        """ASSUMED: the layer specs and how they’re combined."""
        return {
            "layers": [asdict(layer) for layer in self.layers],
            "composition": "tensor_product",
            "dim_total": self.dim_total,
        }

    def compute_basic_metrics(self, psi0: Optional[np.ndarray] = None, t: float = 1.0) -> Dict[str, Any]:
        """
        Example diagnostics:
          - total energy ⟨ψ0|H|ψ0⟩,
          - purity and entropy of each layer’s reduced state at t and 2t.
        """
        if psi0 is None:
            psi0 = self.random_state(seed=1234)

        if psi0.shape[0] != self.dim_total:
            raise ValueError("psi0 has wrong dimension.")

        # initial energy
        E0 = np.real(np.vdot(psi0, self.H_total @ psi0))

        psi_t = self.evolve_state(psi0, t)
        psi_2t = self.evolve_state(psi0, 2 * t)

        layer_data: List[Dict[str, Any]] = []
        for ell in range(self.n_layers):
            rho_t = self.layer_reduced_density(psi_t, ell)
            rho_2t = self.layer_reduced_density(psi_2t, ell)

            def purity(rho):
                return float(np.real(np.trace(rho @ rho)))

            def von_neumann_entropy(rho):
                evals = np.linalg.eigvalsh(rho)
                evals = np.clip(evals, 0.0, 1.0)
                nz = evals[evals > 1e-12]
                return float(-np.sum(nz * np.log2(nz)))

            layer_data.append({
                "layer_index": ell,
                "name": self.layers[ell].name,
                "dim": self.layers[ell].dim,
                "purity_t": purity(rho_t),
                "purity_2t": purity(rho_2t),
                "entropy_t": von_neumann_entropy(rho_t),
                "entropy_2t": von_neumann_entropy(rho_2t),
            })

        return {
            "E0": float(E0),
            "t": float(t),
            "layers": layer_data,
        }

    def structured_output(self, psi0: Optional[np.ndarray] = None, t: float = 1.0) -> Dict[str, Any]:
        """
        High-level JSON-style output with:
          - assumptions (layers, dims),
          - metrics (energy, entropies, purities),
          - placeholder for higher-level emergent features.
        """
        metrics = self.compute_basic_metrics(psi0=psi0, t=t)

        return {
            "assumptions": self.assumptions_dict(),
            "metrics": metrics,
            "notes": {
                "philosophy": (
                    "This is a hierarchical Hilbert substrate: each layer is itself "
                    "a full Hilbert space with its own generator H^(ℓ). The global "
                    "system is the tensor product, i.e. 'substrate all the way down.'"
                ),
                "limitations": (
                    "Currently uses dense matrices and simple tensor-product structure. "
                    "No direct-sum branching yet, and no explicit emergent particle sectors."
                ),
            },
        }


# =============================================================================
# Convenience: a default 3-layer hierarchy (position ⊗ spin ⊗ micro)
# =============================================================================


def default_hierarchy() -> HierarchicalSubstrate:
    """
    Build a simple 3-layer hierarchy:
      - Layer 0: position ring with L sites
      - Layer 1: SU(2) spin (dim 2)
      - Layer 2: micro-substrate (dim 2)

    This is just an example of "every state carries its own substrate":
      - Each position site has an internal spin.
      - Each spin state has a 2-dim micro Hilbert space under it.
    """
    L = 8  # keep it small; dim_total = 8 * 2 * 2 = 32

    layers = [
        LayerSpec(
            name="position",
            dim=L,
            type="tensor",
            hamiltonian_builder=lambda spec: position_ring_builder(spec, hopping=1.0),
        ),
        LayerSpec(
            name="spin",
            dim=2,
            type="tensor",
            hamiltonian_builder=internal_spin_builder,
        ),
        LayerSpec(
            name="micro_substrate",
            dim=2,
            type="tensor",
            hamiltonian_builder=micro_substrate_builder,
        ),
    ]

    return HierarchicalSubstrate(layers)


# =============================================================================
# Main (test)
# =============================================================================


def quick_test() -> Dict[str, Any]:
    """
    Simple demo: build the default hierarchy, evolve a random state,
    and print some metrics.
    """
    print("=" * 72)
    print("HIERARCHICAL HILBERT SUBSTRATE - QUICK TEST")
    print("=" * 72)

    hs = default_hierarchy()
    out = hs.structured_output(t=1.0)

    assumptions = out["assumptions"]
    metrics = out["metrics"]

    print("\nASSUMPTIONS:")
    print(f"  Composition: {assumptions['composition']}")
    print(f"  dim_total :  {assumptions['dim_total']}")
    print("  Layers:")
    for layer in assumptions["layers"]:
        print(f"    - {layer['name']}: dim={layer['dim']}, type={layer['type']}")

    print("\nMETRICS:")
    print(f"  Initial energy E0: {metrics['E0']:.6f}")
    print(f"  Time t:            {metrics['t']:.3f}")
    print("  Layer reduced-state purities & entropies:")
    for ld in metrics["layers"]:
        print(f"    Layer {ld['layer_index']} ({ld['name']}), dim={ld['dim']}:")
        print(f"      purity(t):  {ld['purity_t']:.6f}")
        print(f"      purity(2t): {ld['purity_2t']:.6f}")
        print(f"      S_vN(t):    {ld['entropy_t']:.6f}")
        print(f"      S_vN(2t):   {ld['entropy_2t']:.6f}")

    print("\nNOTES:")
    print(" ", out["notes"]["philosophy"])
    print(" ", out["notes"]["limitations"])

    return out


if __name__ == "__main__":
    quick_test()
