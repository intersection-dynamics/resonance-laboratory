"""
Emergent Spacetime from No-Signaling + Lieb–Robinson
====================================================

Axioms:
  - Hilbert Space Realism
  - Unitary Evolution
  - Classical Emergence
Constraint:
  - No-signaling / approximate locality

This file does *two* related things:

  1. Propagation-based metric (your original PoC):
       - Build a many-body Hamiltonian on an abstract graph.
       - Locally excite a mode and see how ⟨n_j(t)⟩ spreads.
       - Define an operational "distance" from arrival times of excitation.

  2. Lieb–Robinson commutator metric (actual LR object):
       - Build local number operators n_i on the full Hilbert space.
       - Heisenberg-evolve A_i(t) = e^{iHt} A_i e^{-iHt}.
       - Compute C_ij(t) = [A_i(t), n_j] and its operator norm.
       - Extract arrival times from ||C_ij(t)|| crossing a threshold.
       - Use those times as an emergent LR distance.

All geometry is "operational": distance = how long it takes influences
(or commutators) to become appreciable.
"""

import numpy as np
from scipy.linalg import expm
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# =============================================================================
# Substrate: abstract modes, no assumed spatial embedding
# =============================================================================


class Substrate:
    """Quantum modes with no assumed spatial structure."""

    def __init__(self, n_modes: int, dim: int = 2):
        """
        n_modes: number of modes/sites
        dim    : local Hilbert-space dimension per mode (d=2 = qubit / 0-1 boson)
        """
        self.n_modes = n_modes
        self.d = dim
        self.dim_total = dim ** n_modes

    def vacuum(self) -> np.ndarray:
        """|0,0,...,0>."""
        psi = np.zeros(self.dim_total, dtype=complex)
        psi[0] = 1.0
        return psi

    def index_to_config(self, idx: int) -> Tuple[int, ...]:
        """Integer index → tuple of local occupation numbers (n0,...,n_{M-1})."""
        config = []
        for i in range(self.n_modes):
            config.append(idx // (self.d ** (self.n_modes - 1 - i)) % self.d)
        return tuple(config)

    def config_to_index(self, config: Tuple[int, ...]) -> int:
        """Tuple of local occupations → integer index."""
        idx = 0
        for i, n in enumerate(config):
            idx += n * (self.d ** (self.n_modes - 1 - i))
        return idx

    def excite(self, psi: np.ndarray, mode: int) -> np.ndarray:
        """
        Apply a_i^\dagger to mode, in the truncated bosonic sense:
          |... n_i ...> → sqrt(n_i+1) |... n_i+1 ...>
        """
        result = np.zeros_like(psi)
        for idx in range(self.dim_total):
            config = self.index_to_config(idx)
            if config[mode] < self.d - 1:
                new_config = list(config)
                new_config[mode] += 1
                new_idx = self.config_to_index(tuple(new_config))
                result[new_idx] += psi[idx] * np.sqrt(config[mode] + 1)
        return result

    def measure(self, psi: np.ndarray, mode: int) -> float:
        """
        Expectation value of local occupation number n_mode.
        """
        exp_val = 0.0
        for idx in range(self.dim_total):
            config = self.index_to_config(idx)
            exp_val += np.abs(psi[idx])**2 * config[mode]
        return exp_val

    # --- NEW: explicit local number operator matrices -------------------

    def build_number_operator(self, mode: int) -> np.ndarray:
        """
        Build the full many-body operator n_mode as a matrix in the global basis:
          n_mode |config> = n_mode(config) |config>.
        """
        dim = self.dim_total
        n_op = np.zeros((dim, dim), dtype=complex)
        for idx in range(dim):
            config = self.index_to_config(idx)
            occ = config[mode]
            n_op[idx, idx] = occ
        return n_op


# =============================================================================
# Hamiltonian construction on a graph
# =============================================================================


def build_hamiltonian(substrate: Substrate,
                      connectivity: Dict[int, List[int]],
                      coupling: float = 1.0) -> np.ndarray:
    """
    H = coupling * Σ_{connected (i,j)} (a†_i a_j + a†_j a_i).

    Implemented in the truncated boson Fock space.
    """
    dim = substrate.dim_total
    H = np.zeros((dim, dim), dtype=complex)

    for i, neighbors in connectivity.items():
        for j in neighbors:
            if j > i:
                for idx in range(dim):
                    config = list(substrate.index_to_config(idx))

                    # a_i^\dagger a_j
                    if config[j] > 0 and config[i] < substrate.d - 1:
                        coeff = np.sqrt(config[j]) * np.sqrt(config[i] + 1)
                        new_config = config.copy()
                        new_config[j] -= 1
                        new_config[i] += 1
                        new_idx = substrate.config_to_index(tuple(new_config))
                        H[new_idx, idx] += coupling * coeff

                    # a_j^\dagger a_i
                    if config[i] > 0 and config[j] < substrate.d - 1:
                        coeff = np.sqrt(config[i]) * np.sqrt(config[j] + 1)
                        new_config = config.copy()
                        new_config[i] -= 1
                        new_config[j] += 1
                        new_idx = substrate.config_to_index(tuple(new_config))
                        H[new_idx, idx] += coupling * coeff

    return H


def linear_chain(n: int) -> Dict[int, List[int]]:
    """Nearest-neighbor chain graph."""
    return {i: [j for j in [i-1, i+1] if 0 <= j < n] for i in range(n)}


def fully_connected(n: int) -> Dict[int, List[int]]:
    """All-to-all graph."""
    return {i: [j for j in range(n) if j != i] for i in range(n)}


# =============================================================================
# Propagation-based emergent metric (original PoC)
# =============================================================================


def propagate(substrate: Substrate, H: np.ndarray, source: int,
              t_max: float = 10.0, n_steps: int = 100) -> Dict:
    """
    Track information spread from a single initially excited mode via ⟨n_j(t)⟩.

    This is your original operational notion:
      - Start from vacuum
      - Excite mode `source`
      - Evolve under H
      - Record ⟨n_j(t)⟩ for all j
      - Define arrival time where ⟨n_j(t)⟩ crosses a threshold.
    """
    psi = substrate.excite(substrate.vacuum(), source)
    psi = psi / np.linalg.norm(psi)

    times = np.linspace(0, t_max, n_steps)
    dt = times[1] - times[0]
    U = expm(-1j * H * dt)

    history = np.zeros((substrate.n_modes, n_steps))

    for t_idx in range(n_steps):
        for mode in range(substrate.n_modes):
            history[mode, t_idx] = substrate.measure(psi, mode)
        psi = U @ psi

    threshold = 0.01
    arrival = {}
    for mode in range(substrate.n_modes):
        if mode == source:
            arrival[mode] = 0.0
        else:
            crossed = np.where(history[mode] > threshold)[0]
            arrival[mode] = times[crossed[0]] if len(crossed) > 0 else np.inf

    return {'times': times, 'history': history, 'arrival': arrival, 'source': source}


def compute_metric(substrate: Substrate, H: np.ndarray, t_max: float = 10.0) -> np.ndarray:
    """
    Emergent distance matrix from excitation arrival times (original scheme).
    """
    n = substrate.n_modes
    metric = np.zeros((n, n))

    for source in range(n):
        result = propagate(substrate, H, source, t_max)
        for target in range(n):
            metric[source, target] = result['arrival'][target]

    return (metric + metric.T) / 2.0


# =============================================================================
# Lieb–Robinson commutators: true LR object
# =============================================================================


def lieb_robinson_commutators(substrate: Substrate,
                              H: np.ndarray,
                              source: int,
                              t_max: float = 10.0,
                              n_steps: int = 100,
                              norm_type: str = "fro") -> Dict:
    """
    Compute LR commutators:
        C_ij(t) = [A_i(t), n_j]
      with A_i(0) = n_i, and A_i(t) in Heisenberg picture:
        A_i(t+dt) = U† A_i(t) U, with U = exp(-i H dt).

    Returns:
      - times:      array of times
      - comm_norms: shape (n_modes, n_steps), comm_norms[j,t] = ||[A_i(t), n_j]||
      - arrival:    dict j -> first time comm_norms[j,t] exceeds threshold
      - source:     source site i
    """
    n_modes = substrate.n_modes
    dim = substrate.dim_total

    times = np.linspace(0, t_max, n_steps)
    dt = times[1] - times[0]

    U = expm(-1j * H * dt)
    U_dag = U.conj().T

    # Local operators n_j
    n_ops = [substrate.build_number_operator(j) for j in range(n_modes)]

    # A_i(0) = n_i for LR probe (you can swap this for other local ops)
    A_t = n_ops[source].copy()

    comm_norms = np.zeros((n_modes, n_steps))

    for t_idx in range(n_steps):
        # For each target j, compute commutator [A_i(t), n_j] and its norm
        for j in range(n_modes):
            C = A_t @ n_ops[j] - n_ops[j] @ A_t
            if norm_type == "spectral":
                # operator (spectral) norm
                comm_norms[j, t_idx] = np.linalg.norm(C, 2)
            else:
                # Frobenius norm (Hilbert–Schmidt) – cheaper, still valid
                comm_norms[j, t_idx] = np.linalg.norm(C, "fro")

        # Heisenberg update for next step: A(t+dt) = U† A(t) U
        A_t = U_dag @ A_t @ U

    # Extract arrival times from commutator norms
    threshold = 1e-3  # small but nonzero; you can tune this
    arrival = {}
    for j in range(n_modes):
        if j == source:
            arrival[j] = 0.0
        else:
            crossed = np.where(comm_norms[j, :] > threshold)[0]
            arrival[j] = times[crossed[0]] if len(crossed) > 0 else np.inf

    return {
        "times": times,
        "comm_norms": comm_norms,
        "arrival": arrival,
        "source": source,
        "norm_type": norm_type,
    }


def compute_lr_metric(substrate: Substrate,
                      H: np.ndarray,
                      t_max: float = 10.0,
                      n_steps: int = 100,
                      norm_type: str = "fro") -> np.ndarray:
    """
    Build an emergent distance matrix from **Lieb–Robinson commutator**
    arrival times, not from excitation proxies.

    d_ij = (arrival time from i to j based on ||[A_i(t), n_j]|| crossing threshold),
    symmetrized over i and j.
    """
    n = substrate.n_modes
    dist = np.zeros((n, n))

    for source in range(n):
        lr = lieb_robinson_commutators(substrate, H, source,
                                       t_max=t_max, n_steps=n_steps,
                                       norm_type=norm_type)
        for target in range(n):
            dist[source, target] = lr["arrival"][target]

    return (dist + dist.T) / 2.0


# =============================================================================
# Plotting
# =============================================================================


def plot_light_cone(result: Dict, title: str, filename: str):
    """Visualize light cone of ⟨n_j(t)⟩ (original propagation)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    history = result['history']
    times = result['times']
    n_modes = history.shape[0]
    source = result['source']

    im = ax.imshow(history, aspect='auto', origin='lower',
                   extent=[times[0], times[-1], -0.5, n_modes-0.5],
                   cmap='inferno', vmin=0, vmax=1)

    for mode, t_arr in result['arrival'].items():
        if 0 < t_arr < np.inf:
            ax.plot(t_arr, mode, 'w^', markersize=8)

    ax.axhline(source, color='cyan', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Mode', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yticks(range(n_modes))

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Excitation ⟨n_j(t)⟩', fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_lr_light_cone(lr_result: Dict, filename: str):
    """Visualize Lieb–Robinson commutator norms as a light cone."""
    fig, ax = plt.subplots(figsize=(10, 6))

    comm_norms = lr_result["comm_norms"]
    times = lr_result["times"]
    n_modes = comm_norms.shape[0]
    source = lr_result["source"]

    im = ax.imshow(comm_norms, aspect='auto', origin='lower',
                   extent=[times[0], times[-1], -0.5, n_modes - 0.5],
                   cmap='viridis')

    # arrival markers
    for j, t_arr in lr_result["arrival"].items():
        if 0 < t_arr < np.inf:
            ax.plot(t_arr, j, 'w^', markersize=8)

    ax.axhline(source, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Mode j", fontsize=12)
    ax.set_title("Lieb–Robinson commutator cone", fontsize=14)
    ax.set_yticks(range(n_modes))

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\|[A_i(t), n_j]\|$", fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_metric(metric: np.ndarray, title: str, filename: str):
    """Visualize emergent distance matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))

    n = metric.shape[0]
    metric_vis = np.where(metric == np.inf, np.nan, metric)

    im = ax.imshow(metric_vis, cmap='viridis', origin='lower')

    ax.set_xlabel('Mode j', fontsize=12)
    ax.set_ylabel('Mode i', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    for i in range(n):
        for j in range(n):
            val = metric[i, j]
            if val < np.inf:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color='white' if val > np.nanmax(metric_vis) / 2 else 'black',
                        fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Distance (arrival time)', fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_comparison(result_local: Dict, result_full: Dict, filename: str):
    """Compare local vs full connectivity (⟨n_j(t)⟩ light cones)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, result, title in zip(
        axes,
        [result_local, result_full],
        ['Local Connectivity', 'Full Connectivity'],
    ):
        history = result['history']
        times = result['times']
        n_modes = history.shape[0]

        im = ax.imshow(history, aspect='auto', origin='lower',
                       extent=[times[0], times[-1], -0.5, n_modes-0.5],
                       cmap='inferno', vmin=0, vmax=1)

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Mode', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_yticks(range(n_modes))
        plt.colorbar(im, ax=ax, label='Excitation')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_entanglement(correlations: np.ndarray, filename: str):
    """Visualize entanglement/correlation structure."""
    fig, ax = plt.subplots(figsize=(8, 7))

    n = correlations.shape[0]

    im = ax.imshow(correlations, cmap='RdBu', origin='lower', vmin=-0.5, vmax=0.5)

    ax.set_xlabel('Mode j', fontsize=12)
    ax.set_ylabel('Mode i', fontsize=12)
    ax.set_title('Correlation Structure', fontsize=14)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    for i in range(n):
        for j in range(n):
            val = correlations[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color='white' if abs(val) > 0.25 else 'black', fontsize=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# =============================================================================
# Main demo
# =============================================================================


def main():
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    print("Emergent Spacetime + Lieb–Robinson")
    print("=" * 40)

    n_modes = 8
    substrate = Substrate(n_modes, dim=2)
    source = n_modes // 2

    print(f"Modes: {n_modes}, Source: {source}")

    # Local vs full connectivity (propagation)
    H_local = build_hamiltonian(substrate, linear_chain(n_modes))
    H_full = build_hamiltonian(substrate, fully_connected(n_modes))

    result_local = propagate(substrate, H_local, source, t_max=8.0, n_steps=120)
    result_full = propagate(substrate, H_full, source, t_max=8.0, n_steps=120)

    # Propagation-based metric (original)
    metric_prop = compute_metric(substrate, H_local, t_max=10.0)

    # Lieb–Robinson commutators on the local chain
    lr_local = lieb_robinson_commutators(substrate, H_local, source,
                                         t_max=8.0, n_steps=120,
                                         norm_type="fro")

    metric_lr = compute_lr_metric(substrate, H_local, t_max=8.0, n_steps=120,
                                  norm_type="fro")

    # Entanglement structure (unchanged 4-mode demo)
    n_small = 4
    sub_small = Substrate(n_small, dim=2)
    psi = np.zeros(sub_small.dim_total, dtype=complex)
    psi[sub_small.config_to_index((0, 0, 0, 0))] = 0.5
    psi[sub_small.config_to_index((0, 0, 1, 1))] = 0.5
    psi[sub_small.config_to_index((1, 1, 0, 0))] = 0.5
    psi[sub_small.config_to_index((1, 1, 1, 1))] = 0.5
    psi = psi / np.linalg.norm(psi)

    correlations = np.zeros((n_small, n_small))
    for i in range(n_small):
        for j in range(n_small):
            n_i = sub_small.measure(psi, i)
            n_j = sub_small.measure(psi, j)
            n_ij = 0.0
            for idx in range(sub_small.dim_total):
                conf = sub_small.index_to_config(idx)
                n_ij += np.abs(psi[idx])**2 * conf[i] * conf[j]
            correlations[i, j] = n_ij - n_i * n_j

    # Generate plots
    plot_light_cone(result_local, 'Light Cone (Local, ⟨n_j(t)⟩)', f'{output_dir}/light_cone_local.png')
    plot_light_cone(result_full, 'Light Cone (Full, ⟨n_j(t)⟩)', f'{output_dir}/light_cone_full.png')
    plot_comparison(result_local, result_full, f'{output_dir}/connectivity_comparison.png')
    plot_metric(metric_prop, 'Emergent Metric from ⟨n_j(t)⟩ arrival', f'{output_dir}/emergent_metric_propagation.png')

    # LR plots
    plot_lr_light_cone(lr_local, f'{output_dir}/lr_light_cone_local.png')
    plot_metric(metric_lr, 'Emergent Metric from LR commutators', f'{output_dir}/emergent_metric_lr.png')

    # Entanglement structure
    plot_entanglement(correlations, f'{output_dir}/entanglement_structure.png')

    print(f"\nPropagation-based arrival times (d from source, local chain):")
    for m in range(n_modes):
        print(f"  {m}: {result_local['arrival'][m]:.2f}")

    print(f"\nLR commutator arrival times (d_LR from source, local chain):")
    for m in range(n_modes):
        print(f"  {m}: {lr_local['arrival'][m]:.3f}")

    print(f"\nOutputs: {output_dir}/")


if __name__ == "__main__":
    main()
