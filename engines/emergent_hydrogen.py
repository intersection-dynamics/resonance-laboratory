#!/usr/bin/env python3
"""
EMERGENT HYDROGEN VIA POINTER STATE SELECTION (GPU/CuPy)

New diagnostics:
1. Initialize electron in HIGH modes (non-pointer) - watch migration to pointer states
2. Track coherence decay: pointer-pointer vs pointer-nonpointer modes

Strictly unitary.
"""

import cupy as cp
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
from substrate import Config, Substrate


@dataclass
class HydrogenConfig:
    n_nodes: int = 64
    internal_dim: int = 8
    n_proton_nodes: int = 3
    n_electron_nodes: int = 8
    proton_mass_factor: float = 100.0
    coupling_strength: float = 1.0
    dt: float = 0.02
    n_steps: int = 1500
    record_every: int = 10
    seed: int = 42
    
    # New: start electron in excited (high) modes
    init_in_high_modes: bool = True


def build_hydrogen_substrate(cfg: HydrogenConfig) -> Tuple[Substrate, Dict[str, Any]]:
    sub_cfg = Config(
        n_nodes=cfg.n_nodes,
        internal_dim=cfg.internal_dim,
        monogamy_budget=cfg.coupling_strength,
        defrag_rate=0.0,
        dt=cfg.dt,
        seed=cfg.seed,
        connectivity=0.4,
        use_gpu=True,
    )
    sub = Substrate(sub_cfg)
    rng = cp.random.default_rng(cfg.seed)
    d = cfg.internal_dim

    # Proton = most connected nodes
    connections = np.array([len(sub._neighbors[i]) for i in range(cfg.n_nodes)])
    proton_nodes = np.argsort(connections)[::-1][:cfg.n_proton_nodes].tolist()

    # Electron = neighbors of proton
    electron_set = set()
    for p in proton_nodes:
        electron_set.update(sub._neighbors[p])
    electron_set -= set(proton_nodes)
    electron_nodes = list(electron_set)[:cfg.n_electron_nodes]

    if len(electron_nodes) < cfg.n_electron_nodes:
        others = [i for i in range(cfg.n_nodes) if i not in proton_nodes and i not in electron_nodes]
        electron_nodes.extend(others[:cfg.n_electron_nodes - len(electron_nodes)])

    atom_nodes = proton_nodes + electron_nodes
    environment_nodes = [i for i in range(cfg.n_nodes) if i not in atom_nodes]

    indices = {'proton': proton_nodes, 'electron': electron_nodes,
               'atom': atom_nodes, 'environment': environment_nodes}

    # Seed proton: localized in ground mode
    for p in proton_nodes:
        psi = cp.zeros(d, dtype=cp.complex128)
        psi[0] = 1.0
        sub.states[p] = psi

    # Seed electron: either high modes or uniform
    if cfg.init_in_high_modes:
        # Start in modes d-3, d-2, d-1 (the "excited" states)
        print("  Initializing electrons in HIGH modes (excited states)")
        for i, e in enumerate(electron_nodes):
            psi = cp.zeros(d, dtype=cp.complex128)
            # Put amplitude in top 3 modes
            for k in range(d-3, d):
                psi[k] = cp.exp(1j * rng.uniform(0, 2*cp.pi)) / cp.sqrt(3)
            sub.states[e] = psi
    else:
        # Uniform superposition
        for i, e in enumerate(electron_nodes):
            psi = cp.zeros(d, dtype=cp.complex128)
            for k in range(d):
                psi[k] = cp.exp(1j * rng.uniform(0, 2*cp.pi)) / cp.sqrt(d)
            sub.states[e] = psi

    # Environment: random states
    for env in environment_nodes:
        psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        sub.states[env] = psi / cp.linalg.norm(psi)

    # Couplings
    J = sub.couplings.copy()
    
    for i in proton_nodes:
        for j in proton_nodes:
            if i != j:
                J[i, j] *= cp.sqrt(cfg.proton_mass_factor)
                J[j, i] = cp.conj(J[i, j])

    for p in proton_nodes:
        for e in electron_nodes:
            dist = sub.graph_distance(p, e)
            if dist < float('inf'):
                coulomb = cfg.coupling_strength / max(dist, 0.5)
                J[p, e] = -coulomb * (1 + 0.3j)
                J[e, p] = cp.conj(J[p, e])

    sub.couplings = J
    sub._rebuild_neighbors_from_couplings()
    return sub, indices


def correlation_matrix(sub: Substrate) -> cp.ndarray:
    """C[i,j] = |⟨ψ_i|ψ_j⟩|² for all node pairs."""
    states = sub.states.copy()
    norms = cp.linalg.norm(states, axis=1, keepdims=True)
    norms = cp.maximum(norms, 1e-12)
    states_normed = states / norms
    inner = states_normed @ cp.conj(states_normed).T
    return cp.abs(inner) ** 2


def atom_environment_entanglement(sub: Substrate, indices: Dict) -> float:
    C = correlation_matrix(sub)
    atom, env = indices['atom'], indices['environment']
    total = sum(float(C[a, e].get()) for a in atom for e in env)
    return total / max(len(atom) * len(env), 1)


def electron_environment_entanglement(sub: Substrate, indices: Dict) -> float:
    C = correlation_matrix(sub)
    electrons, env = indices['electron'], indices['environment']
    total = sum(float(C[e, env_n].get()) for e in electrons for env_n in env)
    return total / max(len(electrons) * len(env), 1)


def proton_electron_entanglement(sub: Substrate, indices: Dict) -> float:
    C = correlation_matrix(sub)
    protons, electrons = indices['proton'], indices['electron']
    total = sum(float(C[p, e].get()) for p in protons for e in electrons)
    return total / max(len(protons) * len(electrons), 1)


def mode_environment_entanglement(sub: Substrate, indices: Dict) -> np.ndarray:
    """Entanglement of each internal mode with environment."""
    d = sub.states.shape[1]
    electrons = indices['electron']
    env = indices['environment']
    J = sub.couplings
    
    mode_ent = np.zeros(d)
    
    for k in range(d):
        coupling = 0.0
        for e in electrons:
            psi_e = sub.states[e] / cp.linalg.norm(sub.states[e])
            e_amp_k = psi_e[k]
            for env_node in env:
                J_e_env = cp.abs(J[e, env_node])
                env_state = sub.states[env_node] / cp.linalg.norm(sub.states[env_node])
                env_k = cp.abs(env_state[k])
                coupling += float((J_e_env * cp.abs(e_amp_k) * env_k).get())
        mode_ent[k] = coupling
    
    total = mode_ent.sum()
    if total > 1e-12:
        mode_ent /= total
    return mode_ent


def find_pointer_states(sub: Substrate, indices: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Find modes with least environment entanglement."""
    ent = mode_environment_entanglement(sub, indices)
    pointer_modes = np.argsort(ent)
    return pointer_modes, ent


def mode_populations(sub: Substrate, indices: Dict) -> np.ndarray:
    """Average occupation probability per internal mode across electrons."""
    d = sub.states.shape[1]
    pop = np.zeros(d)
    
    for e in indices['electron']:
        psi = sub.states[e] / cp.linalg.norm(sub.states[e])
        pop += cp.abs(psi).get() ** 2
    
    return pop / len(indices['electron'])


def mode_coherence_matrix(sub: Substrate, indices: Dict) -> np.ndarray:
    """
    Coherence between modes: C[j,k] = |ρ_jk| averaged over electrons.
    Off-diagonal elements measure superposition coherence.
    """
    d = sub.states.shape[1]
    coherence = np.zeros((d, d))
    
    for e in indices['electron']:
        psi = sub.states[e] / cp.linalg.norm(sub.states[e])
        # Single-site density matrix
        rho = cp.outer(psi, cp.conj(psi))
        coherence += cp.abs(rho).get()
    
    coherence /= len(indices['electron'])
    return coherence


def pointer_vs_nonpointer_coherence(sub: Substrate, indices: Dict, 
                                     pointer_modes: np.ndarray) -> Tuple[float, float]:
    """
    Compare coherence within pointer subspace vs between pointer and non-pointer.
    
    Returns:
        pointer_pointer_coherence: avg |ρ_jk| for j,k both in pointer set
        pointer_nonpointer_coherence: avg |ρ_jk| for j in pointer, k not in pointer
    """
    coh = mode_coherence_matrix(sub, indices)
    d = len(pointer_modes)
    
    # Pointer states = first 2 modes (least entangled)
    n_pointer = 2
    pointer_set = set(pointer_modes[:n_pointer])
    nonpointer_set = set(pointer_modes[n_pointer:])
    
    # Pointer-pointer coherence (off-diagonal within pointer subspace)
    pp_coh = []
    for j in pointer_set:
        for k in pointer_set:
            if j != k:
                pp_coh.append(coh[j, k])
    
    # Pointer-nonpointer coherence
    pnp_coh = []
    for j in pointer_set:
        for k in nonpointer_set:
            pnp_coh.append(coh[j, k])
    
    pp = np.mean(pp_coh) if pp_coh else 0.0
    pnp = np.mean(pnp_coh) if pnp_coh else 0.0
    
    return pp, pnp


def low_vs_high_mode_population(sub: Substrate, indices: Dict) -> Tuple[float, float]:
    """Population in low modes (0,1,2) vs high modes (5,6,7)."""
    pop = mode_populations(sub, indices)
    low = pop[:3].sum()
    high = pop[-3:].sum()
    return low, high


def radial_distribution(sub: Substrate, indices: Dict) -> np.ndarray:
    max_dist = 6
    prob = np.zeros(max_dist + 1)
    count = np.zeros(max_dist + 1)
    
    for e in indices['electron']:
        psi_e = sub.states[e]
        amp_sq = float(cp.sum(cp.abs(psi_e)**2).get())
        min_dist = min(sub.graph_distance(p, e) for p in indices['proton'])
        if min_dist <= max_dist:
            prob[int(min_dist)] += amp_sq
            count[int(min_dist)] += 1
    
    for d in range(max_dist + 1):
        if count[d] > 0:
            prob[d] /= count[d]
    return prob


def run_experiment(cfg: HydrogenConfig) -> Dict[str, Any]:
    print("=" * 70)
    print("EMERGENT HYDROGEN VIA POINTER STATE SELECTION")
    print("Tracking mode migration and coherence decay")
    print("=" * 70)
    print(f"Nodes: {cfg.n_nodes} | Proton: {cfg.n_proton_nodes} | Electron: {cfg.n_electron_nodes}")
    print(f"Internal dim: {cfg.internal_dim}")
    print(f"Steps: {cfg.n_steps} | dt: {cfg.dt}")

    sub, indices = build_hydrogen_substrate(cfg)
    print(f"Proton nodes: {indices['proton']}")
    print(f"Electron nodes: {indices['electron']}")

    # Storage
    times = []
    atom_env_ent = []
    elec_env_ent = []
    pe_ent = []
    low_pop = []
    high_pop = []
    pp_coherence = []
    pnp_coherence = []
    mode_pop_history = []
    mode_ent_history = []

    print("\nEvolving...")
    for step in range(cfg.n_steps + 1):
        if step % cfg.record_every == 0:
            t = step * cfg.dt
            times.append(t)
            
            # Entanglement
            atom_env_ent.append(atom_environment_entanglement(sub, indices))
            elec_env_ent.append(electron_environment_entanglement(sub, indices))
            pe_ent.append(proton_electron_entanglement(sub, indices))
            
            # Mode populations
            pop = mode_populations(sub, indices)
            mode_pop_history.append(pop)
            low, high = low_vs_high_mode_population(sub, indices)
            low_pop.append(low)
            high_pop.append(high)
            
            # Pointer analysis
            pointer_modes, mode_ent = find_pointer_states(sub, indices)
            mode_ent_history.append(mode_ent)
            
            # Coherence decay
            pp, pnp = pointer_vs_nonpointer_coherence(sub, indices, pointer_modes)
            pp_coherence.append(pp)
            pnp_coherence.append(pnp)

            if step % (cfg.record_every * 15) == 0:
                print(f"  t={t:.2f}: low={low:.3f}, high={high:.3f}, "
                      f"coh_pp={pp:.4f}, coh_pnp={pnp:.4f}")

        if step < cfg.n_steps:
            sub.evolve(n_steps=1)

    # Final analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    pointer_modes, final_mode_ent = find_pointer_states(sub, indices)
    final_pop = mode_populations(sub, indices)
    final_radial = radial_distribution(sub, indices)

    print(f"\nMode Migration (started in HIGH modes):")
    print(f"  Low modes (0,1,2):  {low_pop[0]:.3f} → {low_pop[-1]:.3f}")
    print(f"  High modes (5,6,7): {high_pop[0]:.3f} → {high_pop[-1]:.3f}")
    
    migration = low_pop[-1] - low_pop[0]
    if migration > 0.1:
        print(f"  ★★★ Strong migration toward low modes! (+{migration:.3f}) ★★★")
    elif migration > 0.05:
        print(f"  ★★ Moderate migration (+{migration:.3f}) ★★")
    elif migration > 0:
        print(f"  ★ Weak migration (+{migration:.3f}) ★")
    else:
        print(f"  No migration toward pointer states")

    print(f"\nCoherence Decay:")
    print(f"  Pointer-Pointer:     {pp_coherence[0]:.4f} → {pp_coherence[-1]:.4f}")
    print(f"  Pointer-NonPointer:  {pnp_coherence[0]:.4f} → {pnp_coherence[-1]:.4f}")
    
    pp_decay = pp_coherence[0] - pp_coherence[-1]
    pnp_decay = pnp_coherence[0] - pnp_coherence[-1]
    
    if pnp_decay > pp_decay + 0.01:
        print(f"  ★★★ Pointer-NonPointer decoheres faster! (Einselection working) ★★★")
    elif pnp_decay > pp_decay:
        print(f"  ★★ Slight differential decoherence ★★")
    else:
        print(f"  No differential decoherence observed")

    print(f"\nPointer States (least entangled modes): {pointer_modes[:3]}")
    print(f"Entanglement per mode: {np.round(final_mode_ent, 3)}")

    print(f"\nFinal Mode Populations:")
    for k in range(len(final_pop)):
        bar = "█" * int(final_pop[k] * 50)
        marker = " ← pointer" if k in pointer_modes[:2] else ""
        print(f"  mode {k}: {final_pop[k]:.3f} {bar}{marker}")

    print(f"\nRadial Distribution:")
    for r, p in enumerate(final_radial):
        if p > 0.01:
            bar = "█" * int(p * 40)
            print(f"  r={r}: {p:.3f} {bar}")

    return {
        'times': np.array(times),
        'atom_env_ent': np.array(atom_env_ent),
        'elec_env_ent': np.array(elec_env_ent),
        'pe_ent': np.array(pe_ent),
        'low_pop': np.array(low_pop),
        'high_pop': np.array(high_pop),
        'pp_coherence': np.array(pp_coherence),
        'pnp_coherence': np.array(pnp_coherence),
        'mode_pop_history': np.array(mode_pop_history),
        'mode_ent_history': np.array(mode_ent_history),
        'final_pointer_modes': pointer_modes,
        'final_pop': final_pop,
        'final_radial': final_radial,
        'indices': indices
    }


def plot_results(results: Dict):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available")
        return

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    t = results['times']

    # Mode population migration
    ax = axes[0, 0]
    ax.plot(t, results['low_pop'], 'b-', lw=2, label='Low modes (0,1,2)')
    ax.plot(t, results['high_pop'], 'r-', lw=2, label='High modes (5,6,7)')
    ax.axhline(0.375, color='gray', ls=':', alpha=0.5, label='Uniform (3/8)')
    ax.set_xlabel('Time'); ax.set_ylabel('Population')
    ax.set_title('Mode Migration')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Coherence decay
    ax = axes[0, 1]
    ax.plot(t, results['pp_coherence'], 'g-', lw=2, label='Pointer-Pointer')
    ax.plot(t, results['pnp_coherence'], 'm-', lw=2, label='Pointer-NonPointer')
    ax.set_xlabel('Time'); ax.set_ylabel('Coherence |ρ_jk|')
    ax.set_title('Coherence Decay')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Entanglement
    ax = axes[0, 2]
    ax.plot(t, results['atom_env_ent'], 'r-', lw=2, label='Atom-Env')
    ax.plot(t, results['elec_env_ent'], 'b-', lw=2, label='Elec-Env')
    ax.plot(t, results['pe_ent'], 'g-', lw=2, label='Proton-Elec')
    ax.set_xlabel('Time'); ax.set_ylabel('Correlation')
    ax.set_title('Entanglement')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Mode population heatmap
    ax = axes[0, 3]
    pop_hist = results['mode_pop_history']
    im = ax.imshow(pop_hist.T, aspect='auto', origin='lower',
                   extent=[t[0], t[-1], 0, pop_hist.shape[1]], cmap='viridis')
    ax.set_xlabel('Time'); ax.set_ylabel('Mode')
    ax.set_title('Mode Population Evolution')
    plt.colorbar(im, ax=ax)

    # Mode-environment entanglement heatmap
    ax = axes[1, 0]
    ent_hist = results['mode_ent_history']
    im = ax.imshow(ent_hist.T, aspect='auto', origin='lower',
                   extent=[t[0], t[-1], 0, ent_hist.shape[1]], cmap='hot')
    ax.set_xlabel('Time'); ax.set_ylabel('Mode')
    ax.set_title('Mode-Env Entanglement')
    plt.colorbar(im, ax=ax)

    # Final mode populations
    ax = axes[1, 1]
    pop = results['final_pop']
    pointer = results['final_pointer_modes']
    colors = ['green' if i in pointer[:2] else 'blue' for i in range(len(pop))]
    ax.bar(range(len(pop)), pop, color=colors, alpha=0.7)
    ax.set_xlabel('Mode'); ax.set_ylabel('Population')
    ax.set_title('Final Mode Population')
    ax.grid(True, alpha=0.3)

    # Radial distribution
    ax = axes[1, 2]
    radial = results['final_radial']
    ax.bar(range(len(radial)), radial, color='orange', alpha=0.7)
    ax.set_xlabel('Distance from proton'); ax.set_ylabel('Probability')
    ax.set_title('Radial Distribution')
    ax.grid(True, alpha=0.3)

    # Coherence ratio
    ax = axes[1, 3]
    ratio = results['pp_coherence'] / (results['pnp_coherence'] + 1e-10)
    ax.plot(t, ratio, 'k-', lw=2)
    ax.axhline(1.0, color='r', ls=':', alpha=0.5)
    ax.set_xlabel('Time'); ax.set_ylabel('PP / PNP Ratio')
    ax.set_title('Coherence Selectivity')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('emergent_hydrogen_gpu.png', dpi=150)
    print("\nSaved: emergent_hydrogen_gpu.png")
    plt.show()


if __name__ == "__main__":
    cfg = HydrogenConfig()
    results = run_experiment(cfg)
    plot_results(results)
    print("\nDone.")