#!/usr/bin/env python3
"""
Unified Substrate: Dynamic Geometry & Emergent History
==========================================================================

A framework combining:
  1. Hilbert Space Realism (The Universal Wave Function)
  2. Unitary Evolution (No collapse, only rotation)
  3. Dynamic Gauge Fields (Links that store memory/history)

This script models a "Universe" where matter resides on sites and 
geometry/gauge fields reside on the edges. Motion requires interaction 
with the edges, creating a "wake" of history.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import sys

# =============================================================================
# Configuration & Constants
# =============================================================================

@dataclass
class UniverseConfig:
    n_sites: int = 4            # Number of spatial sites
    d_matter_local: int = 2     # Matter dimension (0=vac, 1=particle)
    l_trunc: int = 2            # Gauge truncation (allows states -2 to +2)
    
    # Physics Parameters
    t_hopping: float = 1.0      # Kinetic amplitude
    g_electric: float = 0.5     # "Stiffness" of the links (Energy cost of history)
    
    # Simulation
    t_max: float = 6.0
    dt: float = 0.1
    
    # Connectivity: simple chain for demonstration
    # (0)-(1)-(2)-(3) has links 0->1, 1->2, 2->3
    connectivity: str = "chain" 

# =============================================================================
# The Unified Hilbert Space
# =============================================================================

class UnifiedSubstrate:
    """
    The fundamental substrate containing both Matter (Sites) and Geometry (Links).
    
    State Vector Structure:
    |Psi> = Sum ( c_i * |Matter_i> (x) |Gauge_i> )
    """
    
    def __init__(self, cfg: UniverseConfig):
        self.cfg = cfg
        
        # 1. Define Graph & Links
        # Maps site_idx -> list of (neighbor_idx, link_id, direction)
        # direction: +1 if link is i->j, -1 if link is j->i
        self.adj, self.links = self._build_graph()
        self.n_links = len(self.links)
        
        # 2. Build Basis
        # We explicitly enumerate the full Hilbert space (careful, grows fast!)
        # Basis state: (matter_config_tuple, gauge_config_tuple)
        self.basis_states = self._enumerate_basis()
        self.dim = len(self.basis_states)
        self.state_map = {s: i for i, s in enumerate(self.basis_states)}
        
        print(f"Substrate Initialized.")
        print(f"  Sites: {cfg.n_sites}, Links: {self.n_links}")
        print(f"  Hilbert Space Dimension: {self.dim}")

    def _build_graph(self):
        """Construct the connectivity graph and identify unique links."""
        adj = {i: [] for i in range(self.cfg.n_sites)}
        links = [] # List of tuples (u, v) representing unique links
        
        # Simple Chain Generation
        if self.cfg.connectivity == "chain":
            for i in range(self.cfg.n_sites - 1):
                u, v = i, i+1
                link_id = len(links)
                links.append((u, v))
                adj[u].append((v, link_id, 1))  # 1 = Outgoing defined direction
                adj[v].append((u, link_id, -1)) # -1 = Incoming against definition
        
        return adj, links

    def _enumerate_basis(self):
        """
        Generate all valid basis states |n_1...n_N> |l_1...l_M>.
        Warning: No conservation symmetries enforced here (full space).
        For larger systems, we would use sparse/implicit addressing.
        """
        import itertools
        
        # Matter sector: tuples of (0, 1) for each site
        matter_states = list(itertools.product(
            range(self.cfg.d_matter_local), repeat=self.cfg.n_sites
        ))
        
        # Gauge sector: tuples of (-L, ..., +L) for each link
        gauge_vals = range(-self.cfg.l_trunc, self.cfg.l_trunc + 1)
        gauge_states = list(itertools.product(
            gauge_vals, repeat=self.n_links
        ))
        
        # Combined basis
        full_basis = []
        for m in matter_states:
            # OPTIONAL: Restrict to single-particle sector to save memory for demo
            if sum(m) == 1: 
                for g in gauge_states:
                    full_basis.append((m, g))
                    
        return full_basis

    # =========================================================================
    # Operators & Hamiltonian
    # =========================================================================

    def build_hamiltonian(self):
        """
        Constructs the Universal Hamiltonian H.
        H = H_electric + H_kinetic(gauged)
        """
        dim = self.dim
        # Use sparse matrix for efficiency
        H = sp.lil_matrix((dim, dim), dtype=np.complex128)
        
        # Precompute lookups
        L_max = self.cfg.l_trunc
        
        for idx, (m_conf, g_conf) in enumerate(self.basis_states):
            m_conf = list(m_conf)
            g_conf = list(g_conf)
            
            # --- 1. Electric Energy (Diagonal) ---
            # H_el = (g^2 / 2) * Sum( l_i^2 )
            # The "tension" of the substrate.
            energy = 0.0
            for l_val in g_conf:
                energy += (self.cfg.g_electric / 2.0) * (l_val**2)
            
            H[idx, idx] = energy
            
            # --- 2. Gauged Hopping (Off-Diagonal) ---
            # H_hop = -t * (a^dag_i U_ij a_j + h.c.)
            # U_ij acts on link by lowering angular momentum (l -> l-1)
            
            # Iterate over all sites to find particles that can hop
            for i in range(self.cfg.n_sites):
                if m_conf[i] > 0: # If there is a particle at i
                    # Try to hop to neighbors
                    for neighbor, link_id, direction in self.adj[i]:
                        j = neighbor
                        # Check target availability (hard-core bosons: max occupancy 1)
                        if m_conf[j] < self.cfg.d_matter_local - 1:
                            
                            # -- Logic for Gauge Interaction --
                            # If hopping i -> j (direction +1):
                            #   We are moving WITH the link definition.
                            #   Operator: U_ij = e^{-i theta}. Lowers l by 1.
                            # If hopping i -> j (direction -1):
                            #   We are moving AGAINST the link definition.
                            #   Operator: U_ji = U_ij^dag. Raises l by 1.
                            
                            current_l = g_conf[link_id]
                            target_l = current_l - direction # -1 if dir=1, +1 if dir=-1
                            
                            # Check truncation limits
                            if abs(target_l) <= L_max:
                                # Construct new state
                                new_m = m_conf.copy()
                                new_m[i] -= 1
                                new_m[j] += 1
                                
                                new_g = g_conf.copy()
                                new_g[link_id] = target_l
                                
                                new_state = (tuple(new_m), tuple(new_g))
                                
                                if new_state in self.state_map:
                                    target_idx = self.state_map[new_state]
                                    # Add hopping term
                                    H[target_idx, idx] -= self.cfg.t_hopping
        
        return H.tocsr()

    # =========================================================================
    # Dynamics & Observables
    # =========================================================================

    def get_observables(self, psi):
        """Calculates Density <n_i> and Gauge Excitation <l^2_k>."""
        prob = np.abs(psi)**2
        
        rho_matter = np.zeros(self.cfg.n_sites)
        rho_gauge = np.zeros(self.n_links)
        
        for idx, p in enumerate(prob):
            if p > 1e-10: # optimization
                m, g = self.basis_states[idx]
                
                # Matter density
                for site_i, occ in enumerate(m):
                    rho_matter[site_i] += p * occ
                
                # Gauge energy/excitation
                for link_k, l_val in enumerate(g):
                    rho_gauge[link_k] += p * (l_val**2)
                    
        return rho_matter, rho_gauge

# =============================================================================
# Demonstration Routine
# =============================================================================

def run_simulation():
    print("--- Substrate Framework: Unified Field Test ---")
    
    # 1. Setup Universe
    cfg = UniverseConfig(
        n_sites=5, 
        l_trunc=2, 
        t_hopping=1.0, 
        g_electric=0.2, # Low cost = strong memory/long wake
        t_max=8.0
    )
    
    universe = UnifiedSubstrate(cfg)
    H = universe.build_hamiltonian()
    
    # 2. Initial State
    # Particle at Site 0, Vacuum Gauge (all links 0)
    print("Initializing State: |1,0,0,0,0> x |0,0,0,0>")
    
    # Construct index for |1,0,0...> |0,0...>
    init_m = tuple([1] + [0]*(cfg.n_sites-1))
    init_g = tuple([0] * universe.n_links)
    
    try:
        start_idx = universe.state_map[(init_m, init_g)]
    except KeyError:
        print("Error: Initial state not found in basis (check particle number conservation).")
        return

    psi_0 = np.zeros(universe.dim, dtype=np.complex128)
    psi_0[start_idx] = 1.0
    
    # 3. Time Evolution (Step-by-step for animation data)
    times = np.arange(0, cfg.t_max, cfg.dt)
    n_steps = len(times)
    
    # Storage for history
    history_matter = np.zeros((n_steps, cfg.n_sites))
    history_gauge = np.zeros((n_steps, universe.n_links))
    
    print(f"Evolving for {cfg.t_max} time units...")
    
    # Using expm_multiply for efficient sparse evolution
    # Note: For very large steps or times, this can be slow. 
    # Here we do it stepwise to record observables.
    
    psi = psi_0
    step_propagator = sp.linalg.expm(-1j * H * cfg.dt)
    
    for t_i in range(n_steps):
        # Record Observables
        rho_m, rho_g = universe.get_observables(psi)
        history_matter[t_i] = rho_m
        history_gauge[t_i] = rho_g
        
        # Step
        psi = step_propagator @ psi
        
        # Normalize (numerical stability)
        norm = np.linalg.norm(psi)
        psi /= norm
        
    print("Simulation Complete.")
    
    # 4. Visualization
    plot_results(times, history_matter, history_gauge, cfg)

def plot_results(times, mat, gauge, cfg):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Matter Density
    ax1.imshow(mat.T, aspect='auto', origin='lower', cmap='inferno',
               extent=[times[0], times[-1], -0.5, cfg.n_sites-0.5])
    ax1.set_ylabel("Site Index (Matter)")
    ax1.set_title(f"Matter Density Evolution (g={cfg.g_electric})")
    ax1.grid(False)
    
    # Gauge Excitation
    # We plot lines for the gauge links
    for i in range(gauge.shape[1]):
        ax2.plot(times, gauge[:, i], label=f'Link {i}-{i+1}', linewidth=2)
        
    ax2.set_ylabel("Gauge Excitation <L^2>")
    ax2.set_xlabel("Time")
    ax2.set_title("Emergent History (Gauge Wake)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()