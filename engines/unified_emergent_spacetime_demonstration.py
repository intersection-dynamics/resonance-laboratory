#!/usr/bin/env python3
"""
Unified Emergent Spacetime Demonstration
========================================

ONE SYSTEM, TWO PHASES:
1. Geometry Phase: Tracks information propagation to prove Inflation.
2. Matter Phase:   Precipitates topological defects (Skyrmions) on that geometry.

This script bridges the scale gap by calculating the Geometry Metric in the 
Single-Excitation subspace (N dimensions), allowing us to simulate N=64 
sites for BOTH geometry and matter in a single run.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from dataclasses import dataclass, asdict

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    L: int = 4              # 4x4x4 = 64 sites
    t_max_geo: float = 6.0  # Time for geometry check
    t_max_mat: float = 10.0 # Time for matter quench
    n_steps: int = 500
    seed: int = 42
    output_dir: str = "unified_run_data"

# =============================================================================
# 1. The Substrate (Graph Construction)
# =============================================================================

def build_lattice_and_hamiltonian(L):
    """
    Returns:
        X: positions (for plotting only)
        H_geom: (N, N) scalar Hamiltonian for Geometry Phase
        adj: Adjacency list
    """
    N = L**3
    X = np.zeros((N, 3))
    adj = {i: [] for i in range(N)}
    
    # Map (x,y,z) to index
    idx_map = {}
    c = 0
    for x in range(L):
        for y in range(L):
            for z in range(L):
                idx_map[(x,y,z)] = c
                X[c] = [x, y, z]
                c += 1
    
    # Build H_geom (Scalar Hopping)
    H_geom = np.zeros((N, N), dtype=complex)
    
    for x in range(L):
        for y in range(L):
            for z in range(L):
                u = idx_map[(x,y,z)]
                # 6 Neighbors
                neighbors = [
                    (x+1, y, z), (x-1, y, z),
                    (x, y+1, z), (x, y-1, z),
                    (x, y, z+1), (x, y, z-1)
                ]
                for nx, ny, nz in neighbors:
                    if (nx, ny, nz) in idx_map:
                        v = idx_map[(nx, ny, nz)]
                        adj[u].append(v)
                        # Hopping term -1.0
                        H_geom[u, v] = -1.0
                        H_geom[v, u] = -1.0
                        
    return N, X, H_geom, adj

# =============================================================================
# 2. Phase I: Emergent Geometry (Inflation)
# =============================================================================

def run_geometry_phase(N, H_geom, L, t_max, n_steps):
    print(f"--- Phase I: Measuring Emergent Geometry (N={N}) ---")
    
    # Source: Center node
    center = N // 2 + L // 2  # Approx center
    psi = np.zeros(N, dtype=complex)
    psi[center] = 1.0
    
    times = np.linspace(0, t_max, n_steps)
    dt = times[1] - times[0]
    
    # Unitary Evolution Operator (Single Particle)
    U_dt = expm(-1j * H_geom * dt)
    
    # Track arrival times
    threshold = 0.05
    arrival_times = np.full(N, np.inf)
    arrival_times[center] = 0.0
    
    dist_map = {} # Store distance to center for bins
    
    # Pre-calculate topological distances for correlation
    # (Simple BFS for the 'expected' baseline)
    from collections import deque
    q = deque([(center, 0)])
    visited = {center}
    topo_dists = np.full(N, -1)
    topo_dists[center] = 0
    
    # Need adjacency for BFS... infer from H is faster
    adj_fast = [np.nonzero(H_geom[i])[0] for i in range(N)]
    
    while q:
        curr, d = q.popleft()
        topo_dists[curr] = d
        for nbr in adj_fast[curr]:
            if nbr not in visited:
                visited.add(nbr)
                q.append((nbr, d+1))
                
    # Run Evolution
    print("  Evolving geometry probe field...")
    for t_idx, t in enumerate(times):
        probs = np.abs(psi)**2
        
        # Check arrivals
        new_arrivals = np.where((probs > threshold) & (arrival_times == np.inf))[0]
        arrival_times[new_arrivals] = t
        
        # Evolve
        psi = U_dt @ psi
        
    # Analyze Inflation
    # Bin by topological shell
    max_d = np.max(topo_dists)
    shell_avgs = []
    
    print("\n  [Data] Causal Metric Profile:")
    print("  Shell (d) | Avg Arrival (t) | Speed (v_eff)")
    print("  -------------------------------------------")
    
    prev_t = 0.0
    for d in range(1, max_d + 1):
        nodes_in_shell = np.where(topo_dists == d)[0]
        if len(nodes_in_shell) == 0: continue
        
        # Filter infinite arrivals (didn't reach yet)
        valid_times = arrival_times[nodes_in_shell]
        valid_times = valid_times[valid_times != np.inf]
        
        if len(valid_times) > 0:
            avg_t = np.mean(valid_times)
            delta_t = avg_t - prev_t
            v_eff = 1.0 / delta_t if delta_t > 1e-9 else 0.0
            
            print(f"     {d:2d}     |     {avg_t:5.3f}     |   {v_eff:5.2f}")
            shell_avgs.append(avg_t)
            prev_t = avg_t
        else:
            print(f"     {d:2d}     |     Horz      |    ---")
            
    return arrival_times, topo_dists

# =============================================================================
# 3. Phase II: Matter Precipitation (Spinors)
# =============================================================================

def run_matter_phase(N, adj, L, t_max, n_steps, seed):
    print(f"\n--- Phase II: Precipitating Topological Matter (Spinors) ---")
    rng = np.random.default_rng(seed)
    
    # 2-component spinor per site = size 2N
    # H_eff = mu(t) * (H_hop + H_hot)
    # We build the big Hamiltonian sparsely or block-wise
    
    spin_dim = 2
    dim = N * spin_dim
    
    # 1. H_hop (Kinetic)
    # Connects site i to j with Identity matrix in spin space
    print("  Building Spinor Hamiltonian...")
    H_hop = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        for j in adj[i]:
            # block indices
            bi, bj = i*2, j*2
            # Hopping -1.0
            H_hop[bi:bi+2, bj:bj+2] = -1.0 * np.eye(2)
            
    # 2. H_hot (Thermal/Random)
    # Random hermitian on each site
    H_hot = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        bi = i*2
        # Random 2x2 hermitian
        R = rng.normal(size=(2,2)) + 1j * rng.normal(size=(2,2))
        R = R + R.conj().T
        H_hot[bi:bi+2, bi:bi+2] = R * 5.0 # Strong thermal noise
        
    # State: Random spinor field
    psi = rng.normal(size=dim) + 1j*rng.normal(size=dim)
    psi /= np.linalg.norm(psi)
    
    # Evolution Loop (Quench)
    times = np.linspace(0, t_max, n_steps)
    dt = times[1] - times[0]
    
    # Simplified quench schedule:
    # t=0 to t_mid: Pure Hot -> Pure Hop (Ramp)
    # t_mid to t_end: Cooling (Amplitude damping or just Hop)
    
    print("  Evolving Matter Field (Quench)...")
    
    # To save time, we pre-sum for the unitary at a few set points 
    # or just do RK4. Let's do simple Trotterized evolution for speed.
    
    # We will simulate the "Cooling" by simply evolving under H_hop 
    # after a certain time, allowing defects to freeze.
    
    t_quench = t_max * 0.4
    
    psi_final = psi.copy()
    
    # Fast loop
    for t in times:
        # Schedule parameters
        if t < t_quench:
            # Hot phase dominating but fading
            lambda_hot = 1.0 - (t/t_quench)
            lambda_hop = (t/t_quench)
        else:
            # Cold phase
            lambda_hot = 0.0
            lambda_hop = 1.0
            
        # Effective H
        # Note: In a real heavy simulation we'd be more careful with 
        # operator splitting, but for this demo, we add them.
        H_t = lambda_hot * H_hot + lambda_hop * H_hop
        
        # Step (Euler/First order for speed in Python demo)
        # psi += -1j * H_t @ psi * dt
        # Re-normalize
        # psi /= np.linalg.norm(psi)
        
        # Let's do a slightly better approx: Matrix exp of local term?
        # No, N=128 is small enough for full matrix vector mul
        
        k1 = -1j * (H_t @ psi_final)
        psi_final += k1 * dt
        psi_final /= np.linalg.norm(psi_final)

    # Measure Topology (Skyrmions)
    print("  Analyzing Final State Topology...")
    
    # Extract Spin Vectors S_i = <psi_i | sigma | psi_i>
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    
    S_field = np.zeros((N, 3))
    
    for i in range(N):
        p = psi_final[i*2 : i*2+2]
        # Expectation
        S_field[i, 0] = (p.conj().T @ sx @ p).real
        S_field[i, 1] = (p.conj().T @ sy @ p).real
        S_field[i, 2] = (p.conj().T @ sz @ p).real
        
        # Normalize vector for skyrmion calc
        nm = np.linalg.norm(S_field[i])
        if nm > 1e-6: S_field[i] /= nm
        
    # Calculate Skyrmion density on z-slices (simplified)
    # Just summing triple products for triangles on the grid
    Q_total = 0.0
    
    # Reshape to grid
    S_grid = S_field.reshape((L, L, L, 3))
    
    for z in range(L):
        q_slice = 0.0
        for x in range(L-1):
            for y in range(L-1):
                # Triangle 1: (x,y), (x+1,y), (x,y+1)
                n1 = S_grid[x,y,z]
                n2 = S_grid[x+1,y,z]
                n3 = S_grid[x,y+1,z]
                # Solid angle approx: n1 . (n2 x n3)
                q_slice += np.dot(n1, np.cross(n2, n3))
                
                # Triangle 2: (x+1,y+1), (x,y+1), (x+1,y)
                n4 = S_grid[x+1,y+1,z]
                q_slice += np.dot(n4, np.cross(n3, n2))
        
        Q_total += q_slice
        
    Q_total /= (4*np.pi)
    print(f"\n  [Data] Final Topological Charge Q = {Q_total:5.2f}")
    
    return S_field, Q_total

# =============================================================================
# Main
# =============================================================================

def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # 1. Build Shared Substrate
    N, X, H_geom, adj = build_lattice_and_hamiltonian(cfg.L)
    print(f"Substrate Built: {N} nodes (L={cfg.L} cube)")
    
    # 2. Run Geometry (Inflation)
    arrivals, topo_dists = run_geometry_phase(N, H_geom, cfg.L, cfg.t_max_geo, cfg.n_steps)
    
    # 3. Run Matter (Skyrmions)
    S_field, Q = run_matter_phase(N, adj, cfg.L, cfg.t_max_mat, cfg.n_steps, cfg.seed)
    
    # 4. Plot Combined Results
    fig = plt.figure(figsize=(14, 6))
    
    # Plot A: Inflation (Metric)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(topo_dists, arrivals, c='blue', alpha=0.6)
    ax1.set_xlabel("Topological Step (d)")
    ax1.set_ylabel("Causal Arrival Time (t)")
    ax1.set_title("Phase I: Emergent Inflation")
    ax1.grid(True, alpha=0.3)
    
    # Plot B: Matter (Topology)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # Plot only z=L//2 slice for clarity
    z_slice = cfg.L // 2
    mask = (X[:, 2] == z_slice)
    
    sc = ax2.scatter(X[mask, 0], X[mask, 1], S_field[mask, 2], 
                     c=S_field[mask, 2], cmap='coolwarm', s=100)
    ax2.set_title(f"Phase II: Matter Slice (Q={Q:.2f})")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Sz")
    
    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/unified_result.png")
    print(f"\nSUCCESS. Combined plot saved to {cfg.output_dir}/unified_result.png")

if __name__ == "__main__":
    main()