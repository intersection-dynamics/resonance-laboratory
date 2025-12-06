import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import itertools

# =============================================================================
# Configuration
# =============================================================================
N_SITES = 6             # Length of the 1D Universe
N_LINKS = N_SITES - 1
L_TRUNC = 1             # Truncation of Gauge Field (-1, 0, +1)
DIM_LINK = 2 * L_TRUNC + 1
FORCE = 0.5             # The "Tilt" (Potential Gradient)

# Basis Generation
# State: (pos, (l_0, l_1...))
states = []
gauge_configs = list(itertools.product(range(-L_TRUNC, L_TRUNC+1), repeat=N_LINKS))
for pos in range(N_SITES):
    for g in gauge_configs:
        states.append((pos, g))

DIM = len(states)
state_map = {s: i for i, s in enumerate(states)}
print(f"Hilbert Space Dimension: {DIM}")

# =============================================================================
# Physics Engine: The Hamiltonian
# =============================================================================
def build_hamiltonian(stiffness_g):
    """
    Builds H for a specific vacuum stiffness.
    H = H_kinetic + H_gauge_strain + H_force
    """
    H = np.zeros((DIM, DIM), dtype=np.complex128)
    
    for idx, (pos, links) in enumerate(states):
        # 1. H_gauge_strain (Potential Energy of the Geometry)
        # The cost to twist the vacuum.
        # E = 0.5 * g * sum(l^2)
        strain_energy = 0.5 * stiffness_g * sum([l**2 for l in links])
        H[idx, idx] += strain_energy
        
        # 2. H_force (External Potential / "Gravity")
        # V = -F * x
        # This pulls the particle to the right.
        potential_energy = -FORCE * pos
        H[idx, idx] += potential_energy
        
        # 3. H_kinetic (Gauged Hopping)
        # Particle moves, but must twist the link it crosses.
        
        # Hop Right (pos -> pos+1)
        if pos < N_SITES - 1:
            link_id = pos
            current_l = links[link_id]
            # Peierls/Gauge Rule: Moving Right decreases link value (e^-iA)
            target_l = current_l - 1
            
            if abs(target_l) <= L_TRUNC:
                new_links = list(links)
                new_links[link_id] = target_l
                new_state = (pos+1, tuple(new_links))
                
                if new_state in state_map:
                    j = state_map[new_state]
                    H[j, idx] -= 1.0 # Hopping Amp t=1
                    H[idx, j] -= 1.0 # Hermitian
                    
    return H

# =============================================================================
# The Inertia Test
# =============================================================================
def run_test(stiffness_g):
    print(f"Simulating Vacuum Stiffness g = {stiffness_g}...")
    H = build_hamiltonian(stiffness_g)
    
    # Start at x=0, Gauge=0
    psi = np.zeros(DIM, dtype=np.complex128)
    start_links = tuple([0]*N_LINKS)
    start_idx = state_map[(0, start_links)]
    psi[start_idx] = 1.0
    
    # Time Evolution
    times = np.linspace(0, 4, 40)
    evals, evecs = eigh(H)
    coeffs = evecs.conj().T @ psi
    
    positions = []
    
    for t in times:
        psi_t = evecs @ (coeffs * np.exp(-1j * evals * t))
        prob = np.abs(psi_t)**2
        
        # Expectation value <X>
        avg_x = 0.0
        for i, p in enumerate(prob):
            s_pos = states[i][0]
            avg_x += p * s_pos
        positions.append(avg_x)
        
    return times, positions

# =============================================================================
# Execution & Visualization
# =============================================================================
g_values = [0.0, 2.0, 10.0]
results = {}

for g in g_values:
    results[g] = run_test(g)

plt.figure(figsize=(10, 6))

for g in g_values:
    t, x = results[g]
    # Simple finite difference for velocity to check inertia visually
    # But plotting position is enough: Curvature = Acceleration.
    plt.plot(t, x, linewidth=2.5, label=f"Stiffness g={g}")

plt.title(f"Deriving Inertia: F=ma from Gauge Constraints (Force={FORCE})")
plt.xlabel("Time")
plt.ylabel("Particle Position <X>")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("inertia_derivation.png")
print("Derivation complete. Plot saved.")