import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# =============================================================================
# Configuration: The Scales of Reality
# =============================================================================
GRID_SIZE = 30
# Mass Ratios (approximate for visualization)
MASS_QUARK = 10.0
MASS_ELECTRON = 1.0

# Coupling Constants (The "Stiffness" of the Substrate for different fields)
# Strong Force (Color) is STIFF (Linear confinement)
G_STRONG = 2.0  
# EM Force (Charge) is LOOSE (1/r spread)
G_EM = 0.5      

print("--- Substrate: Emergent Hydrogen from Constituent Quarks ---")

# =============================================================================
# 1. The Particles (Fermions)
# =============================================================================
class Fermion:
    def __init__(self, name, mass, charge, color_vec, pos):
        self.name = name
        self.mass = mass
        self.charge = charge
        self.color = np.array(color_vec) # [Red, Green, Blue]
        self.pos = np.array(pos, dtype=float)
        
    def __repr__(self):
        return f"{self.name} at {self.pos.astype(int)}"

# Initialize the ingredients of Hydrogen
# Proton Constituents: uud
particles = [
    Fermion("Up_R",   MASS_QUARK,  2/3, [1, 0, 0], [14, 15]), # Red
    Fermion("Up_G",   MASS_QUARK,  2/3, [0, 1, 0], [16, 15]), # Green
    Fermion("Down_B", MASS_QUARK, -1/3, [0, 0, 1], [15, 16]), # Blue
    # The Electron
    Fermion("Electron", MASS_ELECTRON, -1.0, [0, 0, 0], [5, 5]) # Colorless
]

# =============================================================================
# 2. The Substrate Physics (Forces as Geometric Costs)
# =============================================================================

def get_total_energy(particle_list):
    """
    Calculates the Hamiltonian of the system based on Substrate Constraints.
    H = Kinetic + H_Color_String + H_EM_Field
    """
    energy = 0.0
    
    # --- A. Kinetic Energy (Uncertainty Pressure) ---
    # Modeled as confinement cost (1/d^2) would be better, but simplified here 
    # as just the cost of localization. We skip explicit kinetic term for 
    # the relaxation step, assuming the "Monte Carlo" moves simulate thermal fluctuations.
    
    # --- B. Strong Force (The Color Constraint) ---
    # The Substrate demands local Color Neutrality.
    # Any distance between colored quarks stretches a "Flux Tube".
    # Energy = Sum of string lengths connecting to the 'Center of Color'.
    
    # Find center of color (barycenter of quarks)
    quark_pos = [p.pos for p in particle_list if np.linalg.norm(p.color) > 0]
    if quark_pos:
        center = np.mean(quark_pos, axis=0)
        
        # Calculate String Tension
        # For SU(3), the strings form a Y-junction (Mercedes logo).
        # Energy is sum of distances to the center * G_STRONG.
        for p in particle_list:
            if np.linalg.norm(p.color) > 0:
                dist = np.linalg.norm(p.pos - center)
                # Linear Confinement Potential (V ~ r)
                # This is a direct consequence of constant flux tube cross-section.
                energy += G_STRONG * dist
    
    # --- C. Electro-Magnetic Force (Charge Constraint) ---
    # Charges relax the substrate via Poisson equation (1/r).
    # Pairwise Coulomb interactions.
    
    num = len(particle_list)
    for i in range(num):
        for j in range(i + 1, num):
            p1 = particle_list[i]
            p2 = particle_list[j]
            
            r_vec = p1.pos - p2.pos
            r = np.linalg.norm(r_vec) + 0.5 # Softening to prevent infinity
            
            # Coulomb Potential: V ~ q1*q2 / r
            # Like signs repel (+ energy), Opposites attract (- energy)
            energy += G_EM * (p1.charge * p2.charge) / r
            
    return energy

# =============================================================================
# 3. Evolution (Imaginary Time Relaxation)
# =============================================================================
# We assume the system settles into its Ground State.
# We use a Metropolis-Hastings annealer to find the geometry.

def run_relaxation(steps=2000):
    temperature = 1.0
    
    # History for plotting
    history = {p.name: [] for p in particles}
    
    print(f"Relaxing Substrate... (T_init={temperature})")
    
    current_energy = get_total_energy(particles)
    
    for step in range(steps):
        # Cool down
        temperature *= 0.995
        
        # Pick a particle
        p_idx = np.random.randint(len(particles))
        p = particles[p_idx]
        
        # Propose a move (Quantum Fluctuation)
        # Lighter particles (Electron) have larger De Broglie wavelength (fluctuate more)
        step_size = 2.0 / np.sqrt(p.mass) 
        move = np.random.normal(0, step_size, 2)
        
        old_pos = p.pos.copy()
        p.pos += move
        
        # Boundary conditions (Box)
        p.pos = np.clip(p.pos, 0, GRID_SIZE)
        
        # Calculate Delta E
        new_energy = get_total_energy(particles)
        dE = new_energy - current_energy
        
        # Metropolis Acceptance
        if dE < 0 or np.random.rand() < np.exp(-dE / max(temperature, 1e-5)):
            current_energy = new_energy
        else:
            p.pos = old_pos # Revert
            
        # Record
        if step % 10 == 0:
            for p_obj in particles:
                history[p_obj.name].append(p_obj.pos.copy())

    return history

history = run_relaxation()

# =============================================================================
# 4. Visualization
# =============================================================================
plt.figure(figsize=(10, 8))

# Extract final positions
final_quarks = []
final_electron = None

for p in particles:
    traj = np.array(history[p.name])
    
    # Color logic for plot
    c = 'k'
    lbl = p.name
    
    if "Red" in p.name: c = 'r'; final_quarks.append(traj[-1])
    if "Green" in p.name: c = 'g'; final_quarks.append(traj[-1])
    if "Blue" in p.name: c = 'b'; final_quarks.append(traj[-1])
    if "Electron" in p.name: 
        c = 'cyan'
        final_electron = traj
        
        # Plot Electron Cloud (Density of states over last 50% of time)
        # This represents the probability distribution (Wavefunction)
        cloud = traj[len(traj)//2:]
        H, xedges, yedges = np.histogram2d(cloud[:,0], cloud[:,1], 
                                         bins=GRID_SIZE, range=[[0, GRID_SIZE], [0, GRID_SIZE]])
        # Smooth it to look like a wavefunction
        H = gaussian_filter(H, sigma=1.0)
        plt.imshow(H.T, origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE], 
                   cmap='Blues', alpha=0.5, label='Electron Cloud')

    # Plot Quark Trajectories (showing the binding process)
    if "Electron" not in p.name:
        plt.plot(traj[:,0], traj[:,1], color=c, alpha=0.6, linewidth=1)
        plt.scatter(traj[-1,0], traj[-1,1], color=c, s=100, edgecolors='k', label=lbl, zorder=10)

# Draw the Flux Tubes (The Gluon Field)
center = np.mean(final_quarks, axis=0)
for q_pos in final_quarks:
    plt.plot([center[0], q_pos[0]], [center[1], q_pos[1]], 'k--', linewidth=2, alpha=0.3)

plt.title("Emergent Hydrogen: Composite Proton & Orbital Electron")
plt.xlabel("Substrate X")
plt.ylabel("Substrate Y")

# Manual Legend for Clarity
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='cyan', lw=2),
                Line2D([0], [0], color='k', linestyle='--', lw=2)]
plt.legend(custom_lines, ['Up Quark (R)', 'Up Quark (G)', 'Down Quark (B)', 'Electron', 'Gluon Flux Tube'])

plt.savefig("substrate_proton_composite.png")
print("Simulated Composite Proton. Image saved.")