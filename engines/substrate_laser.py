import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================
L_SIZE = 9             # Grid Size (Odd for center)
CENTER = (L_SIZE//2, L_SIZE//2, L_SIZE//2)

# Physics
HOPPING = 1.0          # Kinetic Energy
MONOPOLE_STRENGTH = 2.0 # Stronger binding to ensure discrete spectrum
LASER_STRENGTH = 0.5   # Amplitude of the electric field

print(f"--- Substrate: Laser Spectroscopy Experiment (L={L_SIZE}^3) ---")

# =============================================================================
# 1. Build the Topological Atom (Hamiltonian H0)
# =============================================================================
# We reuse the Monopole logic from before
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
Id = np.eye(2, dtype=np.complex128)

def get_hopping_matrix(r_vec, direction_vec):
    rx, ry, rz = r_vec[0]-CENTER[0], r_vec[1]-CENTER[1], r_vec[2]-CENTER[2]
    dist = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-6
    n_i = np.array([rx, ry, rz]) / dist
    dest = np.array([rx, ry, rz]) + direction_vec
    dist_dest = np.linalg.norm(dest) + 1e-6
    n_j = dest / dist_dest
    
    k = np.cross(n_i, n_j)
    sin_theta = np.linalg.norm(k)
    cos_theta = np.dot(n_i, n_j)
    
    if sin_theta < 1e-9: return Id
    k_hat = k / sin_theta
    theta = np.arccos(np.clip(cos_theta, -1, 1)) * MONOPOLE_STRENGTH
    rot_gen = k_hat[0]*sigma_x + k_hat[1]*sigma_y + k_hat[2]*sigma_z
    return np.cos(theta/2)*Id - 1j*np.sin(theta/2)*rot_gen

N_SITES = L_SIZE**3
DIM = N_SITES * 2
site_to_idx = lambda x,y,z: (x*L_SIZE**2 + y*L_SIZE + z) * 2

print("Constructing H0 (Static Atom)...")
H0 = sp.lil_matrix((DIM, DIM), dtype=np.complex128)

for x in range(L_SIZE):
    for y in range(L_SIZE):
        for z in range(L_SIZE):
            idx = site_to_idx(x,y,z)
            r_vec = np.array([x,y,z])
            for dim_i, (dx, dy, dz) in enumerate([(1,0,0), (0,1,0), (0,0,1)]):
                nx, ny, nz = x+dx, y+dy, z+dz
                if 0 <= nx < L_SIZE and 0 <= ny < L_SIZE and 0 <= nz < L_SIZE:
                    n_idx = site_to_idx(nx, ny, nz)
                    U = get_hopping_matrix(r_vec, np.array([dx, dy, dz]))
                    for s1 in range(2):
                        for s2 in range(2):
                            val = -HOPPING * U[s1, s2]
                            H0[idx+s1, n_idx+s2] += val
                            H0[n_idx+s2, idx+s1] += np.conj(val)
H0 = H0.tocsr()

# =============================================================================
# 2. Find Resonance Frequency
# =============================================================================
print("Calculating Spectrum...")
vals, vecs = eigsh(H0, k=4, which='SA')

E_ground = vals[0]
E_excited = vals[1] # Assuming this is the dipole allowed transition (p-orbital)
Delta_E = E_excited - E_ground
w_laser = Delta_E

print(f"  Ground State E0: {E_ground:.4f}")
print(f"  Excited State E1: {E_excited:.4f}")
print(f"  Resonance Frequency (w): {w_laser:.4f}")

# =============================================================================
# 3. Time Evolution with Laser
# =============================================================================
print("Firing Laser Pulse...")

# Dipole Operator (Field in X-direction)
# V_laser(t) = - E_field * x * sin(w*t)
X_op = np.zeros(DIM)
for x in range(L_SIZE):
    for y in range(L_SIZE):
        for z in range(L_SIZE):
            idx = site_to_idx(x,y,z)
            # Center x around 0
            x_val = x - CENTER[0]
            X_op[idx] = x_val
            X_op[idx+1] = x_val # Same for both spins

psi = vecs[:, 0].astype(np.complex128) # Start in Ground State

# Time stepping
dt = 0.2
t_max = 4 * np.pi / (LASER_STRENGTH) # Estimate Rabi period
steps = int(t_max / dt)
times = np.linspace(0, t_max, steps)

dipole_moment = []
snapshots = []
snapshot_indices = [0, steps//4, steps//2, 3*steps//4, steps-1]

# We need a dense matrix for small-step expm, or iterative approach
# For L=9, DIM ~ 1400. Dense expm is okay-ish (O(N^3)). 
# Optimization: H0 is sparse. H_total is H0 + V(t). 
# We'll use a simple split-operator or just dense expm for this demo size.
# Actually, let's use Crank-Nicolson for stability and speed with sparse matrices.
# (I - i*H*dt/2) psi(t+1) = (I + i*H*dt/2) psi(t)
# psi(t+1) = (I - i*H*dt/2)^-1 @ (I + i*H*dt/2) @ psi(t)

from scipy.sparse.linalg import factorized
# Pre-factorize H0 parts? No, H is time dependent.
# We will just use the slow but accurate dense method for this visualization proof.

for t_idx, t in enumerate(times):
    # Construct V(t)
    field_amp = LASER_STRENGTH * np.sin(w_laser * t)
    V_diag = -field_amp * X_op
    
    # Total H (approximate constant over dt)
    H_total = H0 + sp.diags(V_diag)
    
    # Evolution Step (Chebyshev or expm)
    # Since we can't do dense expm fast enough for 200 steps on CPU maybe, 
    # let's try a first-order approximation: psi += -1j * H * psi * dt
    # No, that's unstable.
    # Let's use scipy.sparse.linalg.expm_multiply which uses Krylov subspace.
    psi = sp.linalg.expm_multiply(-1j * H_total * dt, psi)
    
    # Measure Dipole <x>
    dens = np.abs(psi)**2
    d = np.sum(dens * X_op)
    dipole_moment.append(d)
    
    # Save Snapshot
    if t_idx in snapshot_indices:
        # Reshape to 3D density
        prob_spatial = dens[0::2] + dens[1::2]
        grid = prob_spatial.reshape((L_SIZE, L_SIZE, L_SIZE))
        snapshots.append(grid[:, :, CENTER[2]]) # XY slice

# =============================================================================
# 4. Visualization
# =============================================================================
fig = plt.figure(figsize=(12, 8))

# Plot 1: Dipole Oscillation (The "Dance" Trace)
ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=5)
ax1.plot(times, dipole_moment, 'r-', linewidth=2)
ax1.set_title(f"Electron Dipole Response (Rabi Oscillations) @ w={w_laser:.3f}")
ax1.set_xlabel("Time")
ax1.set_ylabel("Dipole Moment <x>")
ax1.grid(True, alpha=0.3)

# Plot 2: The Film Strip (Density Snapshots)
for i, snap in enumerate(snapshots):
    ax = plt.subplot2grid((2, 5), (1, i))
    ax.imshow(snap, cmap='inferno', origin='lower', vmin=0, vmax=np.max(snapshots))
    ax.set_title(f"t={times[snapshot_indices[i]]:.1f}")
    ax.axis('off')

plt.tight_layout()
plt.savefig("laser_dance.png")
print("Laser experiment complete. Check 'laser_dance.png'.")