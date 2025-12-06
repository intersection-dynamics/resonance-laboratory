import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpx_sp
import matplotlib.pyplot as plt
import time

# =============================================================================
# Configuration
# =============================================================================
N_SITES = 8 
N_LINKS = 12
DIM = N_SITES * (1 << N_LINKS) # 32,768 states

# Physics
HOPPING_AMP = 1.0
ELECTRIC_COST = 0.0  
DT = 0.1
T_STEPS = 100

print(f"--- GPU Substrate: 3D Cube ---")
print(f"Hilbert Space Dimension: {DIM}")

# =============================================================================
# 1. Vectorized Hamiltonian Construction (CPU -> GPU)
# =============================================================================
def build_hamiltonian_gpu():
    """
    Constructs the Hamiltonian using vectorized CPU operations, 
    then moves it to GPU memory (CSR format).
    """
    t0 = time.time()
    
    # --- A. Diagonal Terms (Electric Energy) ---
    # Create an array of all gauge configurations [0, ..., 4095]
    # We repeat this for each site.
    gauge_indices = np.arange(1 << N_LINKS)
    
    # Calculate energy (population count of bits)
    # This is a fast way to count set bits in numpy
    # (Bitwise AND with 1, shift, repeat) - or just use precomputed map for 4096
    # For speed/simplicity here, we rely on the fact that N_LINKS=12 is small.
    pop_counts = np.zeros(1 << N_LINKS, dtype=np.float32)
    for i in range(1 << N_LINKS):
        pop_counts[i] = bin(i).count('1')
        
    # The diagonal is purely electric energy, repeated for each site
    diag_values = ELECTRIC_COST * np.tile(pop_counts, N_SITES)
    
    # --- B. Off-Diagonal Terms (Hopping) ---
    # We define the Cube Connectivity manually as vectorized shifts
    # Coordinates: 4z + 2y + x
    rows = []
    cols = []
    data = []
    
    # Helper to generate transition arrays
    # For every state (site, g), we find (neighbor, g')
    all_states_g = np.tile(np.arange(1 << N_LINKS), N_SITES)
    all_states_site = np.repeat(np.arange(N_SITES), 1 << N_LINKS)
    current_row_indices = (all_states_site << N_LINKS) + all_states_g
    
    # Define the 12 Links (u, v) and their IDs
    # Manual map of (u, v) -> link_id for the cube
    # (u, v) pairs for x, y, z directions
    connectivity = [
        # X-direction hops (link_ids 0,1,2,3)
        (0,1,0), (2,3,1), (4,5,2), (6,7,3),
        # Y-direction hops (link_ids 4,5,6,7)
        (0,2,4), (1,3,5), (4,6,6), (5,7,7),
        # Z-direction hops (link_ids 8,9,10,11)
        (0,4,8), (1,5,9), (2,6,10), (3,7,11)
    ]
    
    for u, v, link_id in connectivity:
        # 1. Forward Hops (u -> v)
        # Select all indices where site == u
        mask_u = (all_states_site == u)
        indices_u = current_row_indices[mask_u]
        
        # New Site: v
        # New Gauge: g ^ (1 << link_id)
        # The row indices are `indices_u`.
        # The col indices are (v << 12) + (g ^ 2^link_id)
        
        # We can compute col indices directly from row indices:
        # Remove old site bits, add new site bits, flip gauge bit
        # current_g = row & 0xFFF
        # row - (u<<12) + (v<<12) ^ (1<<link_id)
        
        shift_val = (v - u) * (1 << N_LINKS) # integer math
        flip_mask = (1 << link_id)
        
        indices_v = indices_u + shift_val # Change site u->v
        indices_v = indices_v ^ flip_mask # Flip link bit
        
        # Append to sparse lists
        rows.append(indices_v) # Hamiltonian is H[target, source]
        cols.append(indices_u)
        data.append(np.full(len(indices_u), -HOPPING_AMP, dtype=np.complex64))
        
        # 2. Backward Hops (v -> u)
        # Symmetric operations
        mask_v = (all_states_site == v)
        indices_v_start = current_row_indices[mask_v]
        
        shift_val_back = (u - v) * (1 << N_LINKS)
        indices_u_target = indices_v_start + shift_val_back
        indices_u_target = indices_u_target ^ flip_mask
        
        rows.append(indices_u_target)
        cols.append(indices_v_start)
        data.append(np.full(len(indices_v_start), -HOPPING_AMP, dtype=np.complex64))

    # Concatenate lists
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    
    # Add Diagonal
    rows_diag = np.arange(DIM)
    cols_diag = np.arange(DIM)
    
    rows_total = np.concatenate([rows, rows_diag])
    cols_total = np.concatenate([cols, cols_diag])
    data_total = np.concatenate([data, diag_values])
    
    # Construct CSR Matrix on GPU
    print(f"  CPU Build Time: {time.time()-t0:.4f}s")
    print(f"  Transferring to GPU...")
    H_gpu = cpx_sp.coo_matrix(
        (cp.asarray(data_total), (cp.asarray(rows_total), cp.asarray(cols_total))),
        shape=(DIM, DIM)
    ).tocsr()
    
    return H_gpu

# =============================================================================
# 2. GPU Time Evolution (RK4)
# =============================================================================
def rk4_step(psi, H, dt):
    """
    Performs one RK4 step for the Schrödinger equation:
    d|psi>/dt = -i H |psi>
    """
    k1 = -1j * H.dot(psi)
    k2 = -1j * H.dot(psi + 0.5 * dt * k1)
    k3 = -1j * H.dot(psi + 0.5 * dt * k2)
    k4 = -1j * H.dot(psi + dt * k3)
    
    return psi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def run_simulation_gpu(H_gpu, flux_link=None):
    # Initial State: Site 0, Vacuum Gauge (or Flux)
    psi_gpu = cp.zeros(DIM, dtype=cp.complex64)
    
    start_g = 0
    if flux_link is not None:
        start_g = (1 << flux_link)
        
    start_idx = (0 << N_LINKS) + start_g
    psi_gpu[start_idx] = 1.0
    
    probs = []
    
    # Target indices (Site 7, all gauge configs)
    target_start = 7 << N_LINKS
    target_end = 8 << N_LINKS
    
    # Evolution Loop
    for _ in range(T_STEPS):
        # Measure probability at Site 7
        # Slice on GPU, compute norm, pull scalar to CPU
        p7 = cp.sum(cp.abs(psi_gpu[target_start:target_end])**2).item()
        probs.append(p7)
        
        # Step
        psi_gpu = rk4_step(psi_gpu, H_gpu, DT)
        
        # Optional: Re-normalize to prevent drift
        # psi_gpu /= cp.linalg.norm(psi_gpu)
        
    return probs

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    cp.cuda.Device(0).use()
    
    # Build H once
    H_gpu = build_hamiltonian_gpu()
    
    print("Running Vacuum Case...")
    t0 = time.time()
    probs_vac = run_simulation_gpu(H_gpu, flux_link=None)
    cp.cuda.Stream.null.synchronize() # Wait for GPU
    print(f"  Time: {time.time()-t0:.4f}s")
    
    print("Running Flux Case (Link 0 flipped)...")
    t0 = time.time()
    # Flipping Link 0 (between site 0 and 1) creates a geometric defect
    probs_flux = run_simulation_gpu(H_gpu, flux_link=0)
    cp.cuda.Stream.null.synchronize()
    print(f"  Time: {time.time()-t0:.4f}s")
    
    # Plotting
    times = np.arange(T_STEPS) * DT
    plt.figure(figsize=(10, 6))
    plt.plot(times, probs_vac, label="Vacuum", linewidth=3)
    plt.plot(times, probs_flux, label="With Flux (Geometric Phase)", linewidth=3, linestyle='--')
    plt.title("GPU Accelerated: 3D Substrate Interferometer")
    plt.xlabel("Time")
    plt.ylabel("Probability at Corner (7)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("gpu_cube_result.png")
    print("Plot saved to gpu_cube_result.png")