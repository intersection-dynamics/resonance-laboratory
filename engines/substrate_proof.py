import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sp
    HAS_GPU = True
    print(" [✓] CuPy detected. Running on GPU.")
except ImportError:
    import scipy.sparse as sp
    HAS_GPU = False
    print(" [!] CuPy not found. Falling back to CPU (slower).")
    # Aliases to make code compatible with both
    cp = np
    cpx_sp = sp

# =============================================================================
# Configuration
# =============================================================================
N_SITES = 8 
N_LINKS = 12
DIM = N_SITES * (1 << N_LINKS) # 32,768 states
HOPPING_AMP = 1.0

# =============================================================================
# GPU Kernels / Helper Functions
# =============================================================================

def apply_hop(psi_in, u, v, link_id):
    """
    Applies the Hopping Operator U_uv to state vector psi_in.
    Moves amplitude from Site u -> Site v and flips Link bit 'link_id'.
    """
    psi_out = cp.zeros_like(psi_in)
    
    # 1. Calculate Indices
    # We process the entire block of gauge states for site 'u' at once.
    # Site u range: [u * 4096, (u+1) * 4096)
    
    start_u = u * (1 << N_LINKS)
    end_u   = (u + 1) * (1 << N_LINKS)
    
    # Extract amplitudes at source
    # Note: simple slicing in CuPy creates a view, which is efficient
    amp_block = psi_in[start_u:end_u]
    
    # 2. Determine Destination Indices
    # We map gauge `g` to `g ^ (1<<link_id)`
    # The destination base is v * 4096
    
    start_v = v * (1 << N_LINKS)
    
    # Generate the permutation map for the gauge bits
    # g_indices = 0, 1, 2... 4095
    g_indices = cp.arange(1 << N_LINKS, dtype=cp.int32)
    
    # Flip the specific bit for the link we crossed
    g_flipped = g_indices ^ (1 << link_id)
    
    # 3. Assign to Output
    # We write the source amplitudes into the destination, permuted
    # psi_out[start_v + g_flipped] = amp_block
    
    # Using advanced indexing
    dest_indices = start_v + g_flipped
    psi_out[dest_indices] = amp_block
    
    return psi_out

def get_active_gauge_int(psi, site_idx):
    """Extracts the integer representation of the gauge field at a specific site."""
    start = site_idx * (1 << N_LINKS)
    end = (site_idx + 1) * (1 << N_LINKS)
    
    block = psi[start:end]
    # Find index of max amplitude
    idx = int(cp.argmax(cp.abs(block)))
    
    # Check if it's actually occupied
    if abs(block[idx]) < 1e-6:
        return None
    return idx

# =============================================================================
# Main Experiment: The Interferometer
# =============================================================================

def run_experiment():
    print(f"\n--- Substrate Framework: Path Memory Test ---")
    print(f"Hilbert Space Dimension: {DIM}")
    
    # Initial State: Particle at Corner 0, Vacuum Gauge (0)
    psi_0 = cp.zeros(DIM, dtype=cp.complex128)
    psi_0[0] = 1.0 
    
    # ---------------------------------------------------------
    # Path A: 0 -> 1 -> 3 (Right -> Up)
    # ---------------------------------------------------------
    print("\n[1] Simulating Path A (0 -> 1 -> 3)...")
    # Link 0 connects 0-1
    psi_A = apply_hop(psi_0, u=0, v=1, link_id=0) 
    # Link 5 connects 1-3
    psi_A = apply_hop(psi_A, u=1, v=3, link_id=5) 
    
    # ---------------------------------------------------------
    # Path B: 0 -> 2 -> 3 (Up -> Right)
    # ---------------------------------------------------------
    print("[2] Simulating Path B (0 -> 2 -> 3)...")
    # Link 4 connects 0-2
    psi_B = apply_hop(psi_0, u=0, v=2, link_id=4) 
    # Link 1 connects 2-3 (Note: 2->3 is backward relative to defined 3->2? 
    # In Z2 flip, direction doesn't matter, just the bit flip).
    # Checking connectivity defs from previous script:
    # 0->1 is Link 0. 1->3 is Link 5.
    # 0->2 is Link 4. 2->3 is Link 1 (or 3->2 is Link 1).
    psi_B = apply_hop(psi_B, u=2, v=3, link_id=1) 
    
    # ---------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------
    # Check Overlap
    overlap = cp.vdot(psi_A, psi_B)
    mag_A = cp.linalg.norm(psi_A)
    mag_B = cp.linalg.norm(psi_B)
    
    # Move results to CPU for printing
    ov_val = float(abs(overlap))
    
    print(f"\n[3] Results:")
    print(f"    Magnitude |Psi_A|: {float(mag_A):.2f}")
    print(f"    Magnitude |Psi_B|: {float(mag_B):.2f}")
    print(f"    Overlap <A|B>    : {ov_val:.6f}")
    
    # Analyze Gauge State
    g_A = get_active_gauge_int(psi_A, 3)
    g_B = get_active_gauge_int(psi_B, 3)
    
    print(f"\n[4] Gauge Field Memory Scan (Site 3):")
    print(f"    Path A Gauge State: {g_A} \t(Binary: {bin(g_A)[2:].zfill(12)})")
    print(f"    Path B Gauge State: {g_B} \t(Binary: {bin(g_B)[2:].zfill(12)})")
    
    print("\n" + "="*60)
    if g_A != g_B:
        print(" VERDICT: PATHS DISTINGUISHABLE")
        print(" The geometry has recorded the particle's history.")
        print(" This confirms 'Geometry is Memory'.")
    else:
        print(" VERDICT: PATHS INDISTINGUISHABLE")
        print(" Standard Quantum Mechanics (Memory Erased).")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_experiment()