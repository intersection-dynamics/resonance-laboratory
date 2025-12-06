import sys
import numpy as np
import time

# Attempt GPU import
try:
    import cupy as cp
    HAS_GPU = True
    print(" [✓] CuPy detected. Running on GPU.")
except ImportError:
    import numpy as cp
    HAS_GPU = False
    print(" [!] CuPy not found. Running on CPU.")

# =============================================================================
# Configuration
# =============================================================================
N_SITES = 8 
N_LINKS = 12
# Two-Particle Hilbert Space is larger
# Basis: |r1, r2, Gauge>
# Naive dimension: 8 * 8 * 2^12 = 262,144 states.
# We will map (r1, r2, g) -> linear index.
# Index = (r1 * 8 + r2) * 4096 + g
DIM_GAUGE = 1 << N_LINKS
DIM_MATTER = N_SITES * N_SITES
DIM_TOTAL = DIM_MATTER * DIM_GAUGE

# Connectivity Map (u -> v via link_id)
# 0-1 (L0), 1-3 (L5), 3-2 (L1), 2-0 (L4) on the z=0 face.
# Note: Check link definitions from previous standard cube.
# Standard Cube:
# x-edges: (0,1,L0), (2,3,L1)...
# y-edges: (0,2,L4), (1,3,L5)...
LINKS_MAP = {
    (0,1): 0, (1,0): 0,
    (2,3): 1, (3,2): 1,
    (0,2): 4, (2,0): 4,
    (1,3): 5, (3,1): 5
}

# =============================================================================
# Kernel: Two-Particle Hopping
# =============================================================================
def apply_hop_2p(psi_in, p_idx, u, v):
    """
    Moves Particle 'p_idx' (0 or 1) from site u to v.
    Updates the Gauge Field (Z2 Flip).
    """
    # Verify link exists
    if (u,v) not in LINKS_MAP:
        raise ValueError(f"No link between {u} and {v}")
    
    link_id = LINKS_MAP[(u,v)]
    
    psi_out = cp.zeros_like(psi_in)
    
    # We operation on blocks.
    # We need to find all states where P[p_idx] == u.
    # State Index I = (r1 * 8 + r2) * 4096 + g
    
    # To vectorize, we construct the permutation map for the entire array.
    # It's faster to find the indices.
    
    # 1. Create coordinate arrays for the whole basis
    # This is heavy to do every step. Optimization: Do it on the fly with logic.
    
    # We are moving amplitude FROM sources TO destinations.
    # Source Logic:
    # If p_idx == 0 (Particle 1 moves):
    #   Source states are (u, ANY, g)
    #   Dest states are   (v, ANY, g^flip)
    # If p_idx == 1 (Particle 2 moves):
    #   Source states are (ANY, u, g)
    #   Dest states are   (ANY, v, g^flip)
    
    # Let's generate the indices vector for the full dimension
    all_indices = cp.arange(DIM_TOTAL)
    
    # Decode r1, r2
    all_g = all_indices % DIM_GAUGE
    temp  = all_indices // DIM_GAUGE
    all_r2 = temp % N_SITES
    all_r1 = temp // N_SITES
    
    # Select Active Amplitudes
    if p_idx == 0:
        mask = (all_r1 == u)
        # Check Hard-Core Constraint: Target v must not be occupied by r2
        mask = mask & (all_r2 != v)
    else:
        mask = (all_r2 == u)
        mask = mask & (all_r1 != v)
        
    src_indices = all_indices[mask]
    
    if src_indices.size == 0:
        return psi_out # No valid moves
    
    # Compute Dest Indices
    # Extract gauge part of sources
    src_g = all_g[mask]
    dst_g = src_g ^ (1 << link_id)
    
    # Extract matter part
    src_r1 = all_r1[mask]
    src_r2 = all_r2[mask]
    
    if p_idx == 0:
        dst_r1 = cp.full_like(src_r1, v)
        dst_r2 = src_r2
    else:
        dst_r1 = src_r1
        dst_r2 = cp.full_like(src_r2, v)
        
    # Reassemble Index
    dst_indices = (dst_r1 * 8 + dst_r2) * DIM_GAUGE + dst_g
    
    # Transfer Amplitude
    # Note: In Z2 gauge, phase is usually +1. If you wanted magnetic flux, 
    # you would multiply by -1 here if link was already 1. 
    # For pure geometric memory, we just move the amplitude.
    
    # Use Scatter add (or just set if we assume unitary flow without collisions)
    psi_out[dst_indices] = psi_in[src_indices]
    
    # Add back the parts of wavefunction that DIDN'T move?
    # No, this function represents the OPERATION "Hop". 
    # If the particle wasn't at u, it gets annihilated (operator gives 0).
    # We are evolving a specific state, so this is correct.
    
    return psi_out

# =============================================================================
# The Experiment
# =============================================================================
def run_fermion_test():
    print(f"\n--- Substrate Framework: Fermionic Exchange Test ---")
    print(f"Dim: {DIM_TOTAL} states.")
    
    # Initial State: P1 at 0, P2 at 3. Vacuum Gauge.
    # Index = (0 * 8 + 3) * 4096 + 0 = 3 * 4096 = 12288
    psi_0 = cp.zeros(DIM_TOTAL, dtype=cp.complex128)
    init_idx = (0 * 8 + 3) * DIM_GAUGE + 0
    psi_0[init_idx] = 1.0
    
    print("\nInitial State: |P1=0, P2=3> |Gauge=0>")
    
    # --- Timeline A: Clockwise Exchange ---
    # P1: 0 -> 1 -> 3
    # P2: 3 -> 2 -> 0
    # Step 1: P1(0->1), P2(3->2)
    # Step 2: P1(1->3), P2(2->0)
    
    print("Simulating Timeline A (Clockwise)...")
    psi_A = psi_0.copy()
    
    # Step 1
    # We chain operations. Note: Order of P1 vs P2 in same timestep shouldn't matter 
    # for non-overlapping paths.
    psi_A = apply_hop_2p(psi_A, p_idx=0, u=0, v=1) # P1 to 1
    psi_A = apply_hop_2p(psi_A, p_idx=1, u=3, v=2) # P2 to 2
    
    # Step 2
    psi_A = apply_hop_2p(psi_A, p_idx=0, u=1, v=3) # P1 to 3
    psi_A = apply_hop_2p(psi_A, p_idx=1, u=2, v=0) # P2 to 0
    
    # --- Timeline B: Counter-Clockwise Exchange ---
    # P1: 0 -> 2 -> 3
    # P2: 3 -> 1 -> 0
    # Step 1: P1(0->2), P2(3->1)
    # Step 2: P1(2->3), P2(1->0)
    
    print("Simulating Timeline B (Counter-Clockwise)...")
    psi_B = psi_0.copy()
    
    # Step 1
    psi_B = apply_hop_2p(psi_B, p_idx=0, u=0, v=2) # P1 to 2
    psi_B = apply_hop_2p(psi_B, p_idx=1, u=3, v=1) # P2 to 1
    
    # Step 2
    psi_B = apply_hop_2p(psi_B, p_idx=0, u=2, v=3) # P1 to 3
    psi_B = apply_hop_2p(psi_B, p_idx=1, u=1, v=0) # P2 to 0
    
    # --- Analysis ---
    # Both timelines result in P1 at 3, P2 at 0.
    # The question is: What is the Gauge State?
    
    overlap = cp.vdot(psi_A, psi_B)
    
    # Get Gauge indices
    def get_gauge(psi):
        # Find index
        idx = int(cp.argmax(cp.abs(psi)))
        # Decode
        g = idx % DIM_GAUGE
        return g
    
    g_A = get_gauge(psi_A)
    g_B = get_gauge(psi_B)
    
    print(f"\nResults:")
    print(f"  Timeline A Gauge: {g_A} (Binary: {bin(g_A)[2:].zfill(12)})")
    print(f"  Timeline B Gauge: {g_B} (Binary: {bin(g_B)[2:].zfill(12)})")
    print(f"  Overlap <A|B>: {overlap:.4f}")
    
    if abs(1.0 - overlap) < 1e-5:
        print("\nDERIVATION: BOSONS")
        print("The histories are identical. Exchanging particles leaves the universe unchanged.")
        print("Constraint Violation: Your substrate generates Bosons, not Fermions.")
    elif abs(1.0 + overlap) < 1e-5:
        print("\nDERIVATION: FERMIONS")
        print("The histories differ by exactly a Phase of -1.")
        print("Success: Pauli Exclusion Principle derived from Geometry.")
    elif abs(overlap) < 1e-5:
        print("\nDERIVATION: DISTINGUISHABLE OBJECTS")
        print("The histories are orthogonal. The universe remembers 'Who went where'.")
        print("This is neither Boson nor Fermion. It is Classical Memory.")
        
    print("="*60)

if __name__ == "__main__":
    run_fermion_test()