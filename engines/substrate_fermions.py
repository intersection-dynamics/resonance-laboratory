import numpy as np

# =============================================================================
# 1. The Substrate: Quaternion Links
# =============================================================================
# We model the link states not as 0/1, but as 2x2 SU(2) Matrices.
# This represents the "twist" stored in the geometry.

sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# The Algebra of the Substrate
# Moving along X-links injects an 'i' twist.
# Moving along Y-links injects a 'j' twist.
# Moving along Z-links injects a 'k' twist.
GEN_X = -1j * sigma_x
GEN_Y = -1j * sigma_y
GEN_Z = -1j * sigma_z

def get_state_name(m):
    if np.allclose(m, sigma_0): return " 1"
    if np.allclose(m, -sigma_0): return "-1"
    if np.allclose(m, GEN_X): return " i"
    if np.allclose(m, -GEN_X): return "-i"
    if np.allclose(m, GEN_Y): return " j"
    if np.allclose(m, -GEN_Y): return "-j"
    if np.allclose(m, GEN_Z): return " k"
    if np.allclose(m, -GEN_Z): return "-k"
    return "?"

# =============================================================================
# 2. The Experiment: Exchange of Excitations
# =============================================================================

print("--- Substrate Framework: Quaternion Topology ---")
print("Deriving Statistics from Non-Abelian Connectivity.\n")

# Initial State of the Universe (The Gauge Field)
# We track the cumulative geometric phase of the system.
psi_universe = sigma_0 

print(f"Initial Universe State: {get_state_name(psi_universe)}")

# --- Timeline A (Clockwise Exchange) ---
# Excitation 1 moves Right (X), Excitation 2 moves Up (Y).
# Order: X then Y.
print("\n[Timeline A] Sequence: Hop X -> Hop Y")
psi_A = psi_universe
psi_A = GEN_X @ psi_A  # Apply X-twist
psi_A = GEN_Y @ psi_A  # Apply Y-twist

# --- Timeline B (Counter-Clockwise Exchange) ---
# Excitation 1 moves Up (Y), Excitation 2 moves Right (X).
# Order: Y then X.
print("[Timeline B] Sequence: Hop Y -> Hop X")
psi_B = psi_universe
psi_B = GEN_Y @ psi_B  # Apply Y-twist
psi_B = GEN_X @ psi_B  # Apply X-twist

# =============================================================================
# 3. The Comparison
# =============================================================================

print("\n--- Results ---")
print(f"Timeline A Final State: {get_state_name(psi_A)}")
print(f"Timeline B Final State: {get_state_name(psi_B)}")

# Measure Overlap (Trace of A * B_dagger)
# Standard inner product for matrices
overlap = np.trace(psi_A @ psi_B.conj().T) / 2.0

print(f"Overlap <A|B>: {overlap:.4f}")

print("\n" + "="*60)
if np.isclose(overlap, 1.0):
    print(" VERDICT: BOSONS")
elif np.isclose(overlap, -1.0):
    print(" VERDICT: FERMIONS")
    print(" The Non-Commutative Geometry of the Substrate")
    print(" has forced the wavefunction to acquire a (-1) phase.")
else:
    print(" VERDICT: UNKNOWN")
print("="*60)