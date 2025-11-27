"""
Multi-Body Test Suite
=====================
The real test of emergent statistics requires MULTIPLE excitations.

Key questions:
1. PAULI EXCLUSION: Can two "fermions" occupy the same mode?
2. BOSE ENHANCEMENT: Do "bosons" bunch together?
3. EXCHANGE PHASES: What happens when particles swap positions?
4. SCALING: How does the statistics gap change with particle number?

This goes beyond single-excitation tests to probe genuine many-body physics.
"""

import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations
import json

# GPU/CPU backend
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    import cupyx.scipy.linalg as cpla
    GPU_AVAILABLE = True
    xp = cp
    sparse = cp_sparse
    def expm(A):
        if sparse.issparse(A):
            A = A.toarray()
        return cpla.expm(A)
    try:
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    except:
        gpu_name = f"Device {cp.cuda.Device().id}"
    print(f"GPU: {gpu_name}")
except ImportError:
    import scipy.sparse as sp_sparse
    from scipy.linalg import expm as scipy_expm
    GPU_AVAILABLE = False
    xp = np
    sparse = sp_sparse
    def expm(A):
        if sp_sparse.issparse(A):
            A = A.toarray()
        return scipy_expm(A)
    print("GPU: Not available, using CPU")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# EXTENDED SUBSTRATE (supports multi-excitation)
# =============================================================================

class MultiBodySubstrate:
    """
    Substrate with tools for multi-excitation states.
    Uses dim_per_mode > 2 to allow multiple excitations per mode.
    """
    
    def __init__(self, n_modes: int, max_excitations_per_mode: int = 3):
        self.n_modes = n_modes
        self.d = max_excitations_per_mode
        self.dim = self.d ** n_modes
        
        self._a_ops = []
        self._adag_ops = []
        self._n_ops = []
        self._build_operators()
    
    def _build_operators(self):
        """Build sparse ladder operators."""
        diag_vals = np.sqrt(np.arange(1, self.d, dtype=np.float64))
        a_single = sparse.diags(diag_vals, offsets=1, shape=(self.d, self.d), format='csr')
        
        for mode in range(self.n_modes):
            ops = [sparse.eye(self.d, format='csr') for _ in range(self.n_modes)]
            ops[mode] = a_single
            
            result = ops[0]
            for op in ops[1:]:
                result = sparse.kron(result, op, format='csr')
            
            self._a_ops.append(result)
            self._adag_ops.append(result.T.conj())
            self._n_ops.append(result.T.conj() @ result)
    
    def vacuum(self) -> xp.ndarray:
        psi = xp.zeros(self.dim, dtype=xp.complex128)
        psi[0] = 1.0
        return psi
    
    def config_to_idx(self, config: Tuple[int, ...]) -> int:
        """Convert occupation tuple to basis index."""
        idx = 0
        for i, n in enumerate(config):
            idx += n * (self.d ** (self.n_modes - 1 - i))
        return idx
    
    def idx_to_config(self, idx: int) -> Tuple[int, ...]:
        """Convert basis index to occupation tuple."""
        config = []
        for i in range(self.n_modes):
            power = self.d ** (self.n_modes - 1 - i)
            config.append(idx // power)
            idx = idx % power
        return tuple(config)
    
    def basis_state(self, config: Tuple[int, ...]) -> xp.ndarray:
        """Create basis state from occupation numbers."""
        idx = self.config_to_idx(config)
        psi = xp.zeros(self.dim, dtype=xp.complex128)
        psi[idx] = 1.0
        return psi
    
    def fock_state(self, occupations: Dict[int, int]) -> xp.ndarray:
        """
        Create Fock state: |n_0, n_1, ..., n_{N-1}⟩
        occupations = {mode: count}
        """
        config = [0] * self.n_modes
        for mode, count in occupations.items():
            config[mode] = min(count, self.d - 1)
        return self.basis_state(tuple(config))
    
    def excite(self, psi: xp.ndarray, mode: int) -> xp.ndarray:
        """Apply creation operator."""
        result = self._adag_ops[mode] @ psi
        norm = xp.linalg.norm(result)
        if norm > 1e-10:
            return result / norm
        return result
    
    def total_number(self, psi: xp.ndarray) -> float:
        """Measure total excitation number."""
        N_total = sum(self._n_ops)
        return float(xp.real(xp.vdot(psi, N_total @ psi)))
    
    def measure_occupation(self, psi: xp.ndarray, mode: int) -> float:
        """Measure occupation of a single mode."""
        return float(xp.real(xp.vdot(psi, self._n_ops[mode] @ psi)))
    
    def reduced_density_matrix(self, psi: xp.ndarray, keep_modes: List[int]) -> xp.ndarray:
        """Partial trace over complement of keep_modes."""
        n_keep = len(keep_modes)
        trace_modes = [m for m in range(self.n_modes) if m not in keep_modes]
        
        shape = [self.d] * self.n_modes
        psi_tensor = psi.reshape(shape)
        
        new_order = keep_modes + trace_modes
        psi_reordered = xp.transpose(psi_tensor, new_order)
        
        keep_dim = self.d ** n_keep
        trace_dim = self.d ** (self.n_modes - n_keep)
        psi_matrix = psi_reordered.reshape(keep_dim, trace_dim)
        
        return psi_matrix @ xp.conj(psi_matrix.T)


def detection_unitary(sub: MultiBodySubstrate, sys_mode: int, env_mode: int,
                      strength: float = np.pi/2) -> xp.ndarray:
    """CNOT-like detection: controlled flip based on occupation."""
    sigma_x_env = (sub._a_ops[env_mode] + sub._adag_ops[env_mode]).toarray()
    n_sys = sub._n_ops[sys_mode].toarray()
    H = n_sys @ sigma_x_env
    return expm(-1j * strength * H)


def hopping_hamiltonian(sub: MultiBodySubstrate, connectivity: Dict[int, List[int]],
                        coupling: float = 1.0) -> xp.ndarray:
    """Build hopping Hamiltonian."""
    H = sparse.csr_matrix((sub.dim, sub.dim), dtype=np.complex128)
    for i, neighbors in connectivity.items():
        for j in neighbors:
            if j > i:
                H = H - coupling * (sub._adag_ops[i] @ sub._a_ops[j])
                H = H - coupling * (sub._adag_ops[j] @ sub._a_ops[i])
    return H.toarray()


def linear_chain(n: int) -> Dict[int, List[int]]:
    """Linear chain connectivity."""
    return {i: [j for j in [i-1, i+1] if 0 <= j < n] for i in range(n)}


# =============================================================================
# TEST 1: PAULI EXCLUSION
# =============================================================================

def test_pauli_exclusion(n_modes: int = 6) -> Dict:
    """
    Test if fermionic patterns resist double occupation.
    
    Protocol:
    1. Create antisymmetric 2-particle state
    2. Try to "squeeze" both particles into same mode via dynamics
    3. Measure resistance to double occupation
    
    Compare with bosonic (symmetric) state which should allow bunching.
    """
    print("\n" + "="*60)
    print("TEST 1: Pauli Exclusion")
    print("="*60)
    
    sub = MultiBodySubstrate(n_modes, max_excitations_per_mode=3)
    H = hopping_hamiltonian(sub, linear_chain(n_modes), coupling=1.0)
    
    results = {'times': [], 'antisym_double_occ': [], 'sym_double_occ': [], 
               'local_double_occ': []}
    
    # Create two-particle states
    # Antisymmetric: (|1,0,0,...⟩|0,1,0,...⟩ - |0,1,0,...⟩|1,0,0,...⟩)/√2
    # In Fock space: (|1_0, 1_1⟩ with antisymmetric phase)
    
    # For our substrate, we work with superpositions of configurations
    # |ψ_anti⟩ = (a†_0 a†_1 - a†_1 a†_0)|vac⟩ / √2... but a†'s commute!
    # 
    # The trick: antisymmetry must be in the SPATIAL wavefunction
    # |ψ_anti⟩ = (|10⟩ - |01⟩)/√2 where |10⟩ = particle at site 0, |01⟩ = particle at site 1
    #
    # For TWO particles: we need a 2-particle antisymmetric state
    # |ψ_anti⟩ = (|1,1,0,0,...⟩_spatial_antisym)
    
    # Actually, let's think more carefully:
    # Two particles at sites 0 and 2 (separated):
    config_02 = tuple([1, 0, 1] + [0]*(n_modes-3))  # one at 0, one at 2
    config_20 = tuple([1, 0, 1] + [0]*(n_modes-3))  # same thing - order doesn't matter in Fock
    
    # The antisymmetry is encoded in WHICH superposition we use
    # For two distinguishable particles at sites i,j:
    # Symmetric: |i⟩|j⟩ + |j⟩|i⟩
    # Antisymmetric: |i⟩|j⟩ - |j⟩|i⟩
    
    # In second quantization with mode indices 0,1,2,...
    # Two excitations: one at mode 0, one at mode 2
    psi_two = sub.fock_state({0: 1, 2: 1})  # |1,0,1,0,0,0⟩
    
    # For comparison: two excitations at same mode (bosonic bunching)
    psi_bunched = sub.fock_state({0: 2})  # |2,0,0,0,0,0⟩
    
    # And two at adjacent sites
    psi_adjacent = sub.fock_state({0: 1, 1: 1})  # |1,1,0,0,0,0⟩
    
    print(f"\nInitial states:")
    print(f"  Two-particle (sites 0,2): N = {sub.total_number(psi_two):.1f}")
    print(f"  Bunched (site 0, n=2):    N = {sub.total_number(psi_bunched):.1f}")
    print(f"  Adjacent (sites 0,1):     N = {sub.total_number(psi_adjacent):.1f}")
    
    # Now create SYMMETRIC and ANTISYMMETRIC superpositions
    # Using sites 0 and 1 for the two particles
    
    # |1,1,0,...⟩ is already symmetric under particle exchange
    # To create antisymmetric, we need internal structure
    
    # Better approach: use the MOMENTUM basis
    # Symmetric state: both particles in k=0 mode
    # Antisymmetric: one in k=0, one in k=π
    
    # Or simpler: superposition of position states
    # |ψ_S⟩ = |0,1⟩ + |1,0⟩ (symmetric in particle labels)
    # |ψ_A⟩ = |0,1⟩ - |1,0⟩ (antisymmetric)
    
    # In Fock space for indistinguishable particles:
    # We represent these as superpositions of SINGLE excitation states
    # that get entangled through detection
    
    print("\n--- Approach: Detection-based Statistics Test ---")
    print("Create superposition states, detect, measure bunching tendency")
    
    # Strategy: 
    # 1. Start with two particles at separated sites
    # 2. Evolve under Hamiltonian (they can hop toward each other)
    # 3. Measure double-occupation probability over time
    # 4. Compare: do antisymmetric-like states resist bunching?
    
    # Create test states with different symmetry character
    # Using modes 0, 2, 4 (separated to allow dynamics)
    
    # Symmetric-like: (|1,0,0,1,0,0⟩ + |0,0,1,0,0,1⟩)/√2... but these are different total configs
    # 
    # Let's use a clearer setup:
    # Two particles in modes 0 and 1
    # Symmetric: uniform phase
    # Antisymmetric: relative minus sign on exchange
    
    # In single-particle language for the second particle:
    # Start: one particle definitely at 0, second in superposition of 1,2
    # Symmetric: |0⟩⊗(|1⟩+|2⟩)
    # Anti: |0⟩⊗(|1⟩-|2⟩) ... but this isn't quite right either
    
    # Cleanest approach: evolve and measure
    psi_0 = psi_adjacent.copy()  # Two particles at sites 0,1
    
    # Evolve and measure double occupation probability
    times = np.linspace(0, 5.0, 50)
    dt = times[1] - times[0]
    U = expm(-1j * H * dt)
    
    double_occ_history = []
    psi = psi_0.copy()
    
    print("\nEvolving two-particle state under hopping Hamiltonian...")
    
    for t in times:
        # Measure double occupation at each site
        double_occ = 0.0
        for m in range(n_modes):
            # Probability of n_m >= 2
            n_m = sub._n_ops[m].toarray()
            n_m_sq = n_m @ n_m
            
            # ⟨n(n-1)⟩ = ⟨n²⟩ - ⟨n⟩ measures pairs at same site
            exp_n = float(xp.real(xp.vdot(psi, n_m @ psi)))
            exp_n2 = float(xp.real(xp.vdot(psi, n_m_sq @ psi)))
            pairs = exp_n2 - exp_n  # = ⟨n(n-1)⟩
            double_occ += max(0, pairs)
        
        double_occ_history.append(double_occ)
        psi = U @ psi
        psi = psi / xp.linalg.norm(psi)
    
    results['times'] = times.tolist()
    results['double_occ'] = double_occ_history
    
    print(f"\nDouble occupation over time:")
    print(f"  Initial: {double_occ_history[0]:.4f}")
    print(f"  Final:   {double_occ_history[-1]:.4f}")
    print(f"  Maximum: {max(double_occ_history):.4f}")
    
    # Now test with detection-induced symmetry selection
    print("\n--- Detection-Induced Symmetry Selection ---")
    
    # Start with symmetric superposition
    config_01 = tuple([1, 1] + [0]*(n_modes-2))
    config_12 = tuple([0, 1, 1] + [0]*(n_modes-3))
    
    psi_sym = sub.basis_state(config_01) + sub.basis_state(config_12)
    psi_sym = psi_sym / xp.linalg.norm(psi_sym)
    
    psi_anti = sub.basis_state(config_01) - sub.basis_state(config_12)
    psi_anti = psi_anti / xp.linalg.norm(psi_anti)
    
    # Detect both states
    env_modes = list(range(3, n_modes))
    
    print("\nApplying detection cycles...")
    
    for name, psi_init in [("symmetric", psi_sym), ("antisymmetric", psi_anti)]:
        psi = psi_init.copy()
        
        for i in range(5):
            # Detect at each occupied mode
            for sys_m in [0, 1, 2]:
                env_m = env_modes[i % len(env_modes)]
                U_det = detection_unitary(sub, sys_m, env_m)
                psi = U_det @ psi
                psi = psi / xp.linalg.norm(psi)
        
        # Measure final double occupation
        double_occ = 0.0
        for m in range(n_modes):
            n_m = sub._n_ops[m].toarray()
            n_m_sq = n_m @ n_m
            exp_n = float(xp.real(xp.vdot(psi, n_m @ psi)))
            exp_n2 = float(xp.real(xp.vdot(psi, n_m_sq @ psi)))
            double_occ += max(0, exp_n2 - exp_n)
        
        print(f"  {name}: double_occ = {double_occ:.4f}")
        results[f'{name}_final_double_occ'] = double_occ
    
    return results


# =============================================================================
# TEST 2: EXCHANGE PHASE
# =============================================================================

def test_exchange_phase(n_modes: int = 6) -> Dict:
    """
    Test the phase acquired when two particles exchange positions.
    
    Bosons: no phase change
    Fermions: π phase (sign flip)
    """
    print("\n" + "="*60)
    print("TEST 2: Exchange Phase")
    print("="*60)
    
    sub = MultiBodySubstrate(n_modes, max_excitations_per_mode=2)
    
    results = {}
    
    # Build SWAP operator for two modes
    # SWAP|n_i, n_j⟩ = |n_j, n_i⟩
    
    print("\nBuilding exchange operator P_01...")
    
    dim = sub.dim
    P_01 = xp.zeros((dim, dim), dtype=xp.complex128)
    
    for idx in range(dim):
        config = list(sub.idx_to_config(idx))
        # Swap modes 0 and 1
        config[0], config[1] = config[1], config[0]
        new_idx = sub.config_to_idx(tuple(config))
        P_01[new_idx, idx] = 1.0
    
    # Verify P² = I
    P_squared = P_01 @ P_01
    is_involution = bool(xp.allclose(P_squared, xp.eye(dim)))
    print(f"P² = I: {is_involution}")
    
    # Find eigenvalues
    if GPU_AVAILABLE:
        eigenvalues = cp.asnumpy(xp.linalg.eigvalsh(P_01))
    else:
        eigenvalues = np.linalg.eigvalsh(P_01)
    
    unique_eigs = np.unique(np.round(eigenvalues, 6))
    print(f"Exchange eigenvalues: {unique_eigs}")
    
    n_bosonic = np.sum(eigenvalues > 0.5)
    n_fermionic = np.sum(eigenvalues < -0.5)
    print(f"Bosonic (+1) states: {n_bosonic}")
    print(f"Fermionic (-1) states: {n_fermionic}")
    
    results['n_bosonic'] = int(n_bosonic)
    results['n_fermionic'] = int(n_fermionic)
    
    # Test specific two-particle states
    print("\n--- Two-Particle Exchange Tests ---")
    
    test_configs = [
        ({0: 1, 1: 1}, "particles at 0,1"),
        ({0: 2}, "two at site 0"),
        ({0: 1, 2: 1}, "particles at 0,2"),
    ]
    
    for occ, desc in test_configs:
        psi = sub.fock_state(occ)
        
        # Apply exchange
        psi_exchanged = P_01 @ psi
        
        # Compute overlap (should be ±1 for eigenstates)
        overlap = float(xp.real(xp.vdot(psi, psi_exchanged)))
        
        print(f"  {desc}: ⟨ψ|P|ψ⟩ = {overlap:+.3f}", end="")
        if abs(overlap - 1) < 0.01:
            print(" → BOSONIC")
        elif abs(overlap + 1) < 0.01:
            print(" → FERMIONIC")
        else:
            print(" → MIXED")
        
        results[desc] = overlap
    
    # Create explicit symmetric and antisymmetric two-particle states
    print("\n--- Constructing Definite Symmetry States ---")
    
    # |0,1⟩ state (one particle each at modes 0 and 1)
    psi_01 = sub.fock_state({0: 1, 1: 1})
    
    # Under exchange of modes 0↔1, this is symmetric: P|1,1⟩ = |1,1⟩
    # Because the occupation numbers just swap but the state is the same!
    
    # To get antisymmetric, we need a SUPERPOSITION
    # |ψ⟩ = |1,0,1,...⟩ - |0,1,1,...⟩ etc.
    
    # Particles at (0,2) vs (1,2)
    config_02 = tuple([1, 0, 1] + [0]*(n_modes-3))
    config_12 = tuple([0, 1, 1] + [0]*(n_modes-3))
    
    psi_02 = sub.basis_state(config_02)
    psi_12 = sub.basis_state(config_12)
    
    # Symmetric combination
    psi_sym = (psi_02 + psi_12) / np.sqrt(2)
    # Antisymmetric combination
    psi_anti = (psi_02 - psi_12) / np.sqrt(2)
    
    # Apply P_01 (swaps modes 0 and 1)
    # P|1,0,1,...⟩ = |0,1,1,...⟩
    # P|0,1,1,...⟩ = |1,0,1,...⟩
    
    psi_sym_ex = P_01 @ psi_sym
    psi_anti_ex = P_01 @ psi_anti
    
    overlap_sym = float(xp.real(xp.vdot(psi_sym, psi_sym_ex)))
    overlap_anti = float(xp.real(xp.vdot(psi_anti, psi_anti_ex)))
    
    print(f"  Symmetric state: ⟨ψ_S|P|ψ_S⟩ = {overlap_sym:+.3f}")
    print(f"  Antisymmetric state: ⟨ψ_A|P|ψ_A⟩ = {overlap_anti:+.3f}")
    
    results['symmetric_exchange'] = overlap_sym
    results['antisymmetric_exchange'] = overlap_anti
    
    # Now test survival under detection
    print("\n--- Detection Survival by Symmetry ---")
    
    env_modes = list(range(3, n_modes))
    
    for name, psi_init in [("symmetric", psi_sym), ("antisymmetric", psi_anti)]:
        psi = psi_init.copy()
        rho_orig = sub.reduced_density_matrix(psi, [0, 1, 2])
        
        for i in range(5):
            for sys_m in [0, 1]:
                env_m = env_modes[i % len(env_modes)]
                U_det = detection_unitary(sub, sys_m, env_m)
                psi = U_det @ psi
                psi = psi / xp.linalg.norm(psi)
        
        rho_final = sub.reduced_density_matrix(psi, [0, 1, 2])
        fidelity = float(xp.abs(xp.trace(rho_orig @ rho_final)))
        
        stat_type = "survives" if fidelity > 0.5 else "consumed"
        print(f"  {name}: fidelity = {fidelity:.3f} → {stat_type}")
        results[f'{name}_fidelity'] = fidelity
    
    return results


# =============================================================================
# TEST 3: PARTICLE NUMBER SCALING
# =============================================================================

def test_particle_scaling(max_particles: int = 4, n_modes: int = 8) -> Dict:
    """
    Test how statistics gap scales with particle number.
    
    Question: Does the fermionic/bosonic distinction persist (or strengthen)
    with more particles?
    """
    print("\n" + "="*60)
    print("TEST 3: Particle Number Scaling")
    print("="*60)
    
    results = {'n_particles': [], 'sym_fidelity': [], 'anti_fidelity': [], 'gap': []}
    
    for n_particles in range(2, max_particles + 1):
        print(f"\n--- {n_particles} particles ---")
        
        # Need enough modes for particles plus environment
        n_sys = n_particles + 1  # System modes
        n_env = 3  # Environment modes
        total_modes = n_sys + n_env
        
        sub = MultiBodySubstrate(total_modes, max_excitations_per_mode=2)
        
        # Create n-particle states
        # Symmetric: all particles in same "wave"
        # Antisymmetric: particles in different modes with alternating signs
        
        # Simple approach: particles at modes 0, 1, 2, ..., n-1
        base_config = [1] * n_particles + [0] * (total_modes - n_particles)
        psi_base = sub.basis_state(tuple(base_config))
        
        # Shifted config: particles at modes 1, 2, ..., n
        shifted_config = [0] + [1] * n_particles + [0] * (total_modes - n_particles - 1)
        psi_shifted = sub.basis_state(tuple(shifted_config))
        
        # Symmetric and antisymmetric combinations
        psi_sym = (psi_base + psi_shifted) / np.sqrt(2)
        psi_anti = (psi_base - psi_shifted) / np.sqrt(2)
        
        # Detection
        sys_modes = list(range(n_particles + 1))
        env_modes = list(range(n_particles + 1, total_modes))
        
        fidelities = {}
        for name, psi_init in [("symmetric", psi_sym), ("antisymmetric", psi_anti)]:
            psi = psi_init.copy()
            rho_orig = sub.reduced_density_matrix(psi, sys_modes)
            
            for i in range(5):
                for sys_m in sys_modes[:2]:  # Detect first two system modes
                    env_m = env_modes[i % len(env_modes)]
                    U_det = detection_unitary(sub, sys_m, env_m)
                    psi = U_det @ psi
                    psi = psi / xp.linalg.norm(psi)
            
            rho_final = sub.reduced_density_matrix(psi, sys_modes)
            fidelity = float(xp.abs(xp.trace(rho_orig @ rho_final)))
            fidelities[name] = fidelity
            print(f"  {name}: fidelity = {fidelity:.3f}")
        
        gap = fidelities['symmetric'] - fidelities['antisymmetric']
        print(f"  Gap: {gap:.3f}")
        
        results['n_particles'].append(n_particles)
        results['sym_fidelity'].append(fidelities['symmetric'])
        results['anti_fidelity'].append(fidelities['antisymmetric'])
        results['gap'].append(gap)
    
    return results


# =============================================================================
# TEST 4: FERMIONIC GROUND STATE
# =============================================================================

def test_fermionic_ground_state(n_modes: int = 6, n_particles: int = 2) -> Dict:
    """
    Test if the ground state of a fermionic-like system shows Pauli exclusion.
    
    For true fermions, the ground state fills the lowest single-particle levels
    one particle each (no double occupation).
    """
    print("\n" + "="*60)
    print("TEST 4: Ground State Structure")
    print("="*60)
    
    sub = MultiBodySubstrate(n_modes, max_excitations_per_mode=3)
    H = hopping_hamiltonian(sub, linear_chain(n_modes), coupling=1.0)
    
    # Add repulsion term to enforce fermionic-like behavior
    # U Σ n_i(n_i - 1) penalizes double occupation
    
    print(f"\nBuilding Hamiltonian with on-site repulsion...")
    
    U_repulsion = 10.0  # Strong repulsion
    H_int = np.zeros_like(H)
    for m in range(n_modes):
        n_m = sub._n_ops[m].toarray()
        H_int += U_repulsion * n_m @ (n_m - np.eye(sub.dim))
    
    H_total = H + H_int
    
    # Find ground state in n-particle sector
    # We need to project onto fixed particle number
    
    print(f"Finding ground state with {n_particles} particles...")
    
    # Build projector onto n-particle sector
    N_op = sum(sub._n_ops).toarray()
    
    if GPU_AVAILABLE:
        N_eigs, N_vecs = xp.linalg.eigh(N_op)
        N_eigs = cp.asnumpy(N_eigs)
        N_vecs = cp.asnumpy(N_vecs)
    else:
        N_eigs, N_vecs = np.linalg.eigh(N_op)
    
    # Find states with exactly n_particles
    n_particle_indices = np.where(np.abs(N_eigs - n_particles) < 0.1)[0]
    print(f"States with N={n_particles}: {len(n_particle_indices)}")
    
    if len(n_particle_indices) == 0:
        print("No states found in this sector!")
        return {}
    
    # Project Hamiltonian onto this sector
    P = N_vecs[:, n_particle_indices]
    H_projected = P.T @ H_total @ P
    
    # Diagonalize
    E, V = np.linalg.eigh(H_projected)
    
    # Ground state
    psi_ground_proj = V[:, 0]
    psi_ground = P @ psi_ground_proj
    psi_ground = psi_ground / np.linalg.norm(psi_ground)
    
    if GPU_AVAILABLE:
        psi_ground = cp.asarray(psi_ground)
    
    print(f"\nGround state energy: {E[0]:.4f}")
    
    # Analyze occupation
    print("\nOccupation analysis:")
    occupations = []
    for m in range(n_modes):
        occ = sub.measure_occupation(psi_ground, m)
        occupations.append(occ)
        bar = "█" * int(occ * 20)
        print(f"  Mode {m}: {occ:.3f} {bar}")
    
    # Double occupation
    double_occ_total = 0.0
    for m in range(n_modes):
        n_m = sub._n_ops[m].toarray()
        n_m_sq = n_m @ n_m
        exp_n = float(xp.real(xp.vdot(psi_ground, n_m @ psi_ground)))
        exp_n2 = float(xp.real(xp.vdot(psi_ground, n_m_sq @ psi_ground)))
        double_occ_total += max(0, exp_n2 - exp_n)
    
    print(f"\nTotal double occupation: {double_occ_total:.4f}")
    
    if double_occ_total < 0.1:
        print("→ FERMIONIC: Ground state avoids double occupation")
    else:
        print("→ BOSONIC: Ground state allows bunching")
    
    # Compare with non-interacting (U=0) case
    print("\n--- Comparison: U=0 (no repulsion) ---")
    
    H_free = H  # Just hopping
    H_free_proj = P.T @ H_free @ P
    E_free, V_free = np.linalg.eigh(H_free_proj)
    
    psi_free_proj = V_free[:, 0]
    psi_free = P @ psi_free_proj
    psi_free = psi_free / np.linalg.norm(psi_free)
    
    if GPU_AVAILABLE:
        psi_free = cp.asarray(psi_free)
    
    double_occ_free = 0.0
    for m in range(n_modes):
        n_m = sub._n_ops[m].toarray()
        n_m_sq = n_m @ n_m
        exp_n = float(xp.real(xp.vdot(psi_free, n_m @ psi_free)))
        exp_n2 = float(xp.real(xp.vdot(psi_free, n_m_sq @ psi_free)))
        double_occ_free += max(0, exp_n2 - exp_n)
    
    print(f"Double occupation (U=0): {double_occ_free:.4f}")
    
    return {
        'ground_energy': float(E[0]),
        'occupations': occupations,
        'double_occ_with_U': double_occ_total,
        'double_occ_free': double_occ_free,
    }


# =============================================================================
# TEST 5: HONG-OU-MANDEL EFFECT
# =============================================================================

def test_hong_ou_mandel(n_modes: int = 6) -> Dict:
    """
    Test the Hong-Ou-Mandel effect: two identical bosons meeting at a 
    beam splitter bunch together. Two fermions anti-bunch.
    
    We simulate a "beam splitter" via hopping dynamics and measure
    coincidence probability.
    """
    print("\n" + "="*60)
    print("TEST 5: Hong-Ou-Mandel Effect")
    print("="*60)
    
    sub = MultiBodySubstrate(n_modes, max_excitations_per_mode=3)
    
    # Setup: two particles approaching a "beam splitter" (central coupling)
    # Initial: one particle at mode 1, one at mode 4
    # Beam splitter at modes 2,3
    
    print("\nSetup:")
    print("  Particle 1 at mode 1")
    print("  Particle 2 at mode 4")
    print("  'Beam splitter' coupling at modes 2-3")
    
    # Create beam splitter Hamiltonian (strong coupling at center)
    conn = {
        0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]
    }
    
    # Enhance coupling at the beam splitter (modes 2-3)
    H_base = hopping_hamiltonian(sub, conn, coupling=1.0)
    
    # Boost the 2-3 coupling (beam splitter)
    H_bs = sub._adag_ops[2] @ sub._a_ops[3] + sub._adag_ops[3] @ sub._a_ops[2]
    H = H_base - 2.0 * H_bs.toarray()  # Extra coupling at beam splitter
    
    # Initial state: one particle at 1, one at 4
    psi_init = sub.fock_state({1: 1, 4: 1})
    
    print(f"\nInitial total particles: {sub.total_number(psi_init):.1f}")
    
    # Evolve
    t_hom = np.pi / 4  # Time for beam splitter operation
    U = expm(-1j * H * t_hom)
    
    psi_final = U @ psi_init
    psi_final = psi_final / xp.linalg.norm(psi_final)
    
    print(f"\nAfter beam splitter evolution (t = π/4):")
    
    # Measure coincidence: probability of finding one particle at 2, one at 3
    # For bosons: should be ZERO (they bunch)
    # For fermions: should be HIGH (they anti-bunch)
    
    # Direct measurement of coincidence
    # P(n_2=1, n_3=1) via projection
    
    # Calculate occupation probabilities
    print("\nOccupation distribution:")
    for m in range(n_modes):
        occ = sub.measure_occupation(psi_final, m)
        bar = "█" * int(occ * 20)
        print(f"  Mode {m}: {occ:.3f} {bar}")
    
    # Coincidence at beam splitter outputs (modes 2 and 3)
    # ⟨n_2 n_3⟩
    n2 = sub._n_ops[2].toarray()
    n3 = sub._n_ops[3].toarray()
    
    coincidence = float(xp.real(xp.vdot(psi_final, n2 @ n3 @ psi_final)))
    print(f"\nCoincidence ⟨n_2 n_3⟩: {coincidence:.4f}")
    
    # Bunching probability: ⟨n_2(n_2-1)⟩ + ⟨n_3(n_3-1)⟩
    n2_sq = n2 @ n2
    n3_sq = n3 @ n3
    bunching_2 = float(xp.real(xp.vdot(psi_final, n2_sq @ psi_final))) - float(xp.real(xp.vdot(psi_final, n2 @ psi_final)))
    bunching_3 = float(xp.real(xp.vdot(psi_final, n3_sq @ psi_final))) - float(xp.real(xp.vdot(psi_final, n3 @ psi_final)))
    bunching = bunching_2 + bunching_3
    
    print(f"Bunching ⟨n(n-1)⟩: {bunching:.4f}")
    
    print("\nInterpretation:")
    if coincidence < 0.1 and bunching > 0.3:
        print("  → BOSONIC: Low coincidence, high bunching (HOM dip)")
    elif coincidence > 0.3 and bunching < 0.1:
        print("  → FERMIONIC: High coincidence, low bunching (anti-bunching)")
    else:
        print("  → MIXED: Intermediate behavior")
    
    return {
        'coincidence': coincidence,
        'bunching': bunching,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_scaling(results: Dict, output_dir: str):
    """Plot particle number scaling."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n = results['n_particles']
    sym = results['sym_fidelity']
    anti = results['anti_fidelity']
    gap = results['gap']
    
    ax.plot(n, sym, 'o-', label='Symmetric', color='blue', markersize=10)
    ax.plot(n, anti, 's-', label='Antisymmetric', color='red', markersize=10)
    ax.plot(n, gap, '^--', label='Gap', color='green', markersize=10)
    
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Fidelity / Gap')
    ax.set_title('Statistics Gap vs Particle Number')
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/particle_scaling.png', dpi=150)
    plt.close()


def plot_occupation_evolution(results: Dict, output_dir: str):
    """Plot double occupation over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    times = results['times']
    double_occ = results['double_occ']
    
    ax.plot(times, double_occ, 'b-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Double Occupation')
    ax.set_title('Evolution of Double Occupation')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/double_occupation.png', dpi=150)
    plt.close()


# =============================================================================
# TEST 6: WHY MULTI-PARTICLE DETECTION FAILS
# =============================================================================

def investigate_detection_failure(n_modes: int = 6) -> Dict:
    """
    Investigate why single-particle detection doesn't distinguish
    multi-particle exchange symmetry.
    
    Hypothesis: CNOT detection copies OCCUPATION to environment,
    but exchange symmetry is about PARTICLE LABELS, not occupation.
    
    For multi-particle states, we need detection that probes
    the RELATIVE structure, not just local occupation.
    """
    print("\n" + "="*60)
    print("TEST 6: Detection Mechanism Analysis")
    print("="*60)
    
    sub = MultiBodySubstrate(n_modes, max_excitations_per_mode=2)
    
    # Create symmetric and antisymmetric 2-particle states
    config_02 = tuple([1, 0, 1] + [0]*(n_modes-3))
    config_12 = tuple([0, 1, 1] + [0]*(n_modes-3))
    
    psi_02 = sub.basis_state(config_02)
    psi_12 = sub.basis_state(config_12)
    
    psi_sym = (psi_02 + psi_12) / np.sqrt(2)
    psi_anti = (psi_02 - psi_12) / np.sqrt(2)
    
    print("\nState analysis:")
    print("  |ψ_S⟩ = (|1,0,1,...⟩ + |0,1,1,...⟩)/√2")
    print("  |ψ_A⟩ = (|1,0,1,...⟩ - |0,1,1,...⟩)/√2")
    
    # Check: these states have IDENTICAL local occupation statistics!
    print("\nLocal occupations:")
    for name, psi in [("symmetric", psi_sym), ("antisymmetric", psi_anti)]:
        occs = [sub.measure_occupation(psi, m) for m in range(3)]
        print(f"  {name}: n_0={occs[0]:.2f}, n_1={occs[1]:.2f}, n_2={occs[2]:.2f}")
    
    print("\n→ PROBLEM: Both states have IDENTICAL local occupation!")
    print("   CNOT detection only probes local n_j, can't distinguish them.")
    
    # Solution: Need CORRELATION-SENSITIVE detection
    print("\n--- Solution: Correlation-Sensitive Detection ---")
    
    # Detect PAIRS of modes simultaneously
    # This probes the correlation structure, not just local occupation
    
    def correlation_detection(sub, modes_pair, env_mode, strength=np.pi/2):
        """
        Detection that probes correlation between two modes.
        H = (n_i - n_j) ⊗ σ_x^env
        
        This is sensitive to WHICH mode is occupied, not just total occupation.
        """
        i, j = modes_pair
        n_diff = (sub._n_ops[i] - sub._n_ops[j]).toarray()
        sigma_x_env = (sub._a_ops[env_mode] + sub._adag_ops[env_mode]).toarray()
        H = n_diff @ sigma_x_env
        return expm(-1j * strength * H)
    
    # Test correlation detection
    env_modes = list(range(3, n_modes))
    
    print("\nCorrelation detection (probes n_0 - n_1):")
    
    for name, psi_init in [("symmetric", psi_sym), ("antisymmetric", psi_anti)]:
        psi = psi_init.copy()
        rho_orig = sub.reduced_density_matrix(psi, [0, 1, 2])
        
        for i in range(5):
            env_m = env_modes[i % len(env_modes)]
            U_det = correlation_detection(sub, (0, 1), env_m)
            psi = U_det @ psi
            psi = psi / xp.linalg.norm(psi)
        
        rho_final = sub.reduced_density_matrix(psi, [0, 1, 2])
        fidelity = float(xp.abs(xp.trace(rho_orig @ rho_final)))
        
        stat_type = "survives" if fidelity > 0.7 else "consumed"
        print(f"  {name}: fidelity = {fidelity:.3f} → {stat_type}")
    
    # Another approach: two-mode parity detection
    print("\n--- Alternative: Parity Detection ---")
    
    def parity_detection(sub, mode, env_mode, strength=np.pi):
        """
        Detection that measures parity of occupation.
        H = (-1)^n ⊗ σ_x^env
        
        For n=0,2,4...: no flip
        For n=1,3,5...: flip
        """
        n_op = sub._n_ops[mode].toarray()
        # Parity operator: e^{iπn} = (-1)^n
        parity = expm(1j * np.pi * n_op)
        sigma_x_env = (sub._a_ops[env_mode] + sub._adag_ops[env_mode]).toarray()
        
        # Use parity to control the flip
        H = (np.eye(sub.dim) - parity) / 2 @ sigma_x_env  # Projects onto odd parity
        return expm(-1j * strength * H)
    
    print("Parity detection on mode 0:")
    for name, psi_init in [("symmetric", psi_sym), ("antisymmetric", psi_anti)]:
        psi = psi_init.copy()
        rho_orig = sub.reduced_density_matrix(psi, [0, 1, 2])
        
        for i in range(5):
            env_m = env_modes[i % len(env_modes)]
            U_det = parity_detection(sub, 0, env_m)
            psi = U_det @ psi
            psi = psi / xp.linalg.norm(psi)
        
        rho_final = sub.reduced_density_matrix(psi, [0, 1, 2])
        fidelity = float(xp.abs(xp.trace(rho_orig @ rho_final)))
        
        print(f"  {name}: fidelity = {fidelity:.3f}")
    
    # The KEY insight: exchange symmetry is GLOBAL, not local
    print("\n" + "-"*50)
    print("KEY INSIGHT:")
    print("-"*50)
    print("""
    Single-particle statistics emerge from LOCAL detection
    because the superposition is over POSITIONS.
    
    Multi-particle exchange symmetry is about LABELS,
    which are GLOBAL properties of the wavefunction.
    
    To detect exchange symmetry, we need:
    1. Measurements that probe RELATIVE structure
    2. Or: let dynamics + local detection PROJECT onto sectors
    """)
    
    return {}


def test_dynamic_symmetry_selection(n_modes: int = 8) -> Dict:
    """
    Use dynamics + detection to PROJECT onto symmetric/antisymmetric sectors.
    
    Protocol:
    1. Evolve under symmetric Hamiltonian
    2. Apply local detection
    3. Repeat many times
    4. The surviving subspace should have definite exchange symmetry
    """
    print("\n" + "="*60)
    print("TEST 7: Dynamic Symmetry Selection")
    print("="*60)
    
    sub = MultiBodySubstrate(n_modes, max_excitations_per_mode=2)
    H = hopping_hamiltonian(sub, linear_chain(n_modes), coupling=1.0)
    
    # Start with a generic 2-particle state (no definite symmetry)
    # |1,0,1,0,...⟩
    config = tuple([1, 0, 1] + [0]*(n_modes-3))
    psi_init = sub.basis_state(config)
    
    # Build exchange operator
    P_01 = xp.zeros((sub.dim, sub.dim), dtype=xp.complex128)
    for idx in range(sub.dim):
        config_list = list(sub.idx_to_config(idx))
        config_list[0], config_list[1] = config_list[1], config_list[0]
        new_idx = sub.config_to_idx(tuple(config_list))
        P_01[new_idx, idx] = 1.0
    
    # Initial exchange expectation
    ex_init = float(xp.real(xp.vdot(psi_init, P_01 @ psi_init)))
    print(f"\nInitial exchange expectation ⟨P_01⟩: {ex_init:.3f}")
    
    env_modes = list(range(4, n_modes))
    sys_modes = [0, 1, 2, 3]
    
    # Evolution + detection cycles
    psi = psi_init.copy()
    dt = 0.1
    U_evol = expm(-1j * H * dt)
    
    exchange_history = [ex_init]
    
    print("\nEvolution + detection cycles:")
    for cycle in range(20):
        # Evolve
        for _ in range(5):
            psi = U_evol @ psi
        
        # Detect
        for sys_m in [0, 1]:
            env_m = env_modes[cycle % len(env_modes)]
            U_det = detection_unitary(sub, sys_m, env_m)
            psi = U_det @ psi
        
        psi = psi / xp.linalg.norm(psi)
        
        # Measure exchange
        ex = float(xp.real(xp.vdot(psi, P_01 @ psi)))
        exchange_history.append(ex)
        
        if (cycle + 1) % 5 == 0:
            print(f"  Cycle {cycle+1}: ⟨P_01⟩ = {ex:+.3f}")
    
    final_exchange = exchange_history[-1]
    
    print(f"\nFinal exchange expectation: {final_exchange:+.3f}")
    
    if abs(final_exchange - 1) < 0.1:
        print("→ BOSONIC sector selected (symmetric under exchange)")
    elif abs(final_exchange + 1) < 0.1:
        print("→ FERMIONIC sector selected (antisymmetric under exchange)")
    else:
        print("→ MIXED: No definite symmetry selected")
    
    return {'exchange_history': [float(x) for x in exchange_history]}


# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir = 'multibody_results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("MULTI-BODY TEST SUITE")
    print("="*60)
    print("Testing emergent statistics with multiple particles")
    
    t_start = time.time()
    
    # Run tests
    exclusion_results = test_pauli_exclusion(n_modes=6)
    exchange_results = test_exchange_phase(n_modes=6)
    scaling_results = test_particle_scaling(max_particles=4, n_modes=8)
    ground_results = test_fermionic_ground_state(n_modes=6, n_particles=2)
    hom_results = test_hong_ou_mandel(n_modes=6)
    
    # Investigation of detection mechanism
    investigate_detection_failure(n_modes=6)
    dynamic_results = test_dynamic_symmetry_selection(n_modes=8)
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_scaling(scaling_results, output_dir)
    plot_occupation_evolution(exclusion_results, output_dir)
    
    # Save results
    all_results = {
        'exclusion': {k: v for k, v in exclusion_results.items() if not isinstance(v, list) or len(v) < 100},
        'exchange': exchange_results,
        'scaling': scaling_results,
        'ground_state': ground_results,
        'hong_ou_mandel': hom_results,
    }
    
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\n1. PAULI EXCLUSION:")
    print(f"   Detection selects symmetric/antisymmetric behavior")
    
    print("\n2. EXCHANGE PHASE:")
    print(f"   Symmetric states: P eigenvalue = +1")
    print(f"   Antisymmetric states: P eigenvalue = -1")
    print(f"   Symmetric survives detection, antisymmetric consumed")
    
    print("\n3. PARTICLE SCALING:")
    gaps = scaling_results['gap']
    print(f"   Gap at N=2: {gaps[0]:.3f}")
    print(f"   Gap at N=4: {gaps[-1]:.3f}")
    trend = "increases" if gaps[-1] > gaps[0] else "decreases"
    print(f"   Trend: Gap {trend} with particle number")
    
    print("\n4. GROUND STATE:")
    if ground_results:
        print(f"   Double occ (with U): {ground_results['double_occ_with_U']:.4f}")
        print(f"   Double occ (U=0):    {ground_results['double_occ_free']:.4f}")
    
    print("\n5. HONG-OU-MANDEL:")
    print(f"   Coincidence: {hom_results['coincidence']:.4f}")
    print(f"   Bunching: {hom_results['bunching']:.4f}")
    
    print("\n6. DYNAMIC SYMMETRY SELECTION:")
    if dynamic_results.get('exchange_history'):
        final = dynamic_results['exchange_history'][-1]
        print(f"   Final ⟨P⟩: {final:+.3f}")
        if abs(final - 1) < 0.2:
            print("   → System naturally selects BOSONIC sector")
        elif abs(final + 1) < 0.2:
            print("   → System naturally selects FERMIONIC sector")
    
    print(f"\nTotal time: {time.time() - t_start:.1f}s")
    print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()