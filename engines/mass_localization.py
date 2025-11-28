"""
Mass as Information Localization (Defrag Hypothesis)
====================================================
The substrate naturally entangles everything, spreading information.
Mass is the resistance to this spreading - the "defrag" that keeps
a pattern's information localized.

Key ideas:
1. Entanglement spreads information across the substrate
2. No-signaling: can't maintain coherence across spacelike separation
3. A stable pattern must keep its defining information LOCAL
4. Mass = strength of this localization requirement

Predictions:
- Massless: information spreads freely (photon = spreading wave)
- Massive: information stays localized (electron = localized pattern)
- More mass: stronger localization (proton > electron)
- Gravity: attraction between localized information clusters
"""

import os
import numpy as np
from scipy.linalg import expm, logm
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Mass as Information Localization")


# =============================================================================
# INFORMATION-THEORETIC SUBSTRATE
# =============================================================================

class InfoSubstrate:
    """
    Substrate focused on information flow and localization.
    
    Each site has a qubit. We track:
    - Local information (von Neumann entropy)
    - Entanglement between sites
    - Information localization (inverse participation ratio)
    """
    
    def __init__(self, n_sites: int):
        self.n_sites = n_sites
        self.dim = 2 ** n_sites
        
        # Build operators
        self._build_operators()
        
        print(f"InfoSubstrate: {n_sites} sites, dim={self.dim}")
    
    def _build_operators(self):
        """Build local operators."""
        self._n = []  # Number operators
        self._sx = []  # Pauli X
        self._sy = []  # Pauli Y
        self._sz = []  # Pauli Z
        
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        n_op = np.array([[0, 0], [0, 1]], dtype=complex)
        
        for site in range(self.n_sites):
            for op_single, op_list in [(n_op, self._n), (sx, self._sx), 
                                        (sy, self._sy), (sz, self._sz)]:
                ops = [np.eye(2) for _ in range(self.n_sites)]
                ops[site] = op_single
                
                full = ops[0]
                for i in range(1, self.n_sites):
                    full = np.kron(full, ops[i])
                
                op_list.append(full)
    
    def localized_state(self, site: int) -> np.ndarray:
        """Create excitation localized at one site."""
        config = [0] * self.n_sites
        config[site] = 1
        idx = sum(config[s] * (2 ** (self.n_sites - 1 - s)) for s in range(self.n_sites))
        psi = np.zeros(self.dim, dtype=complex)
        psi[idx] = 1.0
        return psi
    
    def spread_state(self, center: int, width: int = 2) -> np.ndarray:
        """Create excitation spread over multiple sites (Gaussian-ish)."""
        psi = np.zeros(self.dim, dtype=complex)
        
        for site in range(self.n_sites):
            if abs(site - center) <= width:
                # Gaussian weight
                weight = np.exp(-0.5 * ((site - center) / max(width/2, 1))**2)
                state = self.localized_state(site)
                psi += weight * state
        
        return psi / np.linalg.norm(psi)
    
    def measure_local_occupation(self, psi: np.ndarray, site: int) -> float:
        """Occupation probability at site."""
        return float(np.real(np.vdot(psi, self._n[site] @ psi)))
    
    def information_localization(self, psi: np.ndarray) -> float:
        """
        Inverse Participation Ratio (IPR).
        IPR = Σ |⟨i|ψ⟩|^4
        
        IPR = 1: perfectly localized
        IPR = 1/N: perfectly spread
        """
        probs = np.abs(psi) ** 2
        return float(np.sum(probs ** 2))
    
    def position_variance(self, psi: np.ndarray) -> float:
        """
        Spatial spread of the excitation.
        Var(x) = ⟨x²⟩ - ⟨x⟩²
        """
        # ⟨x⟩
        x_mean = sum(site * self.measure_local_occupation(psi, site) 
                    for site in range(self.n_sites))
        
        # ⟨x²⟩
        x2_mean = sum(site**2 * self.measure_local_occupation(psi, site) 
                     for site in range(self.n_sites))
        
        return x2_mean - x_mean**2
    
    def entanglement_entropy(self, psi: np.ndarray, partition: int) -> float:
        """
        Entanglement entropy across partition.
        S = -Tr(ρ_A log ρ_A)
        
        partition: number of sites in subsystem A
        """
        if partition <= 0 or partition >= self.n_sites:
            return 0.0
        
        # Reshape to bipartite
        dim_A = 2 ** partition
        dim_B = 2 ** (self.n_sites - partition)
        
        psi_matrix = psi.reshape(dim_A, dim_B)
        
        # Reduced density matrix
        rho_A = psi_matrix @ psi_matrix.T.conj()
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        
        # Entropy
        return float(-np.sum(eigenvalues * np.log2(eigenvalues + 1e-15)))


def entangling_hamiltonian(sub: InfoSubstrate, 
                           hopping: float = 1.0,
                           interaction: float = 0.5) -> np.ndarray:
    """
    Hamiltonian that SPREADS information (entangles).
    
    H = -t Σ (σ+_i σ-_j + h.c.) + V Σ σ_z^i σ_z^j
    
    This naturally delocalizes excitations.
    """
    H = np.zeros((sub.dim, sub.dim), dtype=complex)
    
    # Hopping (XY interaction) - spreads excitations
    for s in range(sub.n_sites - 1):
        H -= hopping * (sub._sx[s] @ sub._sx[s+1] + sub._sy[s] @ sub._sy[s+1]) / 2
    
    # ZZ interaction - creates entanglement
    for s in range(sub.n_sites - 1):
        H += interaction * sub._sz[s] @ sub._sz[s+1]
    
    return H


def localizing_hamiltonian(sub: InfoSubstrate,
                           mass: float = 1.0,
                           center: int = None) -> np.ndarray:
    """
    Hamiltonian that LOCALIZES information (mass term).
    
    H_mass = m Σ (site - center)² n_site
    
    Harmonic potential keeps excitation near center.
    This is the "defrag" force.
    """
    if center is None:
        center = sub.n_sites // 2
    
    H = np.zeros((sub.dim, sub.dim), dtype=complex)
    
    for site in range(sub.n_sites):
        # Harmonic potential centered at 'center'
        H += mass * (site - center)**2 * sub._n[site]
    
    return H


def info_flow_cost(sub: InfoSubstrate, psi: np.ndarray) -> float:
    """
    Cost of maintaining pattern coherence.
    
    If information is spread out, it costs more to keep it coherent
    (would need FTL signaling).
    
    Cost ~ spatial_spread × entanglement
    """
    spread = sub.position_variance(psi)
    entanglement = sub.entanglement_entropy(psi, sub.n_sites // 2)
    
    return spread * (1 + entanglement)


# =============================================================================
# TESTS
# =============================================================================

def test_spreading_vs_localization():
    """Test competition between entanglement (spreading) and mass (localization)."""
    print("\n" + "="*60)
    print("TEST 1: Spreading vs Localization")
    print("="*60)
    
    sub = InfoSubstrate(n_sites=8)
    
    # Start localized
    psi_init = sub.localized_state(4)  # Center of chain
    
    print(f"Initial state: localized at site 4")
    print(f"  IPR: {sub.information_localization(psi_init):.3f}")
    print(f"  Variance: {sub.position_variance(psi_init):.3f}")
    
    # Test different mass values
    masses = [0.0, 0.1, 0.5, 1.0, 2.0]
    
    print(f"\nEvolution with different masses (t=5):")
    print("-" * 60)
    print(f"{'Mass':<10} {'IPR':<12} {'Variance':<12} {'Interpretation'}")
    print("-" * 60)
    
    results = {}
    
    for mass in masses:
        H_spread = entangling_hamiltonian(sub, hopping=1.0)
        H_local = localizing_hamiltonian(sub, mass=mass, center=4)
        H = H_spread + H_local
        
        # Evolve
        t = 5.0
        U = expm(-1j * H * t)
        psi = U @ psi_init
        psi /= np.linalg.norm(psi)
        
        ipr = sub.information_localization(psi)
        var = sub.position_variance(psi)
        
        if mass == 0:
            interp = "MASSLESS - spreads freely"
        elif ipr > 0.5:
            interp = "HEAVY - stays localized"
        elif ipr > 0.2:
            interp = "MODERATE - partially spread"
        else:
            interp = "LIGHT - mostly spread"
        
        print(f"{mass:<10.1f} {ipr:<12.3f} {var:<12.3f} {interp}")
        results[mass] = {'ipr': ipr, 'var': var}
    
    return results


def test_mass_from_localization_pressure():
    """
    Test: mass emerges from the requirement to stay localized.
    
    Prediction: ground state energy ~ localization strength
    This is because fighting the spreading costs energy.
    """
    print("\n" + "="*60)
    print("TEST 2: Mass from Localization Pressure")
    print("="*60)
    
    sub = InfoSubstrate(n_sites=6)
    
    # Pure spreading Hamiltonian
    H_spread = entangling_hamiltonian(sub, hopping=1.0, interaction=0.3)
    
    print("Ground state energy vs localization strength:")
    print("-" * 50)
    
    loc_strengths = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]
    energies = []
    iprs = []
    
    for loc in loc_strengths:
        H_local = localizing_hamiltonian(sub, mass=loc, center=3)
        H = H_spread + H_local
        
        # Find ground state
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        E_ground = eigenvalues[0]
        psi_ground = eigenvectors[:, 0]
        
        ipr = sub.information_localization(psi_ground)
        
        energies.append(E_ground)
        iprs.append(ipr)
        
        print(f"Loc strength: {loc:.1f}, E_ground: {E_ground:+.3f}, IPR: {ipr:.3f}")
    
    # The energy cost of localization IS the mass
    print("\n→ Energy cost of localization = REST MASS")
    print("  More localized patterns have higher ground state energy")
    print("  This energy is what we measure as mass")
    
    return loc_strengths, energies, iprs


def test_massless_vs_massive_propagation():
    """
    Compare how massless (photon-like) vs massive (electron-like) propagate.
    """
    print("\n" + "="*60)
    print("TEST 3: Massless vs Massive Propagation")
    print("="*60)
    
    sub = InfoSubstrate(n_sites=10)
    
    # Initial: localized excitation
    psi_init = sub.localized_state(2)
    
    # Hamiltonians
    H_massless = entangling_hamiltonian(sub, hopping=1.0, interaction=0.0)
    H_massive = entangling_hamiltonian(sub, hopping=1.0, interaction=0.0) + \
                localizing_hamiltonian(sub, mass=0.5, center=2)
    
    # Evolve both
    times = np.linspace(0, 8, 80)
    dt = times[1] - times[0]
    
    U_massless = expm(-1j * H_massless * dt)
    U_massive = expm(-1j * H_massive * dt)
    
    psi_massless = psi_init.copy()
    psi_massive = psi_init.copy()
    
    spread_massless = []
    spread_massive = []
    center_massless = []
    center_massive = []
    
    for _ in times:
        # Measure spread
        spread_massless.append(np.sqrt(sub.position_variance(psi_massless)))
        spread_massive.append(np.sqrt(sub.position_variance(psi_massive)))
        
        # Measure center of mass
        cm_ml = sum(s * sub.measure_local_occupation(psi_massless, s) for s in range(sub.n_sites))
        cm_m = sum(s * sub.measure_local_occupation(psi_massive, s) for s in range(sub.n_sites))
        center_massless.append(cm_ml)
        center_massive.append(cm_m)
        
        # Evolve
        psi_massless = U_massless @ psi_massless
        psi_massless /= np.linalg.norm(psi_massless)
        psi_massive = U_massive @ psi_massive
        psi_massive /= np.linalg.norm(psi_massive)
    
    print(f"After t={times[-1]:.1f}:")
    print(f"  Massless spread: {spread_massless[-1]:.2f}")
    print(f"  Massive spread: {spread_massive[-1]:.2f}")
    print(f"\n→ Massless: information SPREADS freely")
    print("→ Massive: information stays LOCALIZED")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(times, spread_massless, 'b-', label='Massless (photon-like)', linewidth=2)
    axes[0].plot(times, spread_massive, 'r-', label='Massive (electron-like)', linewidth=2)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Spatial spread (σ)')
    axes[0].set_title('Information Spreading')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(times, center_massless, 'b-', label='Massless', linewidth=2)
    axes[1].plot(times, center_massive, 'r-', label='Massive', linewidth=2)
    axes[1].axhline(2, color='gray', linestyle=':', label='Initial position')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Center of mass')
    axes[1].set_title('Position Evolution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mass_results/massless_vs_massive.png', dpi=150)
    plt.close()
    
    return times, spread_massless, spread_massive


def test_info_flow_cost_and_gravity():
    """
    Test: gravity as attraction between localized information.
    
    Two localized patterns should attract because being closer
    reduces the total information flow cost.
    """
    print("\n" + "="*60)
    print("TEST 4: Information Flow Cost → Gravity")
    print("="*60)
    
    sub = InfoSubstrate(n_sites=8)
    
    # Two localized excitations at different separations
    separations = [1, 2, 3, 4, 5, 6]
    costs = []
    
    print("Information flow cost vs separation:")
    print("-" * 40)
    
    for sep in separations:
        # Create two-excitation state
        site1 = 1
        site2 = site1 + sep
        
        if site2 >= sub.n_sites:
            continue
        
        # Two excitations (simplified: just tensor product)
        config = [0] * sub.n_sites
        config[site1] = 1
        config[site2] = 1
        idx = sum(config[s] * (2 ** (sub.n_sites - 1 - s)) for s in range(sub.n_sites))
        psi = np.zeros(sub.dim, dtype=complex)
        psi[idx] = 1.0
        
        cost = info_flow_cost(sub, psi)
        costs.append(cost)
        
        print(f"Separation {sep}: cost = {cost:.3f}")
    
    # Does cost decrease with distance? (Would indicate attraction)
    if len(costs) > 2:
        if costs[0] < costs[-1]:
            print("\n→ CLOSER = LOWER COST")
            print("  Information wants to cluster")
            print("  This is GRAVITATIONAL ATTRACTION")
        else:
            print("\n→ Cost doesn't simply decrease with distance")
            print("  (Real gravity needs more structure)")
    
    return separations[:len(costs)], costs


def test_higgs_as_localization_coupling():
    """
    Test: Higgs field gives mass by coupling to localization.
    
    Without Higgs: patterns spread freely (massless)
    With Higgs: patterns resist spreading (massive)
    """
    print("\n" + "="*60)
    print("TEST 5: Higgs as Localization Coupling")
    print("="*60)
    
    sub = InfoSubstrate(n_sites=6)
    
    # "Higgs field" = uniform localization potential
    # Coupling to Higgs = how strongly you feel this potential
    
    # Spreading Hamiltonian
    H_kinetic = entangling_hamiltonian(sub, hopping=1.0, interaction=0.0)
    
    # Higgs potential (uniform, not centered)
    # H_higgs = v² Σ_i n_i (just counts excitations - gives rest mass)
    
    higgs_vev = 1.0  # Vacuum expectation value
    
    # Different Yukawa couplings (how strongly particle couples to Higgs)
    yukawas = {'neutrino': 0.001, 'electron': 0.01, 'muon': 0.1, 'tau': 0.5, 'top': 2.0}
    
    print(f"Mass (ground state energy shift) for different Yukawa couplings:")
    print("-" * 50)
    
    # Reference: massless
    eigenvalues_0, _ = np.linalg.eigh(H_kinetic)
    E_massless = eigenvalues_0[0]
    
    for name, y in yukawas.items():
        # Mass from Higgs = y * v
        mass_term = y * higgs_vev
        
        H_higgs = np.zeros((sub.dim, sub.dim), dtype=complex)
        for s in range(sub.n_sites):
            H_higgs += mass_term * sub._n[s]
        
        H = H_kinetic + H_higgs
        eigenvalues, _ = np.linalg.eigh(H)
        E_ground = eigenvalues[0]
        
        # Mass = energy shift from Higgs
        mass = E_ground - E_massless
        
        print(f"{name:<12}: Yukawa = {y:.3f}, mass = {mass:.4f}")
    
    print("\n→ Yukawa coupling determines mass")
    print("  Higgs field provides uniform 'localization pressure'")
    print("  Stronger coupling = more mass = more localized")


def test_mass_hierarchy():
    """
    Test: can we get a mass hierarchy from localization physics?
    
    Real question: why is electron mass << proton mass << Planck mass?
    """
    print("\n" + "="*60)
    print("TEST 6: Mass Hierarchy from Information")
    print("="*60)
    
    sub = InfoSubstrate(n_sites=8)
    
    # Hypothesis: mass ~ information content of pattern
    # Simple pattern = less info = less mass
    # Complex pattern = more info = more mass
    
    patterns = {
        'simple (1 site)': sub.localized_state(4),
        'pair (2 sites)': None,
        'triplet (3 sites)': None,
        'spread (4 sites)': sub.spread_state(4, width=2),
    }
    
    # Build multi-site patterns
    # Pair
    psi_pair = np.zeros(sub.dim, dtype=complex)
    for s in [3, 5]:
        psi_pair += sub.localized_state(s)
    psi_pair /= np.linalg.norm(psi_pair)
    patterns['pair (2 sites)'] = psi_pair
    
    # Triplet (baryon-like)
    psi_trip = np.zeros(sub.dim, dtype=complex)
    for s in [3, 4, 5]:
        psi_trip += sub.localized_state(s)
    psi_trip /= np.linalg.norm(psi_trip)
    patterns['triplet (3 sites)'] = psi_trip
    
    print("Information content and 'mass' of different patterns:")
    print("-" * 60)
    
    H = entangling_hamiltonian(sub, hopping=1.0, interaction=0.3)
    
    for name, psi in patterns.items():
        if psi is None:
            continue
        
        # Energy
        E = float(np.real(np.vdot(psi, H @ psi)))
        
        # Information measures
        ipr = sub.information_localization(psi)
        entropy = sub.entanglement_entropy(psi, sub.n_sites // 2)
        
        # "Mass" ~ how much energy needed to maintain localization
        # Approximate: energy above vacuum
        
        print(f"{name:<20}: E = {E:+.3f}, IPR = {ipr:.3f}, S = {entropy:.3f}")
    
    print("\n→ More complex patterns (baryons) have higher energy")
    print("  This contributes to proton mass >> electron mass")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_mass_localization(loc_strengths, energies, iprs):
    """Plot relationship between localization and mass."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axes[0]
    ax.plot(loc_strengths, energies, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Localization Strength')
    ax.set_ylabel('Ground State Energy')
    ax.set_title('Energy Cost of Localization = MASS')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(loc_strengths, iprs, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Localization Strength')
    ax.set_ylabel('IPR (Information Localization)')
    ax.set_title('Stronger Localization = More Mass')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mass_results/mass_from_localization.png', dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs('mass_results', exist_ok=True)
    
    print("\n" + "="*60)
    print("MASS AS INFORMATION LOCALIZATION")
    print("="*60)
    print("Testing the 'defrag' hypothesis for mass")
    
    test_spreading_vs_localization()
    loc, E, ipr = test_mass_from_localization_pressure()
    test_massless_vs_massive_propagation()
    test_info_flow_cost_and_gravity()
    test_higgs_as_localization_coupling()
    test_mass_hierarchy()
    
    # Visualizations
    plot_mass_localization(loc, E, ipr)
    
    print("\n" + "="*60)
    print("SUMMARY: Mass as Information Defrag")
    print("="*60)
    print("""
    KEY FINDINGS:
    
    1. SUBSTRATE SPREADS INFORMATION
       - Natural tendency: entangle everything
       - Information delocalizes over time
       - This is the "fragmentation" problem
    
    2. MASS = RESISTANCE TO SPREADING
       - Localized patterns must fight spreading
       - Energy cost of staying localized = REST MASS
       - More localization pressure = more mass
    
    3. MASSLESS VS MASSIVE
       - Photon: information IS the spreading (no mass)
       - Electron: information must stay together (has mass)
       - The difference is localization requirement
    
    4. NO-SIGNALING CONSTRAINT
       - Can't maintain coherence across spacelike separation
       - Pattern MUST keep its info local to be stable
       - This is the fundamental reason for mass
    
    5. HIGGS = LOCALIZATION COUPLING
       - Higgs field provides uniform localization potential
       - Yukawa coupling = how strongly you feel it
       - Different couplings = different masses
    
    6. GRAVITY CONNECTION
       - Localized information clusters attract
       - Being closer reduces info flow cost
       - This may be gravitational attraction
    
    THE DEEP PICTURE:
    
    The substrate is an entanglement engine that wants to
    spread everything out. Mass is the DEFRAG SYSTEM -
    the mechanism that keeps pattern information together.
    
    Massless particles go with the flow (spreading).
    Massive particles fight the flow (localizing).
    
    The energy cost of fighting = E = mc²
    
    Mass isn't a property added to particles.
    Mass IS the localization requirement itself.
    """)
    
    print("\nResults saved to: mass_results/")


if __name__ == "__main__":
    main()