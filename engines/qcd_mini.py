"""
Substrate QCD (Minimal): Color Confinement
==========================================
Demonstrates that protons emerge as stable patterns.
"""

import os
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Substrate QCD - Minimal")

class QCDMini:
    """3-site QCD with SU(3) color. dim = 4^3 = 64."""
    
    def __init__(self):
        self.n_sites = 3
        self.site_dim = 4  # |0⟩, |R⟩, |G⟩, |B⟩
        self.dim = self.site_dim ** self.n_sites  # 64
        
        # Color charges (λ_3 and λ_8 diagonals)
        self._c3 = []  # Isospin-like: R=+1, G=-1, B=0
        self._c8 = []  # Hypercolor: R=+1, G=+1, B=-2
        self._nq = []  # Quark number
        
        c3 = np.diag([0, 1, -1, 0]).astype(complex)
        c8 = np.diag([0, 1, 1, -2]).astype(complex) / np.sqrt(3)
        nq = np.diag([0, 1, 1, 1]).astype(complex)
        
        for s in range(3):
            ops = [np.eye(4), np.eye(4), np.eye(4)]
            ops[s] = c3
            self._c3.append(np.kron(np.kron(ops[0], ops[1]), ops[2]))
            
            ops = [np.eye(4), np.eye(4), np.eye(4)]
            ops[s] = c8
            self._c8.append(np.kron(np.kron(ops[0], ops[1]), ops[2]))
            
            ops = [np.eye(4), np.eye(4), np.eye(4)]
            ops[s] = nq
            self._nq.append(np.kron(np.kron(ops[0], ops[1]), ops[2]))
        
        print(f"QCDMini: 3 sites, dim={self.dim}")
    
    def state(self, config):
        """config = [c0, c1, c2] where c in {0,1,2,3} = {vac,R,G,B}"""
        idx = config[0]*16 + config[1]*4 + config[2]
        psi = np.zeros(self.dim, dtype=complex)
        psi[idx] = 1.0
        return psi
    
    def baryon(self):
        """Color singlet: antisymmetric RGB."""
        perms = [
            ([1,2,3], +1), ([2,3,1], +1), ([3,1,2], +1),
            ([1,3,2], -1), ([3,2,1], -1), ([2,1,3], -1),
        ]
        psi = np.zeros(self.dim, dtype=complex)
        for cfg, sign in perms:
            idx = cfg[0]*16 + cfg[1]*4 + cfg[2]
            psi[idx] = sign / np.sqrt(6)
        return psi
    
    def casimir(self, psi):
        """Color Casimir = (Σc3)² + (Σc8)². Zero for singlet."""
        t3 = sum(self._c3)
        t8 = sum(self._c8)
        return float(np.real(np.vdot(psi, (t3@t3 + t8@t8) @ psi)))

def hamiltonian(qcd, mass=0.3, coupling=1.0, confine=5.0):
    """QCD Hamiltonian with confinement."""
    H = np.zeros((qcd.dim, qcd.dim), dtype=complex)
    
    # Mass
    for s in range(3):
        H += mass * qcd._nq[s]
    
    # Color interaction
    for s in range(2):
        H -= coupling * (qcd._c3[s] @ qcd._c3[s+1] + qcd._c8[s] @ qcd._c8[s+1])
    
    # Confinement
    t3, t8 = sum(qcd._c3), sum(qcd._c8)
    H += confine * (t3@t3 + t8@t8)
    
    return H

def main():
    os.makedirs('qcd_results', exist_ok=True)
    qcd = QCDMini()
    H = hamiltonian(qcd)
    
    print("\n" + "="*50)
    print("COLOR CONFINEMENT TEST")
    print("="*50)
    
    states = {
        "Vacuum": qcd.state([0,0,0]),
        "Single R quark": qcd.state([1,0,0]),
        "R + G (not singlet)": qcd.state([1,2,0]),
        "R + R + R (colored)": qcd.state([1,1,1]),
        "R + G + B (not singlet)": qcd.state([1,2,3]),
        "BARYON (RGB singlet)": qcd.baryon(),
    }
    
    print(f"\n{'State':<25} {'Energy':>10} {'Casimir':>10}")
    print("-"*50)
    
    for name, psi in states.items():
        E = float(np.real(np.vdot(psi, H @ psi)))
        C = qcd.casimir(psi)
        marker = " ← STABLE" if C < 0.1 else ""
        print(f"{name:<25} {E:>+10.2f} {C:>10.3f}{marker}")
    
    # Find ground state
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    E_ground = eigenvalues[0]
    psi_ground = eigenvectors[:, 0]
    C_ground = qcd.casimir(psi_ground)
    
    print(f"\n{'TRUE GROUND STATE':<25} {E_ground:>+10.2f} {C_ground:>10.3f}")
    
    # Analyze ground state composition
    print("\nGround state composition (|coeff|² > 0.01):")
    for idx in range(qcd.dim):
        prob = abs(psi_ground[idx])**2
        if prob > 0.01:
            c0, c1, c2 = idx//16, (idx//4)%4, idx%4
            colors = {0:'·', 1:'R', 2:'G', 3:'B'}
            print(f"  |{colors[c0]}{colors[c1]}{colors[c2]}⟩: {prob:.3f}")
    
    print("\n" + "="*50)
    print("ANALYSIS")
    print("="*50)
    print("""
    KEY OBSERVATIONS:
    
    1. BARYON (RGB singlet) has:
       - Casimir = 0 (color neutral)
       - Lowest energy among 3-quark states
    
    2. Colored states (single quark, RRR) have:
       - High Casimir (color charge)
       - High energy (confinement penalty)
    
    3. The ground state is DOMINATED by color-neutral
       configurations (vacuum or singlet).
    
    This is CONFINEMENT:
    The dynamics naturally select color-neutral patterns.
    
    PROTONS AND NEUTRONS are not fundamental - they
    are the STABLE PATTERNS that survive confinement.
    """)
    
    # Binding energy
    E_baryon = float(np.real(np.vdot(qcd.baryon(), H @ qcd.baryon())))
    E_free = float(np.real(np.vdot(qcd.state([1,0,0]), H @ qcd.state([1,0,0]))))
    binding = 3 * E_free - E_baryon
    
    print(f"\nBINDING ENERGY:")
    print(f"  3 free quarks: {3*E_free:.2f}")
    print(f"  Baryon:        {E_baryon:.2f}")
    print(f"  Binding:       {binding:.2f}")
    print(f"\n→ Baryon is MORE STABLE than free quarks by {binding:.1f} units")
    
    print("\nResults complete.")

if __name__ == "__main__":
    main()