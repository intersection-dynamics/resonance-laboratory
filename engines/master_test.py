#!/usr/bin/env python3
"""
Hilbert Substrate Framework - Test Suite
Ben Bray, 2025

Usage: python master_test.py [--section NAME]
"""

import os
import json
import argparse
import numpy as np
from scipy.linalg import expm

OUTPUT_DIR = 'substrate_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


class Substrate:
    """Core substrate: qubits on a lattice."""
    
    def __init__(self, n_sites: int):
        self.n_sites = n_sites
        self.dim = 2 ** n_sites
        self._build_operators()
    
    def _build_operators(self):
        self._sx, self._sy, self._sz, self._n = [], [], [], []
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        n_op = np.array([[0, 0], [0, 1]], dtype=complex)
        
        for site in range(self.n_sites):
            for op_single, op_list in [(sx, self._sx), (sy, self._sy), 
                                        (sz, self._sz), (n_op, self._n)]:
                ops = [np.eye(2) for _ in range(self.n_sites)]
                ops[site] = op_single
                full = ops[0]
                for i in range(1, self.n_sites):
                    full = np.kron(full, ops[i])
                op_list.append(full)
    
    def excitation(self, site: int) -> np.ndarray:
        config = [0] * self.n_sites
        config[site] = 1
        idx = sum(config[s] * (2 ** (self.n_sites - 1 - s)) for s in range(self.n_sites))
        psi = np.zeros(self.dim, dtype=complex)
        psi[idx] = 1.0
        return psi
    
    def measure(self, psi: np.ndarray, site: int) -> float:
        return float(np.real(np.vdot(psi, self._n[site] @ psi)))


def hopping_H(sub: Substrate, t: float = 1.0) -> np.ndarray:
    H = np.zeros((sub.dim, sub.dim), dtype=complex)
    for s in range(sub.n_sites - 1):
        H -= t * (sub._sx[s] @ sub._sx[s+1] + sub._sy[s] @ sub._sy[s+1]) / 2
    return H


# =============================================================================
# TESTS
# =============================================================================

def test_spacetime(n_sites=8):
    """Measure light cone from arrival times."""
    sub = Substrate(n_sites)
    H = hopping_H(sub)
    psi = sub.excitation(0)
    
    threshold = 0.01
    times = np.linspace(0, n_sites, 100)
    dt = times[1] - times[0]
    U = expm(-1j * H * dt)
    
    arrival = [None] * n_sites
    for t in times:
        for site in range(n_sites):
            if arrival[site] is None and sub.measure(psi, site) > threshold:
                arrival[site] = float(t)
        psi = U @ psi
        psi /= np.linalg.norm(psi)
    
    valid = [(s, t) for s, t in enumerate(arrival) if t is not None and s > 0]
    velocity = np.polyfit([t for s,t in valid], [s for s,t in valid], 1)[0] if len(valid) >= 2 else None
    
    return {'arrival_times': [round(t, 3) if t else None for t in arrival], 'velocity': round(velocity, 3)}


def test_gauge(n_sites=4):
    """Test flux attachment statistics transmutation."""
    n_matter, n_links = n_sites, n_sites - 1
    n_total = n_matter + n_links
    dim = 2 ** n_total
    
    def site_op(op, site):
        ops = [np.eye(2) for _ in range(n_total)]
        ops[site] = op
        result = ops[0]
        for i in range(1, n_total):
            result = np.kron(result, ops[i])
        return result
    
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    link_phase = [site_op(sz, n_matter + l) for l in range(n_links)]
    
    def basis(matter, links):
        config = matter + links
        idx = sum(config[s] * (2 ** (n_total - 1 - s)) for s in range(n_total))
        psi = np.zeros(dim, dtype=complex)
        psi[idx] = 1.0
        return psi
    
    P = np.zeros((dim, dim), dtype=complex)
    for idx in range(dim):
        config = [(idx >> (n_total - 1 - m)) & 1 for m in range(n_total)]
        config[0], config[1] = config[1], config[0]
        new_idx = sum(config[m] << (n_total - 1 - m) for m in range(n_total))
        P[new_idx, idx] = 1.0
    
    P_gauge = P @ link_phase[0]
    
    psi_02 = basis([1,0,1,0], [0,0,0])
    psi_12 = basis([0,1,1,0], [0,0,0])
    psi_sym = (psi_02 + psi_12) / np.sqrt(2)
    psi_anti = (psi_02 - psi_12) / np.sqrt(2)
    
    psi_02_f = basis([1,0,1,0], [1,0,0])
    psi_12_f = basis([0,1,1,0], [1,0,0])
    psi_sym_f = (psi_02_f + psi_12_f) / np.sqrt(2)
    psi_anti_f = (psi_02_f - psi_12_f) / np.sqrt(2)
    
    return {
        'no_flux': {
            'symmetric': round(float(np.real(np.vdot(psi_sym, P @ psi_sym))), 1),
            'antisymmetric': round(float(np.real(np.vdot(psi_anti, P @ psi_anti))), 1)
        },
        'with_flux': {
            'symmetric': round(float(np.real(np.vdot(psi_sym_f, P_gauge @ psi_sym_f))), 1),
            'antisymmetric': round(float(np.real(np.vdot(psi_anti_f, P_gauge @ psi_anti_f))), 1)
        }
    }


def test_qed(n_sites=4):
    """Test photon-fermion scattering."""
    n_matter, n_links = n_sites, n_sites - 1
    matter_dim, photon_dim = 2 ** n_matter, 2 ** n_links
    dim = matter_dim * photon_dim
    
    a = np.array([[0,1],[0,0]], dtype=complex)
    n_op = np.array([[0,0],[0,1]], dtype=complex)
    
    def m_op(op, site):
        ops = [np.eye(2) for _ in range(n_matter)]
        ops[site] = op
        m = ops[0]
        for i in range(1, n_matter): m = np.kron(m, ops[i])
        return np.kron(m, np.eye(photon_dim))
    
    def p_op(op, link):
        ops = [np.eye(2) for _ in range(n_links)]
        ops[link] = op
        p = ops[0]
        for i in range(1, n_links): p = np.kron(p, ops[i])
        return np.kron(np.eye(matter_dim), p)
    
    matter_n = [m_op(n_op, s) for s in range(n_matter)]
    photon_n = [p_op(n_op, l) for l in range(n_links)]
    photon_a = [p_op(a, l) for l in range(n_links)]
    
    H = np.zeros((dim, dim), dtype=complex)
    for s in range(n_matter): H += 0.3 * matter_n[s]
    for l in range(n_links):
        H += 0.3 * photon_n[l]
        if l < n_links - 1:
            H -= 1.0 * (photon_a[l].T.conj() @ photon_a[l+1] + photon_a[l+1].T.conj() @ photon_a[l])
    for s in range(n_matter):
        for l in [s-1, s]:
            if 0 <= l < n_links:
                H += 0.5 * matter_n[s] @ (photon_a[l] + photon_a[l].T.conj())
    
    def basis(m_cfg, p_cfg):
        m_idx = sum(m_cfg[s] << (n_matter - 1 - s) for s in range(n_matter))
        p_idx = sum(p_cfg[l] << (n_links - 1 - l) for l in range(n_links))
        psi = np.zeros(dim, dtype=complex)
        psi[m_idx * photon_dim + p_idx] = 1.0
        return psi
    
    psi = basis([0,0,1,0], [1,0,0])
    photon_init = [float(np.real(np.vdot(psi, photon_n[l] @ psi))) for l in range(n_links)]
    
    U = expm(-1j * H * 0.1)
    for _ in range(100):
        psi = U @ psi
        psi /= np.linalg.norm(psi)
    
    photon_final = [round(float(np.real(np.vdot(psi, photon_n[l] @ psi))), 3) for l in range(n_links)]
    
    return {
        'initial': [1.0, 0.0, 0.0],
        'final': photon_final,
        'absorbed': round(1.0 - photon_final[0], 3),
        'scattered': round(sum(photon_final[1:]), 3)
    }


def test_qcd():
    """Test color confinement."""
    dim = 64
    
    c3 = np.diag([0, 1, -1, 0]).astype(complex)
    c8 = np.diag([0, 1, 1, -2]).astype(complex) / np.sqrt(3)
    nq = np.diag([0, 1, 1, 1]).astype(complex)
    
    def site_op(op, site):
        ops = [np.eye(4), np.eye(4), np.eye(4)]
        ops[site] = op
        return np.kron(np.kron(ops[0], ops[1]), ops[2])
    
    color_3 = [site_op(c3, s) for s in range(3)]
    color_8 = [site_op(c8, s) for s in range(3)]
    n_quark = [site_op(nq, s) for s in range(3)]
    
    H = np.zeros((dim, dim), dtype=complex)
    for s in range(3): H += 0.3 * n_quark[s]
    for s in range(2): H -= 1.0 * (color_3[s] @ color_3[s+1] + color_8[s] @ color_8[s+1])
    t3, t8 = sum(color_3), sum(color_8)
    H += 5.0 * (t3 @ t3 + t8 @ t8)
    
    def basis(cfg):
        psi = np.zeros(dim, dtype=complex)
        psi[cfg[0]*16 + cfg[1]*4 + cfg[2]] = 1.0
        return psi
    
    def baryon():
        perms = [([1,2,3], +1), ([2,3,1], +1), ([3,1,2], +1),
                 ([1,3,2], -1), ([3,2,1], -1), ([2,1,3], -1)]
        psi = np.zeros(dim, dtype=complex)
        for cfg, sign in perms:
            psi[cfg[0]*16 + cfg[1]*4 + cfg[2]] = sign / np.sqrt(6)
        return psi
    
    def casimir(psi):
        return float(np.real(np.vdot(psi, (t3@t3 + t8@t8) @ psi)))
    
    results = {}
    for name, psi in [('single_R', basis([1,0,0])), ('triple_RRR', basis([1,1,1])), ('baryon_RGB', baryon())]:
        E = float(np.real(np.vdot(psi, H @ psi)))
        C = casimir(psi)
        results[name] = {'energy': round(E, 2), 'casimir': round(C, 3)}
    
    return results


def test_weak():
    """Test parity violation from internal chirality."""
    dim = 25
    
    W_plus = np.zeros((5,5), dtype=complex)
    W_plus[1, 2] = 1.0
    
    def site_op(op, site):
        ops = [np.eye(5), np.eye(5)]
        ops[site] = op
        return np.kron(ops[0], ops[1])
    
    W_p = [site_op(W_plus, s) for s in range(2)]
    W_m = [site_op(W_plus.T.conj(), s) for s in range(2)]
    
    H = 0.7 * (W_p[0] @ W_m[1] + W_m[0] @ W_p[1])
    
    def basis(cfg):
        state_map = {'uL': 1, 'dL': 2, 'uR': 3, 'dR': 4}
        vals = [state_map[c] for c in cfg]
        psi = np.zeros(dim, dtype=complex)
        psi[vals[0] * 5 + vals[1]] = 1.0
        return psi
    
    return {
        'left_coupling': round(float(np.real(np.vdot(basis(['dL','uL']), H @ basis(['uL','dL'])))), 3),
        'right_coupling': round(float(np.real(np.vdot(basis(['dR','uR']), H @ basis(['uR','dR'])))), 3)
    }


def test_mass(n_sites=6):
    """Test mass as localization cost."""
    sub = Substrate(n_sites)
    H_kin = hopping_H(sub)
    center = n_sites // 2
    
    results = {}
    for mass in [0.0, 0.5, 2.0]:
        H_loc = np.zeros((sub.dim, sub.dim), dtype=complex)
        for s in range(n_sites):
            H_loc += mass * (s - center)**2 * sub._n[s]
        
        H = H_kin + H_loc
        evals, evecs = np.linalg.eigh(H)
        psi = evecs[:, 0]
        
        probs = np.abs(psi) ** 2
        ipr = float(np.sum(probs ** 2))
        
        results[f'm={mass}'] = {'energy': round(evals[0], 3), 'ipr': round(ipr, 3)}
    
    return results


def test_higgs(n_sites=4):
    """Test Yukawa coupling -> mass."""
    sub = Substrate(n_sites)
    H_kin = hopping_H(sub)
    
    evals_0, _ = np.linalg.eigh(H_kin)
    E_0 = evals_0[0]
    
    results = {}
    for name, y in [('neutrino', 0.001), ('electron', 0.01), ('muon', 0.1), ('top', 1.0)]:
        H_mass = np.zeros((sub.dim, sub.dim), dtype=complex)
        for s in range(n_sites):
            H_mass += y * sub._n[s]
        
        evals, _ = np.linalg.eigh(H_kin + H_mass)
        results[name] = {'yukawa': y, 'mass': round(evals[0] - E_0, 4)}
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def run_all(section=None):
    tests = [
        ('spacetime', test_spacetime),
        ('gauge', test_gauge),
        ('qed', test_qed),
        ('qcd', test_qcd),
        ('weak', test_weak),
        ('mass', test_mass),
        ('higgs', test_higgs),
    ]
    
    results = {}
    
    print("HILBERT SUBSTRATE FRAMEWORK")
    print("="*50)
    
    for name, func in tests:
        if section and section != name:
            continue
        try:
            results[name] = func()
            print(f"\n[{name.upper()}]")
            print(json.dumps(results[name], indent=2))
        except Exception as e:
            results[name] = {'error': str(e)}
            print(f"\n[{name.upper()}] ERROR: {e}")
    
    with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Results: {OUTPUT_DIR}/results.json")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--section', type=str, help='Run specific section')
    args = parser.parse_args()
    run_all(section=args.section)