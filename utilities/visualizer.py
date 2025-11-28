#!/usr/bin/env python3
"""
Substrate Visualizer
====================
3D visualization with:
1. Wavelet decomposition - multi-scale structure
2. Decoherence microscope - varying rates reveal internal structure

Ben Bray, 2025
"""

import os
import numpy as np
from scipy.linalg import expm
from scipy import signal
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

OUTPUT_DIR = 'visualizer_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# SINGLE-EXCITATION SUBSPACE (Memory efficient)
# =============================================================================

class Substrate3D:
    """
    3D lattice in single-excitation subspace.
    Dim = n_sites (not 2^n_sites).
    """
    
    def __init__(self, Lx: int, Ly: int, Lz: int):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.n_sites = Lx * Ly * Lz
        self.dim = self.n_sites  # Single excitation subspace!
        
        self.coords = []
        for z in range(Lz):
            for y in range(Ly):
                for x in range(Lx):
                    self.coords.append((x, y, z))
        
        self.neighbors = self._build_neighbors()
        print(f"Substrate3D: {Lx}x{Ly}x{Lz} = {self.n_sites} sites")
    
    def _build_neighbors(self) -> List[List[int]]:
        neighbors = [[] for _ in range(self.n_sites)]
        for idx, (x, y, z) in enumerate(self.coords):
            if x < self.Lx - 1: neighbors[idx].append(self._idx(x+1, y, z))
            if x > 0: neighbors[idx].append(self._idx(x-1, y, z))
            if y < self.Ly - 1: neighbors[idx].append(self._idx(x, y+1, z))
            if y > 0: neighbors[idx].append(self._idx(x, y-1, z))
            if z < self.Lz - 1: neighbors[idx].append(self._idx(x, y, z+1))
            if z > 0: neighbors[idx].append(self._idx(x, y, z-1))
        return neighbors
    
    def _idx(self, x: int, y: int, z: int) -> int:
        return z * (self.Lx * self.Ly) + y * self.Lx + x
    
    def excitation(self, site: int) -> np.ndarray:
        psi = np.zeros(self.n_sites, dtype=complex)
        psi[site] = 1.0
        return psi
    
    def localized_packet(self, center: Tuple[float, float, float], width: float = 1.0) -> np.ndarray:
        psi = np.zeros(self.n_sites, dtype=complex)
        for site, (x, y, z) in enumerate(self.coords):
            r2 = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
            psi[site] = np.exp(-r2 / (2 * width**2))
        return psi / np.linalg.norm(psi)
    
    def density(self, psi: np.ndarray) -> np.ndarray:
        return np.abs(psi)**2
    
    def hamiltonian(self, hopping: float = 1.0) -> np.ndarray:
        H = np.zeros((self.n_sites, self.n_sites), dtype=complex)
        for site, neighs in enumerate(self.neighbors):
            for neigh in neighs:
                H[site, neigh] = -hopping
        return H
    
    def density_matrix(self, psi: np.ndarray) -> np.ndarray:
        return np.outer(psi, psi.conj())


# =============================================================================
# WAVELET ANALYZER
# =============================================================================

class WaveletAnalyzer:
    """Multi-scale wavelet decomposition."""
    
    def __init__(self, substrate: Substrate3D):
        self.sub = substrate
    
    def decompose(self, psi: np.ndarray, n_scales: int = 3) -> Dict:
        density = self.sub.density(psi)
        density_3d = density.reshape(self.sub.Lz, self.sub.Ly, self.sub.Lx)
        
        scales = []
        current = density_3d.copy()
        
        for scale in range(n_scales):
            if min(current.shape) < 2:
                break
            
            # Low-pass (average)
            kernel = np.ones((2, 2, 2)) / 8 if min(current.shape) >= 2 else np.ones((1,1,1))
            smooth = signal.convolve(current, kernel, mode='same')
            
            # High-pass (detail)
            detail = current - smooth
            
            scales.append({
                'scale': scale,
                'detail': detail.copy(),
                'energy': float(np.sum(detail**2)),
                'smooth': smooth.copy()
            })
            
            current = smooth
        
        scales.append({
            'scale': len(scales),
            'approximation': current,
            'energy': float(np.sum(current**2))
        })
        
        return {'scales': scales, 'total_energy': float(np.sum(density**2)), 'density_3d': density_3d}
    
    def spectral(self, psi: np.ndarray) -> Dict:
        density = self.sub.density(psi).reshape(self.sub.Lz, self.sub.Ly, self.sub.Lx)
        fft = np.fft.fftn(density)
        power = np.abs(fft)**2
        
        center = np.array(density.shape) // 2
        max_k = min(center) if min(center) > 0 else 1
        
        radial = []
        for k in range(max_k):
            shell_power = []
            for z in range(power.shape[0]):
                for y in range(power.shape[1]):
                    for x in range(power.shape[2]):
                        r = np.sqrt((z-center[0])**2 + (y-center[1])**2 + (x-center[2])**2)
                        if k <= r < k+1:
                            shell_power.append(power[z, y, x])
            radial.append(float(np.mean(shell_power)) if shell_power else 0.0)
        
        return {'power': power, 'radial': radial, 'dominant_k': int(np.argmax(radial)) if radial else 0}


# =============================================================================
# DECOHERENCE MICROSCOPE
# =============================================================================

class DecoherenceMicroscope:
    """
    Deco-micro: varying decoherence rates reveal internal structure.
    Different coherence timescales = different structural layers.
    """
    
    def __init__(self, substrate: Substrate3D):
        self.sub = substrate
    
    def apply_dephasing(self, rho: np.ndarray, rate: float, dt: float = 0.1) -> np.ndarray:
        """Apply dephasing (T2 decay) to off-diagonal elements."""
        rho_new = rho.copy()
        decay = np.exp(-rate * dt)
        
        for i in range(self.sub.n_sites):
            for j in range(self.sub.n_sites):
                if i != j:
                    rho_new[i, j] *= decay
        
        # Normalize
        rho_new /= np.trace(rho_new)
        return rho_new
    
    def apply_amplitude_damping(self, rho: np.ndarray, rate: float, dt: float = 0.1) -> np.ndarray:
        """Apply amplitude damping (T1 decay) - excitation loss to environment."""
        decay = np.exp(-rate * dt)
        
        # Diagonal: probability survives with rate decay
        rho_new = rho * decay
        
        # Off-diagonal decays faster
        for i in range(self.sub.n_sites):
            for j in range(self.sub.n_sites):
                if i != j:
                    rho_new[i, j] *= np.sqrt(decay)
        
        # Lost population goes to... (we're in single excitation, so it's lost)
        rho_new /= (np.trace(rho_new) + 1e-10)
        return rho_new
    
    def scan(self, psi: np.ndarray, rates: List[float], n_steps: int = 50, 
             mode: str = 'dephasing') -> Dict:
        """
        Scan through decoherence rates.
        mode: 'dephasing' (T2), 'amplitude' (T1), or 'both'
        """
        rho_init = self.sub.density_matrix(psi)
        
        results = {'rates': rates, 'layers': [], 'coherence': [], 'purity': []}
        
        for rate in rates:
            rho = rho_init.copy()
            
            for _ in range(n_steps):
                if mode in ['dephasing', 'both']:
                    rho = self.apply_dephasing(rho, rate)
                if mode in ['amplitude', 'both']:
                    rho = self.apply_amplitude_damping(rho, rate * 0.5)
            
            # Measure
            diag = np.real(np.diag(rho))
            off_diag_sum = np.sum(np.abs(rho)) - np.sum(np.abs(diag))
            purity = float(np.real(np.trace(rho @ rho)))
            coherence = off_diag_sum / (self.sub.n_sites**2 - self.sub.n_sites + 1e-10)
            
            results['layers'].append({'rate': rate, 'density': diag.tolist(), 'purity': purity})
            results['coherence'].append(float(coherence))
            results['purity'].append(purity)
        
        return results
    
    def differential_scan(self, psi: np.ndarray, n_rates: int = 8) -> Dict:
        """What structure is lost at each decoherence scale?"""
        rates = np.logspace(-2, 0, n_rates)
        
        rho = self.sub.density_matrix(psi)
        prev_diag = np.real(np.diag(rho))
        
        layers = []
        for rate in rates:
            rho = self.apply_dephasing(rho, rate, dt=0.2)
            curr_diag = np.real(np.diag(rho))
            
            layers.append({
                'rate': float(rate),
                'lost': (prev_diag - curr_diag).tolist(),
                'remaining': curr_diag.tolist(),
                'coherence_remaining': float(np.sum(np.abs(rho - np.diag(np.diag(rho)))))
            })
            prev_diag = curr_diag.copy()
        
        return {'rates': rates.tolist(), 'layers': layers}


# =============================================================================
# VISUALIZER
# =============================================================================

class SubstrateVisualizer:
    def __init__(self, substrate: Substrate3D):
        self.sub = substrate
        self.wavelet = WaveletAnalyzer(substrate)
        self.deco = DecoherenceMicroscope(substrate)
    
    def plot_density_3d(self, psi: np.ndarray, title: str = "", filename: str = None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        density = self.sub.density(psi)
        xs = [c[0] for c in self.sub.coords]
        ys = [c[1] for c in self.sub.coords]
        zs = [c[2] for c in self.sub.coords]
        
        sizes = 1000 * density / (np.max(density) + 1e-10)
        scatter = ax.scatter(xs, ys, zs, c=density, s=sizes, cmap='hot', 
                            alpha=0.8, edgecolors='black')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        plt.colorbar(scatter, ax=ax, label='|ψ|²', shrink=0.6)
        
        if filename:
            plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=150, bbox_inches='tight')
            plt.close()
    
    def plot_wavelet(self, psi: np.ndarray, filename: str = "wavelet.png"):
        decomp = self.wavelet.decompose(psi)
        n_scales = len(decomp['scales'])
        
        fig, axes = plt.subplots(2, n_scales, figsize=(4*n_scales, 6))
        if n_scales == 1: axes = axes.reshape(2, 1)
        
        for i, scale in enumerate(decomp['scales']):
            ax = axes[0, i]
            if 'detail' in scale:
                data = scale['detail']
                mid = data.shape[0] // 2
                im = ax.imshow(data[mid] if data.ndim == 3 else data, cmap='RdBu', aspect='equal')
                ax.set_title(f"Scale {scale['scale']} Detail\nE={scale['energy']:.4f}")
            else:
                data = scale['approximation']
                mid = data.shape[0] // 2 if data.ndim == 3 else 0
                im = ax.imshow(data[mid] if data.ndim == 3 else data, cmap='hot', aspect='equal')
                ax.set_title(f"Coarse\nE={scale['energy']:.4f}")
            plt.colorbar(im, ax=ax, shrink=0.6)
            
            ax = axes[1, i]
            if 'smooth' in scale:
                data = scale['smooth']
                mid = data.shape[0] // 2 if data.ndim == 3 else 0
                im = ax.imshow(data[mid] if data.ndim == 3 else data, cmap='viridis', aspect='equal')
                ax.set_title("Smoothed")
                plt.colorbar(im, ax=ax, shrink=0.6)
            else:
                ax.axis('off')
        
        plt.suptitle('Wavelet Decomposition', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=150)
        plt.close()
        return decomp
    
    def plot_deco_scan(self, psi: np.ndarray, filename: str = "deco_scan.png"):
        rates = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        scan = self.deco.scan(psi, rates)
        
        n = len(rates)
        fig = plt.figure(figsize=(16, 8))
        
        # 3D density at each rate
        for i, layer in enumerate(scan['layers']):
            ax = fig.add_subplot(2, n, i+1, projection='3d')
            density = np.array(layer['density'])
            xs = [c[0] for c in self.sub.coords]
            ys = [c[1] for c in self.sub.coords]
            zs = [c[2] for c in self.sub.coords]
            sizes = 800 * density / (np.max(density) + 1e-10)
            ax.scatter(xs, ys, zs, c=density, s=sizes, cmap='hot', alpha=0.8)
            ax.set_title(f"γ={layer['rate']:.2f}\nP={layer['purity']:.3f}")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        
        # Coherence decay
        ax = fig.add_subplot(2, 3, 4)
        ax.semilogy(rates, scan['coherence'], 'bo-', lw=2, ms=8)
        ax.set_xlabel('Decoherence Rate γ')
        ax.set_ylabel('Coherence')
        ax.set_title('Coherence Decay')
        ax.grid(True, alpha=0.3)
        
        # Purity decay
        ax = fig.add_subplot(2, 3, 5)
        ax.plot(rates, scan['purity'], 'rs-', lw=2, ms=8)
        ax.set_xlabel('Decoherence Rate γ')
        ax.set_ylabel('Purity Tr(ρ²)')
        ax.set_title('Quantum → Classical')
        ax.grid(True, alpha=0.3)
        
        # Structure metric: IPR of diagonal
        ax = fig.add_subplot(2, 3, 6)
        iprs = [np.sum(np.array(l['density'])**2) for l in scan['layers']]
        ax.plot(rates, iprs, 'g^-', lw=2, ms=8)
        ax.set_xlabel('Decoherence Rate γ')
        ax.set_ylabel('IPR')
        ax.set_title('Localization')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Decoherence Microscope', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=150)
        plt.close()
        return scan
    
    def plot_differential(self, psi: np.ndarray, filename: str = "deco_diff.png"):
        diff = self.deco.differential_scan(psi)
        n = len(diff['layers'])
        
        fig, axes = plt.subplots(2, min(n, 6), figsize=(3*min(n,6), 6))
        
        for i, layer in enumerate(diff['layers'][:6]):
            # What was lost
            ax = axes[0, i] if n > 1 else axes[0]
            lost = np.array(layer['lost']).reshape(self.sub.Lz, self.sub.Ly, self.sub.Lx)
            mid = lost.shape[0] // 2
            im = ax.imshow(lost[mid], cmap='RdBu', aspect='equal', vmin=-0.05, vmax=0.05)
            ax.set_title(f"γ={layer['rate']:.3f}\nLost")
            plt.colorbar(im, ax=ax, shrink=0.6)
            
            # What remains
            ax = axes[1, i] if n > 1 else axes[1]
            rem = np.array(layer['remaining']).reshape(self.sub.Lz, self.sub.Ly, self.sub.Lx)
            im = ax.imshow(rem[mid], cmap='hot', aspect='equal')
            ax.set_title(f"Coh={layer['coherence_remaining']:.3f}")
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        plt.suptitle('Differential Deco-Micro: Structure Layer by Layer', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=150)
        plt.close()
        return diff
    
    def full_analysis(self, psi: np.ndarray, name: str):
        print(f"\n[{name}]")
        
        self.plot_density_3d(psi, title=f"{name}", filename=f"{name}_density.png")
        print(f"  density: {name}_density.png")
        
        w = self.plot_wavelet(psi, f"{name}_wavelet.png")
        print(f"  wavelet: {len(w['scales'])} scales, total E={w['total_energy']:.4f}")
        
        d = self.plot_deco_scan(psi, f"{name}_deco.png")
        print(f"  deco: coherence decay {d['coherence'][0]:.3f} → {d['coherence'][-1]:.3f}")
        
        df = self.plot_differential(psi, f"{name}_diff.png")
        print(f"  diff: {len(df['layers'])} layers")
        
        return {'wavelet': w, 'deco': d, 'diff': df}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*50)
    print("SUBSTRATE VISUALIZER")
    print("="*50)
    
    # 4x4x4 lattice (64 sites, manageable)
    sub = Substrate3D(4, 4, 4)
    viz = SubstrateVisualizer(sub)
    
    results = {}
    
    # Pattern 1: Single excitation
    psi1 = sub.excitation(0)
    results['point'] = viz.full_analysis(psi1, "point")
    
    # Pattern 2: Gaussian packet
    psi2 = sub.localized_packet((1.5, 1.5, 1.5), width=1.0)
    results['packet'] = viz.full_analysis(psi2, "packet")
    
    # Pattern 3: Superposition (cat state)
    psi3 = (sub.excitation(0) + sub.excitation(63)) / np.sqrt(2)
    results['cat'] = viz.full_analysis(psi3, "cat")
    
    # Pattern 4: Spread state
    psi4 = np.ones(sub.n_sites, dtype=complex) / np.sqrt(sub.n_sites)
    results['spread'] = viz.full_analysis(psi4, "spread")
    
    # Pattern 5: Evolved state
    H = sub.hamiltonian(hopping=1.0)
    psi5 = expm(-1j * H * 2.0) @ sub.excitation(0)
    psi5 /= np.linalg.norm(psi5)
    results['evolved'] = viz.full_analysis(psi5, "evolved")
    
    # Pattern 6: Random phase
    psi6 = sub.localized_packet((1.5, 1.5, 1.5), width=1.5) * np.exp(2j * np.pi * np.random.rand(sub.n_sites))
    psi6 /= np.linalg.norm(psi6)
    results['random_phase'] = viz.full_analysis(psi6, "random_phase")
    
    print(f"\n{'='*50}")
    print(f"Output: {OUTPUT_DIR}/")
    print("="*50)


if __name__ == "__main__":
    main()