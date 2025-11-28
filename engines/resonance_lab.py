#!/usr/bin/env python3
"""
Resonance Laboratory
====================
Locate a proton-like pattern in the substrate.
Probe it to create a pointer state.
Analyze with every tool we have.
"""

import numpy as np
from scipy.linalg import svd, eigh
from scipy.fft import fft, ifft
from scipy.ndimage import gaussian_filter1d
from substrate import Substrate, Config
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =============================================================================
# PROTON PATTERN DETECTION
# =============================================================================

class ProtonLocator:
    """
    Locate proton-like patterns in the substrate.
    
    A proton pattern is:
        - A cluster of 3 nodes (quarks)
        - Strongly entangled with each other
        - Forming a color singlet (SU(3) invariant)
        - Relatively isolated from environment
    """
    
    def __init__(self, substrate: Substrate):
        self.substrate = substrate
        self.d = substrate.config.internal_dim
    
    def find_clusters(self, min_size: int = 3, max_size: int = 5) -> List[List[int]]:
        """Find tightly connected clusters."""
        clusters = []
        visited = set()
        
        for start_id in self.substrate.nodes:
            if start_id in visited:
                continue
            
            # BFS to find connected component
            cluster = []
            queue = [start_id]
            
            while queue and len(cluster) < max_size:
                node_id = queue.pop(0)
                if node_id in visited:
                    continue
                
                visited.add(node_id)
                node = self.substrate.nodes[node_id]
                
                # Only include if strongly connected to cluster
                if cluster:
                    connections_to_cluster = sum(
                        1 for c in node.connections if c in cluster
                    )
                    if connections_to_cluster == 0:
                        continue
                
                cluster.append(node_id)
                
                # Add strongly connected neighbors
                for nbr_id, strength in node.connections.items():
                    if nbr_id not in visited and strength > 0.1:
                        queue.append(nbr_id)
            
            if min_size <= len(cluster) <= max_size:
                clusters.append(cluster)
        
        return clusters
    
    def cluster_entanglement(self, cluster: List[int]) -> float:
        """Measure total internal entanglement of a cluster."""
        total = 0.0
        for i, node_id in enumerate(cluster):
            node = self.substrate.nodes[node_id]
            for other_id in cluster[i+1:]:
                if other_id in node.connections:
                    total += node.connections[other_id]
        return total
    
    def cluster_isolation(self, cluster: List[int]) -> float:
        """Measure how isolated the cluster is from environment."""
        internal = 0.0
        external = 0.0
        
        cluster_set = set(cluster)
        
        for node_id in cluster:
            node = self.substrate.nodes[node_id]
            for nbr_id, strength in node.connections.items():
                if nbr_id in cluster_set:
                    internal += strength
                else:
                    external += strength
        
        if internal + external == 0:
            return 0
        return internal / (internal + external)
    
    def color_singlet_score(self, cluster: List[int]) -> float:
        """
        Measure how close the cluster is to a color singlet.
        
        For SU(3), a singlet has total color = 0.
        We approximate by checking if internal states sum to uniform.
        """
        if self.d != 3:
            return 0.0
        
        # Sum of internal states
        total_state = np.zeros(self.d, dtype=np.complex128)
        for node_id in cluster:
            total_state += self.substrate.nodes[node_id].state
        
        # Singlet = uniform distribution
        uniform = np.ones(self.d) / np.sqrt(self.d)
        overlap = np.abs(np.vdot(total_state / np.linalg.norm(total_state), uniform))
        
        return overlap
    
    def find_proton(self) -> Optional[List[int]]:
        """Find the best proton-like pattern."""
        clusters = self.find_clusters(min_size=3, max_size=4)
        
        if not clusters:
            return None
        
        # Score each cluster
        scores = []
        for cluster in clusters:
            ent = self.cluster_entanglement(cluster)
            iso = self.cluster_isolation(cluster)
            singlet = self.color_singlet_score(cluster) if self.d == 3 else 1.0
            
            score = ent * iso * singlet
            scores.append((score, cluster))
        
        scores.sort(reverse=True)
        return scores[0][1] if scores else None


# =============================================================================
# POINTER STATE
# =============================================================================

@dataclass
class PointerState:
    """
    The pointer state of a proton-like pattern.
    
    Created by probing the pattern and extracting the
    stable, classical-like state that emerges.
    """
    cluster: List[int]
    
    # Combined state vector (tensor product of cluster nodes)
    state_vector: np.ndarray
    
    # Density matrix
    density_matrix: np.ndarray
    
    # Reduced density matrix (traced over environment)
    reduced_dm: np.ndarray
    
    # Eigendecomposition
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    
    # Time series (from probing)
    time_series: np.ndarray
    times: np.ndarray
    
    # Internal structure
    internal_states: Dict[int, np.ndarray]
    entanglement_map: np.ndarray


class Probe:
    """
    Probe the substrate to create a pointer state.
    
    Probing = weak measurement that extracts information
    without fully collapsing the state.
    """
    
    def __init__(self, substrate: Substrate):
        self.substrate = substrate
    
    def create_pointer_state(self, cluster: List[int], 
                              n_probe_steps: int = 100) -> PointerState:
        """Create a pointer state by probing the cluster."""
        
        d = self.substrate.config.internal_dim
        n_cluster = len(cluster)
        
        # Collect time series of internal states
        time_series = []
        times = []
        
        dt = self.substrate.config.dt
        
        for step in range(n_probe_steps):
            # Record current states
            states = [self.substrate.nodes[nid].state.copy() for nid in cluster]
            combined = states[0]
            for s in states[1:]:
                combined = np.kron(combined, s)
            
            time_series.append(combined)
            times.append(step * dt)
            
            # Evolve (probe perturbs slightly)
            self.substrate.evolve(n_steps=1)
            
            # Small probe perturbation
            for nid in cluster:
                node = self.substrate.nodes[nid]
                # Weak measurement in random basis
                noise = 0.01 * (np.random.randn(d) + 1j * np.random.randn(d))
                node.state = node.state + noise
                node.state = node.state / np.linalg.norm(node.state)
        
        time_series = np.array(time_series)
        times = np.array(times)
        
        # Final state vector
        final_states = [self.substrate.nodes[nid].state.copy() for nid in cluster]
        state_vector = final_states[0]
        for s in final_states[1:]:
            state_vector = np.kron(state_vector, s)
        
        # Density matrix
        density_matrix = np.outer(state_vector, state_vector.conj())
        
        # Reduced density matrix (trace over last particle)
        dim_single = d
        dim_first = d ** (n_cluster - 1)
        dm_reshaped = density_matrix.reshape(dim_first, dim_single, dim_first, dim_single)
        reduced_dm = np.trace(dm_reshaped, axis1=1, axis2=3)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(reduced_dm)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Internal states
        internal_states = {nid: self.substrate.nodes[nid].state.copy() for nid in cluster}
        
        # Entanglement map
        entanglement_map = np.zeros((n_cluster, n_cluster))
        for i, ni in enumerate(cluster):
            for j, nj in enumerate(cluster):
                if i != j:
                    node_i = self.substrate.nodes[ni]
                    if nj in node_i.connections:
                        entanglement_map[i, j] = node_i.connections[nj]
        
        return PointerState(
            cluster=cluster,
            state_vector=state_vector,
            density_matrix=density_matrix,
            reduced_dm=reduced_dm,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            time_series=time_series,
            times=times,
            internal_states=internal_states,
            entanglement_map=entanglement_map
        )


# =============================================================================
# ANALYSIS TOOLS
# =============================================================================

class WaveletAnalysis:
    """Wavelet scalogram analysis."""
    
    def __init__(self, pointer: PointerState):
        self.pointer = pointer
    
    def compute_scalogram(self, component: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute wavelet scalogram for a state component."""
        signal = self.pointer.time_series[:, component]
        signal_real = np.real(signal)
        n = len(signal_real)
        
        # Wavelet scales
        widths = np.arange(1, min(21, n // 3))
        
        # Manual Morlet wavelet transform
        scalogram = np.zeros((len(widths), n))
        
        for i, width in enumerate(widths):
            # Morlet wavelet
            t = np.arange(-4*width, 4*width + 1)
            wavelet = np.exp(1j * 2 * np.pi * t / width) * np.exp(-t**2 / (2 * width**2))
            wavelet = wavelet / np.sqrt(width)
            
            # Convolve and center crop
            conv = np.convolve(signal_real, wavelet, mode='full')
            start = (len(conv) - n) // 2
            scalogram[i, :] = np.abs(conv[start:start + n])
        
        return self.pointer.times, widths, scalogram
    
    def compute_all_scalograms(self) -> Dict[int, Tuple]:
        """Compute scalograms for all major components."""
        n_components = min(4, self.pointer.state_vector.shape[0])
        
        scalograms = {}
        for i in range(n_components):
            scalograms[i] = self.compute_scalogram(i)
        
        return scalograms


class DecoherenceMicroscope:
    """Analyze decoherence structure."""
    
    def __init__(self, pointer: PointerState):
        self.pointer = pointer
    
    def purity_evolution(self) -> Tuple[np.ndarray, np.ndarray]:
        """Track purity over time."""
        purities = []
        
        for state in self.pointer.time_series:
            dm = np.outer(state, state.conj())
            purity = np.real(np.trace(dm @ dm))
            purities.append(purity)
        
        return self.pointer.times, np.array(purities)
    
    def coherence_matrix(self) -> np.ndarray:
        """Off-diagonal coherences in density matrix."""
        dm = self.pointer.density_matrix
        n = dm.shape[0]
        
        coherences = np.abs(dm) - np.diag(np.diag(np.abs(dm)))
        return coherences
    
    def decoherence_rates(self) -> np.ndarray:
        """Estimate decoherence rate for each off-diagonal element."""
        n_times = len(self.pointer.times)
        n_dim = self.pointer.time_series.shape[1]
        
        rates = np.zeros((n_dim, n_dim))
        
        for i in range(n_dim):
            for j in range(i+1, n_dim):
                # Track coherence |ρ_ij| over time
                coherences = []
                for state in self.pointer.time_series:
                    dm = np.outer(state, state.conj())
                    coherences.append(np.abs(dm[i, j]))
                
                coherences = np.array(coherences)
                
                # Fit exponential decay
                if coherences[0] > 1e-10:
                    log_coh = np.log(coherences + 1e-10)
                    rate = -np.polyfit(self.pointer.times, log_coh, 1)[0]
                    rates[i, j] = rate
                    rates[j, i] = rate
        
        return rates
    
    def pointer_basis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Find the pointer basis (eigenstates of reduced DM)."""
        return self.pointer.eigenvalues, self.pointer.eigenvectors


class AmplitudeMap:
    """Map amplitude distribution in state space."""
    
    def __init__(self, pointer: PointerState):
        self.pointer = pointer
    
    def probability_distribution(self) -> np.ndarray:
        """Probability distribution over basis states."""
        return np.abs(self.pointer.state_vector) ** 2
    
    def phase_distribution(self) -> np.ndarray:
        """Phase distribution over basis states."""
        return np.angle(self.pointer.state_vector)
    
    def husimi_Q(self, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Husimi Q function (phase space representation).
        
        For simplicity, project onto first two components.
        """
        state = self.pointer.state_vector[:4] if len(self.pointer.state_vector) > 4 else self.pointer.state_vector
        state = state / np.linalg.norm(state)
        
        # Create grid
        x = np.linspace(-3, 3, n_points)
        y = np.linspace(-3, 3, n_points)
        X, Y = np.meshgrid(x, y)
        
        Q = np.zeros_like(X)
        
        # Coherent state overlap
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                alpha = (xi + 1j * yj) / np.sqrt(2)
                
                # Coherent state (truncated)
                coherent = np.zeros(len(state), dtype=np.complex128)
                log_prefactor = -np.abs(alpha)**2 / 2
                for n in range(len(state)):
                    if n == 0:
                        log_coeff = 0
                    else:
                        log_coeff = n * np.log(np.abs(alpha) + 1e-10) - 0.5 * sum(np.log(k) for k in range(1, n+1))
                    phase = n * np.angle(alpha)
                    coherent[n] = np.exp(log_prefactor + log_coeff) * np.exp(1j * phase)
                
                norm = np.linalg.norm(coherent)
                if norm > 1e-10:
                    coherent = coherent / norm
                
                Q[j, i] = np.abs(np.vdot(coherent, state)) ** 2
        
        return X, Y, Q
    
    def internal_amplitudes(self) -> Dict[int, np.ndarray]:
        """Amplitude in each internal direction for each cluster node."""
        result = {}
        for nid, state in self.pointer.internal_states.items():
            result[nid] = np.abs(state) ** 2
        return result


class ComponentAnalysis:
    """Analyze components along each axis."""
    
    def __init__(self, pointer: PointerState):
        self.pointer = pointer
        self.d = len(list(pointer.internal_states.values())[0])
    
    def axis_projections(self) -> Dict[int, Dict[int, float]]:
        """Project each node's state onto each axis."""
        projections = {}
        
        for nid, state in self.pointer.internal_states.items():
            projections[nid] = {}
            for axis in range(self.d):
                basis = np.zeros(self.d, dtype=np.complex128)
                basis[axis] = 1.0
                projections[nid][axis] = np.abs(np.vdot(basis, state)) ** 2
        
        return projections
    
    def bloch_coordinates(self) -> Dict[int, np.ndarray]:
        """Bloch sphere coordinates for each node (d=2 only)."""
        if self.d != 2:
            return {}
        
        coords = {}
        
        for nid, state in self.pointer.internal_states.items():
            # Bloch vector components
            a, b = state[0], state[1]
            
            x = 2 * np.real(a.conj() * b)
            y = 2 * np.imag(a.conj() * b)
            z = np.abs(a)**2 - np.abs(b)**2
            
            coords[nid] = np.array([x, y, z])
        
        return coords
    
    def color_composition(self) -> Dict[int, np.ndarray]:
        """Color composition for each node (d=3)."""
        if self.d != 3:
            return {}
        
        colors = {}
        for nid, state in self.pointer.internal_states.items():
            # RGB from amplitudes
            colors[nid] = np.abs(state) ** 2
        
        return colors
    
    def total_color(self) -> np.ndarray:
        """Total color of the cluster."""
        if self.d != 3:
            return np.array([])
        
        total = np.zeros(3, dtype=np.complex128)
        for state in self.pointer.internal_states.values():
            total += state
        
        return np.abs(total) ** 2 / np.sum(np.abs(total) ** 2)


class EntanglementAnalysis:
    """Analyze entanglement structure."""
    
    def __init__(self, pointer: PointerState):
        self.pointer = pointer
    
    def von_neumann_entropy(self) -> float:
        """Von Neumann entropy of reduced density matrix."""
        evals = self.pointer.eigenvalues
        evals = evals[evals > 1e-10]
        return -np.sum(evals * np.log(evals))
    
    def entanglement_spectrum(self) -> np.ndarray:
        """Entanglement spectrum (eigenvalues of reduced DM)."""
        return self.pointer.eigenvalues
    
    def mutual_information(self) -> np.ndarray:
        """Mutual information between cluster nodes."""
        n = len(self.pointer.cluster)
        MI = np.zeros((n, n))
        
        # Approximate from entanglement map
        ent_map = self.pointer.entanglement_map
        
        for i in range(n):
            for j in range(i+1, n):
                # MI ~ entanglement strength (rough approximation)
                MI[i, j] = ent_map[i, j]
                MI[j, i] = MI[i, j]
        
        return MI
    
    def schmidt_decomposition(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Schmidt decomposition of the state."""
        state = self.pointer.state_vector
        n = len(state)
        
        # Reshape as matrix and SVD
        sqrt_n = int(np.sqrt(n))
        if sqrt_n * sqrt_n == n:
            mat = state.reshape(sqrt_n, sqrt_n)
            U, s, Vh = svd(mat)
            return s, U, Vh
        
        return np.array([1.0]), np.array([[1.0]]), np.array([[1.0]])


# =============================================================================
# VISUALIZATION
# =============================================================================

class ResonanceLaboratory:
    """
    The complete analysis suite.
    
    Combines all tools and generates comprehensive visualization.
    """
    
    def __init__(self, substrate: Substrate):
        self.substrate = substrate
        self.locator = ProtonLocator(substrate)
        self.probe = Probe(substrate)
        
        self.pointer = None
        self.analyses = {}
    
    def locate_and_probe(self, n_probe_steps: int = 100) -> PointerState:
        """Locate proton pattern and create pointer state."""
        cluster = self.locator.find_proton()
        
        if cluster is None:
            # Create artificial cluster from strongest connected nodes
            nodes = list(self.substrate.nodes.values())
            nodes.sort(key=lambda n: n.total_entanglement, reverse=True)
            cluster = [n.id for n in nodes[:3]]
        
        self.pointer = self.probe.create_pointer_state(cluster, n_probe_steps)
        return self.pointer
    
    def run_all_analyses(self):
        """Run all analysis tools."""
        if self.pointer is None:
            raise ValueError("Must call locate_and_probe first")
        
        self.analyses['wavelet'] = WaveletAnalysis(self.pointer)
        self.analyses['decoherence'] = DecoherenceMicroscope(self.pointer)
        self.analyses['amplitude'] = AmplitudeMap(self.pointer)
        self.analyses['components'] = ComponentAnalysis(self.pointer)
        self.analyses['entanglement'] = EntanglementAnalysis(self.pointer)
        
        return self.analyses
    
    def generate_report(self, output_path: str = "resonance_report.png"):
        """Generate comprehensive visualization."""
        
        if not self.analyses:
            self.run_all_analyses()
        
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(f"Resonance Laboratory: Proton Pointer State Analysis\n"
                     f"Cluster: {self.pointer.cluster} | d={self.substrate.config.internal_dim}",
                     fontsize=14, fontweight='bold')
        
        # 1. Wavelet scalogram (top left, spans 2 cols)
        ax1 = fig.add_subplot(gs[0, 0:2])
        times, widths, scalogram = self.analyses['wavelet'].compute_scalogram(0)
        im1 = ax1.imshow(scalogram, aspect='auto', origin='lower',
                        extent=[times[0], times[-1], widths[0], widths[-1]],
                        cmap='magma')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Scale')
        ax1.set_title('Wavelet Scalogram (Component 0)')
        plt.colorbar(im1, ax=ax1, label='|CWT|')
        
        # 2. Decoherence: Purity evolution
        ax2 = fig.add_subplot(gs[0, 2])
        times, purity = self.analyses['decoherence'].purity_evolution()
        ax2.plot(times, purity, 'b-', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Purity')
        ax2.set_title('Purity Evolution')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # 3. Coherence matrix
        ax3 = fig.add_subplot(gs[0, 3])
        coherences = self.analyses['decoherence'].coherence_matrix()
        im3 = ax3.imshow(coherences[:8, :8], cmap='viridis')
        ax3.set_title('Coherence Matrix')
        ax3.set_xlabel('Basis state')
        ax3.set_ylabel('Basis state')
        plt.colorbar(im3, ax=ax3)
        
        # 4. Amplitude distribution
        ax4 = fig.add_subplot(gs[1, 0])
        probs = self.analyses['amplitude'].probability_distribution()
        ax4.bar(range(len(probs[:16])), probs[:16], color='steelblue')
        ax4.set_xlabel('Basis state')
        ax4.set_ylabel('Probability')
        ax4.set_title('Amplitude Distribution')
        
        # 5. Phase distribution
        ax5 = fig.add_subplot(gs[1, 1])
        phases = self.analyses['amplitude'].phase_distribution()
        ax5.bar(range(len(phases[:16])), phases[:16], color='coral')
        ax5.set_xlabel('Basis state')
        ax5.set_ylabel('Phase (rad)')
        ax5.set_title('Phase Distribution')
        
        # 6. Husimi Q function
        ax6 = fig.add_subplot(gs[1, 2])
        X, Y, Q = self.analyses['amplitude'].husimi_Q()
        im6 = ax6.contourf(X, Y, Q, levels=20, cmap='plasma')
        ax6.set_xlabel('Re(α)')
        ax6.set_ylabel('Im(α)')
        ax6.set_title('Husimi Q Function')
        plt.colorbar(im6, ax=ax6)
        
        # 7. Internal amplitudes per node
        ax7 = fig.add_subplot(gs[1, 3])
        internal = self.analyses['amplitude'].internal_amplitudes()
        n_nodes = len(internal)
        d = self.substrate.config.internal_dim
        x = np.arange(d)
        width = 0.8 / n_nodes
        
        for i, (nid, amps) in enumerate(internal.items()):
            ax7.bar(x + i * width, amps, width, label=f'Node {nid}')
        
        ax7.set_xlabel('Internal axis')
        ax7.set_ylabel('Amplitude²')
        ax7.set_title('Internal Structure')
        ax7.legend(fontsize=8)
        ax7.set_xticks(x + width * (n_nodes - 1) / 2)
        ax7.set_xticklabels([f'Axis {i}' for i in range(d)])
        
        # 8. Entanglement spectrum
        ax8 = fig.add_subplot(gs[2, 0])
        spectrum = self.analyses['entanglement'].entanglement_spectrum()
        ax8.bar(range(len(spectrum[:8])), spectrum[:8], color='purple')
        ax8.set_xlabel('Eigenvalue index')
        ax8.set_ylabel('λ')
        ax8.set_title(f'Entanglement Spectrum\nS = {self.analyses["entanglement"].von_neumann_entropy():.3f}')
        
        # 9. Mutual information
        ax9 = fig.add_subplot(gs[2, 1])
        MI = self.analyses['entanglement'].mutual_information()
        im9 = ax9.imshow(MI, cmap='Reds')
        ax9.set_title('Mutual Information')
        ax9.set_xlabel('Node index')
        ax9.set_ylabel('Node index')
        plt.colorbar(im9, ax=ax9)
        
        # 10. Entanglement map (from pointer)
        ax10 = fig.add_subplot(gs[2, 2])
        ent_map = self.pointer.entanglement_map
        im10 = ax10.imshow(ent_map, cmap='Blues')
        ax10.set_title('Entanglement Map')
        ax10.set_xlabel('Node index')
        ax10.set_ylabel('Node index')
        plt.colorbar(im10, ax=ax10)
        
        # 11. Decoherence rates
        ax11 = fig.add_subplot(gs[2, 3])
        rates = self.analyses['decoherence'].decoherence_rates()
        im11 = ax11.imshow(rates[:8, :8], cmap='hot')
        ax11.set_title('Decoherence Rates')
        ax11.set_xlabel('Basis state')
        ax11.set_ylabel('Basis state')
        plt.colorbar(im11, ax=ax11)
        
        # 12. Pointer basis eigenvalues
        ax12 = fig.add_subplot(gs[3, 0])
        eigenvalues, _ = self.analyses['decoherence'].pointer_basis()
        ax12.bar(range(len(eigenvalues[:8])), eigenvalues[:8], color='green')
        ax12.set_xlabel('Eigenstate index')
        ax12.set_ylabel('Eigenvalue')
        ax12.set_title('Pointer Basis')
        
        # 13. Component projections / Bloch / Color
        ax13 = fig.add_subplot(gs[3, 1])
        d = self.substrate.config.internal_dim
        
        if d == 2:
            bloch = self.analyses['components'].bloch_coordinates()
            for nid, coords in bloch.items():
                ax13.scatter([coords[0]], [coords[1]], s=100, label=f'Node {nid}')
            circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
            ax13.add_patch(circle)
            ax13.set_xlim(-1.5, 1.5)
            ax13.set_ylim(-1.5, 1.5)
            ax13.set_aspect('equal')
            ax13.set_xlabel('X')
            ax13.set_ylabel('Y')
            ax13.set_title('Bloch Sphere (XY plane)')
            ax13.legend()
        elif d == 3:
            colors = self.analyses['components'].color_composition()
            for i, (nid, rgb) in enumerate(colors.items()):
                ax13.bar(i, 1, color=rgb, label=f'Node {nid}')
            ax13.set_title('Color Composition')
            ax13.set_xticks(range(len(colors)))
            ax13.set_xticklabels([f'N{nid}' for nid in colors.keys()])
            
            total = self.analyses['components'].total_color()
            ax13.text(0.5, 0.5, f'Total color:\nR={total[0]:.2f}\nG={total[1]:.2f}\nB={total[2]:.2f}',
                     transform=ax13.transAxes, fontsize=10)
        else:
            proj = self.analyses['components'].axis_projections()
            ax13.set_title('Axis Projections')
        
        # 14. Time series (first component)
        ax14 = fig.add_subplot(gs[3, 2])
        ts = self.pointer.time_series[:, 0]
        ax14.plot(self.pointer.times, np.real(ts), 'b-', label='Real', alpha=0.7)
        ax14.plot(self.pointer.times, np.imag(ts), 'r-', label='Imag', alpha=0.7)
        ax14.plot(self.pointer.times, np.abs(ts), 'k-', label='|ψ|', linewidth=2)
        ax14.set_xlabel('Time')
        ax14.set_ylabel('Amplitude')
        ax14.set_title('State Component 0')
        ax14.legend(fontsize=8)
        ax14.grid(True, alpha=0.3)
        
        # 15. Schmidt values
        ax15 = fig.add_subplot(gs[3, 3])
        schmidt, _, _ = self.analyses['entanglement'].schmidt_decomposition()
        ax15.bar(range(len(schmidt)), schmidt ** 2, color='teal')
        ax15.set_xlabel('Schmidt index')
        ax15.set_ylabel('λ²')
        ax15.set_title('Schmidt Decomposition')
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        
        return output_path


# =============================================================================
# MAIN
# =============================================================================

def run_resonance_laboratory(internal_dim: int = 3, n_nodes: int = 64,
                              output_prefix: str = "resonance"):
    """Run the complete Resonance Laboratory analysis."""
    
    print(f"Resonance Laboratory")
    print(f"=" * 40)
    print(f"Internal dimension: d={internal_dim}")
    print(f"Nodes: {n_nodes}")
    
    # Create and evolve substrate
    print("\nCreating substrate...")
    config = Config(
        n_nodes=n_nodes,
        internal_dim=internal_dim,
        defrag_rate=0.03,
        seed=42
    )
    
    substrate = Substrate(config)
    
    print("Evolving to equilibrium...")
    substrate.evolve(n_steps=100)
    
    # Run laboratory
    print("\nInitializing Resonance Laboratory...")
    lab = ResonanceLaboratory(substrate)
    
    print("Locating proton pattern...")
    pointer = lab.locate_and_probe(n_probe_steps=100)
    print(f"  Found cluster: {pointer.cluster}")
    
    print("Running analyses...")
    lab.run_all_analyses()
    
    # Summary
    ent_analysis = lab.analyses['entanglement']
    print(f"\nPointer State Summary:")
    print(f"  Von Neumann entropy: {ent_analysis.von_neumann_entropy():.4f}")
    print(f"  Dominant eigenvalue: {pointer.eigenvalues[0]:.4f}")
    print(f"  Entanglement dimension: {1/np.sum(pointer.eigenvalues**2):.2f}")
    
    # Generate report
    output_path = f"{output_prefix}_d{internal_dim}.png"
    print(f"\nGenerating report: {output_path}")
    lab.generate_report(output_path)
    
    return lab


if __name__ == "__main__":
    
    # Run for each internal dimension
    for d in [2, 3]:
        print(f"\n{'#' * 50}")
        print(f"# INTERNAL DIMENSION d = {d}")
        print(f"{'#' * 50}")
        
        lab = run_resonance_laboratory(internal_dim=d, n_nodes=64,
                                        output_prefix="resonance")
    
    print("\nDone. Reports saved.")