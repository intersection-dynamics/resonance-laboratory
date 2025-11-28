#!/usr/bin/env python3
"""
ENHANCED VISUALIZATIONS
=======================

Show how 1D substrate dynamics connect to 3D topological constraints.

Key narrative:
1. 1D: Excitation density shows wave propagation
2. 2D: Exchange paths can wind around each other (anyons possible)
3. 3D: Paths can be unwound → only ±1 phases stable

This is the visual story of why fermions exist.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import matplotlib

matplotlib.use('Agg')

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 150,
})


# =============================================================================
# Figure 1: 1D Substrate - Excitation Density Evolution
# =============================================================================

def create_1d_evolution():
    """Show excitation density evolving on 1D substrate."""
    
    N = 64
    t_max = 20
    n_times = 100
    
    # Hopping Hamiltonian
    H = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        j = (i + 1) % N
        H[i, j] = -1
        H[j, i] = -1
    
    # Initial state: localized Gaussian
    psi0 = np.zeros(N, dtype=np.complex128)
    center = N // 4
    width = 3
    for i in range(N):
        d = min(abs(i - center), N - abs(i - center))
        psi0[i] = np.exp(-d**2 / (2 * width**2))
    psi0 /= np.linalg.norm(psi0)
    
    # Time evolution
    times = np.linspace(0, t_max, n_times)
    densities = []
    
    for t in times:
        psi_t = expm(-1j * H * t) @ psi0
        densities.append(np.abs(psi_t)**2)
    
    densities = np.array(densities)
    
    # Create figure
    fig = plt.figure(figsize=(12, 5))
    
    # Left: 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    X, T = np.meshgrid(np.arange(N), times)
    ax1.plot_surface(X, T, densities, cmap='viridis', alpha=0.9, 
                     linewidth=0, antialiased=True)
    ax1.set_xlabel('Site')
    ax1.set_ylabel('Time')
    ax1.set_zlabel('Density |ψ|²')
    ax1.set_title('Excitation Propagation on 1D Substrate')
    ax1.view_init(elev=25, azim=-60)
    
    # Right: 2D heatmap
    ax2 = fig.add_subplot(122)
    im = ax2.imshow(densities.T, aspect='auto', origin='lower',
                    extent=[0, t_max, 0, N], cmap='viridis')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Site')
    ax2.set_title('Density Evolution (top view)')
    plt.colorbar(im, ax=ax2, label='|ψ|²')
    
    # Add lightcone annotation
    v = 2.0  # propagation speed
    ax2.plot([0, t_max], [center, center + v*t_max], 'r--', alpha=0.7, linewidth=2, label='Lightcone')
    ax2.plot([0, t_max], [center, center - v*t_max], 'r--', alpha=0.7, linewidth=2)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('figures/fig_1d_evolution.png', dpi=200, bbox_inches='tight')
    plt.savefig('figures/fig_1d_evolution.pdf', bbox_inches='tight')
    plt.close()
    print("Created: fig_1d_evolution.png")


# =============================================================================
# Figure 2: Exchange Topology in 2D vs 3D
# =============================================================================

def create_topology_comparison():
    """Visualize why 2D allows anyons but 3D forces ±1."""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # --- Left: 2D exchange paths ---
    ax = axes[0]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('2D: Paths Cannot Unwind', fontsize=12, fontweight='bold')
    
    # Draw two particles
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Particle A path (stays put, shown as circle)
    ax.plot(0, 0, 'ro', markersize=15, label='Particle A')
    
    # Particle B winds around A
    r = 0.8
    path_x = r * np.cos(theta)
    path_y = r * np.sin(theta)
    
    # Color by progress
    points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='coolwarm', linewidth=3)
    lc.set_array(np.linspace(0, 1, len(path_x)))
    ax.add_collection(lc)
    
    # Start/end markers
    ax.plot(r, 0, 'b>', markersize=12, label='Particle B (start)')
    ax.annotate('', xy=(r*np.cos(0.1), r*np.sin(0.1)), xytext=(r, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    ax.text(0, -1.3, 'Winding number = 1\nPhase can be any θ', 
            ha='center', fontsize=10, style='italic')
    ax.legend(loc='upper right', fontsize=9)
    ax.axis('off')
    
    # --- Middle: 3D exchange paths can unwind ---
    ax = axes[1]
    ax = fig.add_subplot(132, projection='3d')
    ax.set_title('3D: Paths CAN Unwind', fontsize=12, fontweight='bold')
    
    # Show a loop that lifts into 3D and unwinds
    t = np.linspace(0, 2*np.pi, 100)
    
    # Original 2D loop
    x1 = 0.8 * np.cos(t)
    y1 = 0.8 * np.sin(t)
    z1 = np.zeros_like(t)
    ax.plot(x1, y1, z1, 'b-', linewidth=2, alpha=0.5, label='2D loop')
    
    # Lifted loop that shrinks
    z2 = 0.5 * np.sin(t)  # Lift out of plane
    x2 = 0.8 * np.cos(t) * (1 - 0.3 * np.abs(np.sin(t)))
    y2 = 0.8 * np.sin(t) * (1 - 0.3 * np.abs(np.sin(t)))
    ax.plot(x2, y2, z2, 'r-', linewidth=3, label='Lifted → can shrink')
    
    # Central particle
    ax.scatter([0], [0], [0], c='green', s=100, label='Particle A')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(loc='upper left', fontsize=9)
    
    # --- Right: Consequence for statistics ---
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Consequence for Statistics', fontsize=12, fontweight='bold')
    
    # 2D box
    rect1 = mpatches.FancyBboxPatch((0.5, 5.5), 4, 3.5, 
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(rect1)
    ax.text(2.5, 8.3, '2D', fontsize=14, fontweight='bold', ha='center')
    ax.text(2.5, 7.2, 'π₁ = ℤ (integers)', fontsize=11, ha='center')
    ax.text(2.5, 6.4, 'Any phase φ allowed', fontsize=10, ha='center')
    ax.text(2.5, 5.7, '→ Anyons exist!', fontsize=10, ha='center', color='purple', fontweight='bold')
    
    # 3D box
    rect2 = mpatches.FancyBboxPatch((5.5, 5.5), 4, 3.5,
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(rect2)
    ax.text(7.5, 8.3, '3D', fontsize=14, fontweight='bold', ha='center')
    ax.text(7.5, 7.2, 'π₁ = ℤ₂ (two elements)', fontsize=11, ha='center')
    ax.text(7.5, 6.4, 'Only φ = 0, π allowed', fontsize=10, ha='center')
    ax.text(7.5, 5.7, '→ Bosons & Fermions only!', fontsize=10, ha='center', color='red', fontweight='bold')
    
    # Arrow and conclusion
    ax.annotate('', xy=(7.5, 2.5), xytext=(2.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(5, 3.2, 'Going to 3D', fontsize=11, ha='center')
    ax.text(5, 1.5, 'TOPOLOGY CONSTRAINS STATISTICS', fontsize=12, 
            ha='center', fontweight='bold', color='darkred',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figures/fig_topology_comparison.png', dpi=200, bbox_inches='tight')
    plt.savefig('figures/fig_topology_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("Created: fig_topology_comparison.png")


# =============================================================================
# Figure 3: The Complete Story - From Substrate to Statistics
# =============================================================================

def create_complete_story():
    """Visual summary of the entire framework."""
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # --- (0,0): Hilbert Space ---
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('1. Hilbert Space', fontsize=12, fontweight='bold')
    
    # Draw abstract Hilbert space
    circle = plt.Circle((5, 5), 3, fill=False, color='blue', linewidth=3)
    ax.add_patch(circle)
    ax.text(5, 5, 'ℋ', fontsize=30, ha='center', va='center', color='blue')
    ax.text(5, 1, 'Fundamental reality\nis a quantum state', fontsize=10, ha='center')
    
    # Internal structure
    for i, (x, y, d) in enumerate([(3, 6, 1), (5, 7, 2), (7, 6, 3)]):
        c = plt.Circle((x, y), 0.4, fill=True, color=['red', 'green', 'purple'][i], alpha=0.7)
        ax.add_patch(c)
        ax.text(x, y, f'{d}D', fontsize=8, ha='center', va='center', color='white', fontweight='bold')
    ax.text(5, 8.5, 'Internal dims: 1, 2, 3', fontsize=9, ha='center')
    
    # --- (0,1): Local Unitarity ---
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('2. Local Unitarity', fontsize=12, fontweight='bold')
    
    # Draw lattice with local connections
    for i in range(5):
        for j in range(5):
            x, y = 1.5 + i*1.5, 2 + j*1.5
            ax.plot(x, y, 'ko', markersize=8)
            if i < 4:
                ax.plot([x, x+1.5], [y, y], 'b-', linewidth=1, alpha=0.5)
            if j < 4:
                ax.plot([x, x], [y, y+1.5], 'b-', linewidth=1, alpha=0.5)
    
    ax.text(5, 1, 'Nearest-neighbor\ninteractions only', fontsize=10, ha='center')
    ax.text(5, 9.5, 'H = Σ hᵢⱼ', fontsize=14, ha='center')
    
    # --- (0,2): Emergence ---
    ax = axes[0, 2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('3. What Emerges', fontsize=12, fontweight='bold')
    
    emergent = [
        ('Lightcone', 'Finite propagation'),
        ('Gauge groups', 'U(1)×SU(2)×SU(3)'),
        ('Statistics', 'Bosons & Fermions'),
        ('Confinement', 'Color singlets'),
    ]
    
    for i, (name, detail) in enumerate(emergent):
        y = 8 - i*2
        rect = mpatches.FancyBboxPatch((1, y-0.5), 8, 1.3,
                                        boxstyle="round,pad=0.1",
                                        facecolor='lightgreen', edgecolor='darkgreen', linewidth=1)
        ax.add_patch(rect)
        ax.text(5, y+0.3, name, fontsize=11, fontweight='bold', ha='center')
        ax.text(5, y-0.2, detail, fontsize=9, ha='center', color='darkgreen')
    
    # --- (1,0): Falsification ---
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('4. What We Falsified', fontsize=12, fontweight='bold', color='red')
    
    # Crossed out hypothesis
    ax.text(5, 7, 'Copyability → Statistics?', fontsize=12, ha='center')
    ax.plot([1.5, 8.5], [7.5, 6.5], 'r-', linewidth=4)
    ax.plot([1.5, 8.5], [6.5, 7.5], 'r-', linewidth=4)
    
    ax.text(5, 4.5, 'Symmetric & Antisymmetric\nstates have IDENTICAL\ncopyability', 
            fontsize=10, ha='center', style='italic')
    ax.text(5, 2, '→ Statistics is GEOMETRIC,\nnot information-theoretic', 
            fontsize=11, ha='center', fontweight='bold', color='darkred')
    
    # --- (1,1): The Data ---
    ax = axes[1, 1]
    
    # Load and plot stability data
    try:
        data = np.loadtxt('data/stability_vs_phi_L32.dat', skiprows=1)
        ax.plot(data[:, 0] / np.pi, data[:, 1], 'bo-', markersize=4, linewidth=2)
        ax.set_xlabel('Exchange phase φ / π')
        ax.set_ylabel('Stability S(φ)')
        ax.set_title('5. The Data', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Annotations
        ax.annotate('Boson\n(+1)', xy=(0, 0.95), ha='center', fontsize=9, color='blue')
        ax.annotate('Fermion\n(-1)', xy=(1, 0.95), ha='center', fontsize=9, color='blue')
        ax.annotate('Suppressed', xy=(0.5, 0.15), ha='center', fontsize=9, color='red')
    except:
        ax.text(0.5, 0.5, 'Data not found', transform=ax.transAxes, ha='center')
    
    # --- (1,2): The Conclusion ---
    ax = axes[1, 2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('6. The Conclusion', fontsize=12, fontweight='bold')
    
    # Main statement
    rect = mpatches.FancyBboxPatch((0.5, 5), 9, 3,
                                    boxstyle="round,pad=0.2",
                                    facecolor='lightyellow', edgecolor='goldenrod', linewidth=3)
    ax.add_patch(rect)
    ax.text(5, 7, 'Fermions are not fundamental.', fontsize=11, ha='center', fontweight='bold')
    ax.text(5, 5.8, 'They are the only non-trivial\nbraid representation in 3D.', 
            fontsize=10, ha='center')
    
    # What's fundamental
    ax.text(5, 3.5, "What's fundamental:", fontsize=10, ha='center', fontweight='bold')
    ax.text(5, 2.5, '• Hilbert space (dims 1, 2, 3)', fontsize=9, ha='center')
    ax.text(5, 1.8, '• Unitarity', fontsize=9, ha='center')
    ax.text(5, 1.1, '• Locality', fontsize=9, ha='center')
    
    plt.tight_layout()
    plt.savefig('figures/fig_complete_story.png', dpi=200, bbox_inches='tight')
    plt.savefig('figures/fig_complete_story.pdf', bbox_inches='tight')
    plt.close()
    print("Created: fig_complete_story.png")


# =============================================================================
# Figure 4: Two-Particle Exchange Animation Frames
# =============================================================================

def create_exchange_frames():
    """Create frames showing particle exchange in different dimensions."""
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    
    phases = [0, np.pi/4, np.pi/2, np.pi]
    labels = ['φ = 0\n(Boson)', 'φ = π/4\n(Intermediate)', 'φ = π/2\n(Intermediate)', 'φ = π\n(Fermion)']
    colors = ['blue', 'purple', 'orange', 'red']
    stabilities = ['STABLE', 'UNSTABLE', 'UNSTABLE', 'STABLE']
    
    for ax, phi, label, color, stab in zip(axes, phases, labels, colors, stabilities):
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw exchange paths
        theta = np.linspace(0, np.pi, 50)
        
        # Particle A: left to right along top semicircle
        x1 = np.cos(theta)
        y1 = 0.5 * np.sin(theta)
        ax.plot(x1, y1, color=color, linewidth=3, alpha=0.7)
        ax.plot(-1, 0, 'o', color=color, markersize=15)
        ax.plot(1, 0, 'o', color=color, markersize=15, alpha=0.5)
        
        # Particle B: right to left along bottom semicircle
        x2 = -np.cos(theta)
        y2 = -0.5 * np.sin(theta)
        ax.plot(x2, y2, color=color, linewidth=3, alpha=0.7, linestyle='--')
        ax.plot(1, 0, 's', color=color, markersize=12)
        ax.plot(-1, 0, 's', color=color, markersize=12, alpha=0.5)
        
        # Phase indicator
        ax.text(0, -1.0, label, fontsize=11, ha='center', fontweight='bold', color=color)
        
        # Stability indicator
        if stab == 'STABLE':
            ax.text(0, -1.4, stab, fontsize=10, ha='center', color='green', fontweight='bold')
        else:
            ax.text(0, -1.4, stab, fontsize=10, ha='center', color='red')
    
    plt.suptitle('Exchange Phase and Stability in 3D', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig_exchange_phases.png', dpi=200, bbox_inches='tight')
    plt.savefig('figures/fig_exchange_phases.pdf', bbox_inches='tight')
    plt.close()
    print("Created: fig_exchange_phases.png")


# =============================================================================
# Main
# =============================================================================

def main():
    import os
    os.makedirs('figures', exist_ok=True)
    
    print("=" * 60)
    print("GENERATING ENHANCED VISUALIZATIONS")
    print("=" * 60)
    
    create_1d_evolution()
    create_topology_comparison()
    create_complete_story()
    create_exchange_frames()
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 60)
    print("\nNew figures:")
    print("  • fig_1d_evolution.png - Substrate dynamics")
    print("  • fig_topology_comparison.png - 2D vs 3D topology")  
    print("  • fig_complete_story.png - Full framework summary")
    print("  • fig_exchange_phases.png - Exchange phase stability")


if __name__ == "__main__":
    main()