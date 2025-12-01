#!/usr/bin/env python3
"""
test_substrate_topology.py

Test the new substrate features:
  - defrag_sharpen_alpha: does sharpening concentrate entanglement?
  - defrag_cutoff: does the graph actually sparsify over time?
  - Does graph diameter increase as edges are pruned?

This probes whether "space" can emerge from entanglement dynamics.
"""

import numpy as np
from substrate import Config, Substrate


def measure_graph_stats(substrate: Substrate) -> dict:
    """Compute basic graph statistics."""
    n = substrate.n_nodes
    neighbors = substrate._neighbors
    
    # Degree distribution
    degrees = [len(neighbors[i]) for i in range(n)]
    mean_degree = np.mean(degrees)
    min_degree = min(degrees)
    max_degree = max(degrees)
    
    # Sample graph diameter (expensive for large graphs, so sample)
    sample_size = min(20, n)
    sample_nodes = np.random.choice(n, size=sample_size, replace=False)
    
    max_dist = 0
    for i in sample_nodes:
        for j in sample_nodes:
            if i < j:
                d = substrate.graph_distance(i, j, max_radius=20)
                if not np.isinf(d):
                    max_dist = max(max_dist, int(d))
    
    # Count edges
    n_edges = substrate.num_edges
    
    # Coupling magnitude distribution
    J_np = substrate.couplings
    if hasattr(J_np, 'get'):
        J_np = J_np.get()
    mags = np.abs(J_np).flatten()
    mags = mags[mags > 0]
    
    return {
        "n_edges": n_edges,
        "mean_degree": mean_degree,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "sample_diameter": max_dist,
        "mean_coupling": float(np.mean(mags)) if len(mags) > 0 else 0.0,
        "std_coupling": float(np.std(mags)) if len(mags) > 0 else 0.0,
        "max_coupling": float(np.max(mags)) if len(mags) > 0 else 0.0,
    }


def test_baseline(n_steps=500):
    """Test baseline: moderate p, no cutoff."""
    print("\n" + "="*60)
    print("TEST 1: Baseline (p=0.8, q=1.5, cutoff=0.0)")
    print("="*60)
    
    config = Config(
        n_nodes=64,
        internal_dim=3,
        connectivity=0.25,
        defrag_rate=0.2,
        pressure_p=0.8,  # closer to 1 = less sparsification
        pressure_q=1.5,
        defrag_cutoff=0.0,
        seed=42,
    )
    
    sub = Substrate(config)
    
    print("\nInitial state:")
    stats = measure_graph_stats(sub)
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print(f"\nEvolving {n_steps} steps...")
    sub.evolve(n_steps=n_steps)
    
    print("\nFinal state:")
    stats = measure_graph_stats(sub)
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


def test_sharpening(n_steps=500):
    """Test strong sparsification: p < 0.5 drives small couplings to zero faster."""
    print("\n" + "="*60)
    print("TEST 2: Strong Sparsification (p=0.3, q=2.0, cutoff=0.0)")
    print("="*60)
    
    config = Config(
        n_nodes=64,
        internal_dim=3,
        connectivity=0.25,
        defrag_rate=0.2,
        pressure_p=0.3,  # smaller p = stronger sparsification
        pressure_q=2.0,
        defrag_cutoff=0.0,
        seed=42,
    )
    
    sub = Substrate(config)
    
    print("\nInitial state:")
    stats = measure_graph_stats(sub)
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print(f"\nEvolving {n_steps} steps...")
    sub.evolve(n_steps=n_steps)
    
    print("\nFinal state:")
    stats = measure_graph_stats(sub)
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print("\nExpect: higher std_coupling, higher max_coupling (concentration)")


def test_cutoff(n_steps=500):
    """Test cutoff: small edges should be pruned, graph should sparsify."""
    print("\n" + "="*60)
    print("TEST 3: Cutoff (alpha=1.0, cutoff=0.01)")
    print("="*60)
    
    config = Config(
        n_nodes=64,
        internal_dim=3,
        connectivity=0.25,
        defrag_rate=0.2,
        defrag_sharpen_alpha=1.0,
        defrag_cutoff=0.01,  # Prune weak edges
        seed=42,
    )
    
    sub = Substrate(config)
    
    print("\nInitial state:")
    stats = measure_graph_stats(sub)
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print(f"\nEvolving {n_steps} steps...")
    sub.evolve(n_steps=n_steps)
    
    print("\nFinal state:")
    stats = measure_graph_stats(sub)
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print("\nExpect: fewer edges, possibly larger diameter (space emerges)")


def test_sharpen_plus_cutoff(n_steps=1000):
    """Test both: aggressive locality emergence."""
    print("\n" + "="*60)
    print("TEST 4: Sharpen + Cutoff (alpha=1.5, cutoff=0.02)")
    print("="*60)
    
    config = Config(
        n_nodes=64,
        internal_dim=3,
        connectivity=0.30,  # Start denser
        defrag_rate=0.2,
        defrag_sharpen_alpha=1.5,
        defrag_cutoff=0.02,
        seed=42,
    )
    
    sub = Substrate(config)
    
    print("\nInitial state:")
    stats_init = measure_graph_stats(sub)
    for k, v in stats_init.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Evolve in stages to watch progression
    checkpoints = [100, 250, 500, 750, 1000]
    
    for cp in checkpoints:
        steps_to_run = cp - (sub.config.seed if cp == checkpoints[0] else checkpoints[checkpoints.index(cp)-1])
        # Actually just run to checkpoint
        pass
    
    print(f"\nEvolving {n_steps} steps with checkpoints...")
    step = 0
    for cp in checkpoints:
        steps_now = cp - step
        sub.evolve(n_steps=steps_now)
        step = cp
        stats = measure_graph_stats(sub)
        print(f"\n  Step {cp}:")
        print(f"    edges={stats['n_edges']}, mean_deg={stats['mean_degree']:.1f}, "
              f"diameter={stats['sample_diameter']}, max_J={stats['max_coupling']:.3f}")
    
    print("\nExpect: edges decrease, diameter increase (locality emerges)")


def test_extreme_sparsification(n_steps=2000):
    """Push hard: can we get a sparse graph with real structure?"""
    print("\n" + "="*60)
    print("TEST 5: Extreme Sparsification (alpha=2.0, cutoff=0.05)")
    print("="*60)
    
    config = Config(
        n_nodes=128,
        internal_dim=3,
        connectivity=0.20,
        defrag_rate=0.3,
        defrag_sharpen_alpha=2.0,
        defrag_cutoff=0.05,
        seed=123,
    )
    
    sub = Substrate(config)
    
    print("\nInitial state:")
    stats = measure_graph_stats(sub)
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print(f"\nEvolving {n_steps} steps...")
    
    # Check for disconnection
    for checkpoint in [500, 1000, 1500, 2000]:
        sub.evolve(n_steps=500 if checkpoint == 500 else 500)
        stats = measure_graph_stats(sub)
        disconnected = stats['min_degree'] == 0
        print(f"  Step {checkpoint}: edges={stats['n_edges']}, "
              f"mean_deg={stats['mean_degree']:.1f}, "
              f"min_deg={stats['min_degree']}, "
              f"diameter={stats['sample_diameter']}"
              f"{' [DISCONNECTED!]' if disconnected else ''}")
    
    print("\nFinal coupling distribution:")
    J_np = sub.couplings
    if hasattr(J_np, 'get'):
        J_np = J_np.get()
    mags = np.abs(J_np).flatten()
    mags = mags[mags > 0]
    if len(mags) > 0:
        print(f"  Non-zero couplings: {len(mags)//2}")
        print(f"  Mean: {np.mean(mags):.4f}")
        print(f"  Std:  {np.std(mags):.4f}")
        print(f"  Max:  {np.max(mags):.4f}")
        print(f"  Min:  {np.min(mags):.4f}")


def main():
    print("Testing new substrate topology features")
    print("="*60)
    
    test_baseline(n_steps=500)
    test_sharpening(n_steps=500)
    test_cutoff(n_steps=500)
    test_sharpen_plus_cutoff(n_steps=1000)
    test_extreme_sparsification(n_steps=2000)
    
    print("\n" + "="*60)
    print("All tests complete.")
    print("="*60)


if __name__ == "__main__":
    main()