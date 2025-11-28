#!/usr/bin/env python3
"""
Universality Test for Hilbert Substrate Framework
==================================================
Tests whether key results are universal (parameter-independent):
    1. d=1 → ALL exchange phases = +1 (bosonic)
    2. d=2 → R(2π) = -1 (fermionic sign)
    3. d=3 → SU(3) color structure
    
Sweeps across:
    - System size
    - Random seeds (initial conditions)
    - Defrag rate
    - Monogamy budget
    - Evolution time
"""

import numpy as np
from substrate import Substrate, Config, run_simulation
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time


@dataclass
class TestResult:
    """Result of a single universality test."""
    internal_dim: int
    param_name: str
    param_value: float
    seed: int
    
    # Key observables
    bosonic_fraction: float      # Fraction of +1 exchange phases
    fermionic_fraction: float    # Fraction of -1 exchange phases
    R_2pi: complex               # R(2π) phase (for d=2)
    mean_degree: float
    effective_dim: float
    
    @property
    def is_bosonic(self) -> bool:
        return self.bosonic_fraction > 0.95
    
    @property
    def is_fermionic_rotation(self) -> bool:
        if self.internal_dim != 2:
            return False
        return np.abs(self.R_2pi + 1) < 0.01


def run_single_test(config: Config, n_steps: int = 100) -> TestResult:
    """Run a single test and extract key observables."""
    substrate, history = run_simulation(config, n_steps=n_steps, record_every=n_steps)
    
    snap = history[-1]
    ex = snap['exchange']
    total = ex['near_+1'] + ex['near_-1'] + ex['intermediate']
    
    R_2pi = snap.get('R_2pi', 1.0)
    if isinstance(R_2pi, dict):
        R_2pi = complex(R_2pi['real'], R_2pi['imag'])
    elif R_2pi is None:
        R_2pi = 1.0
    
    return TestResult(
        internal_dim=config.internal_dim,
        param_name="",
        param_value=0,
        seed=config.seed or 0,
        bosonic_fraction=ex['near_+1'] / max(1, total),
        fermionic_fraction=ex['near_-1'] / max(1, total),
        R_2pi=R_2pi,
        mean_degree=snap['mean_degree'],
        effective_dim=snap['effective_dimension']
    )


def test_across_seeds(internal_dim: int, n_seeds: int = 20) -> List[TestResult]:
    """Test across different random initial conditions."""
    results = []
    
    for seed in range(n_seeds):
        config = Config(
            n_nodes=48,
            internal_dim=internal_dim,
            defrag_rate=0.05,
            seed=seed
        )
        result = run_single_test(config)
        result.param_name = "seed"
        result.param_value = seed
        results.append(result)
    
    return results


def test_across_sizes(internal_dim: int, sizes: List[int] = None) -> List[TestResult]:
    """Test across different system sizes."""
    if sizes is None:
        sizes = [16, 32, 48, 64, 96]
    
    results = []
    
    for n in sizes:
        config = Config(
            n_nodes=n,
            internal_dim=internal_dim,
            defrag_rate=0.05,
            seed=42
        )
        result = run_single_test(config)
        result.param_name = "n_nodes"
        result.param_value = n
        results.append(result)
    
    return results


def test_across_defrag(internal_dim: int, rates: List[float] = None) -> List[TestResult]:
    """Test across different defragmentation rates."""
    if rates is None:
        rates = [0.01, 0.02, 0.05, 0.1, 0.2]
    
    results = []
    
    for rate in rates:
        config = Config(
            n_nodes=48,
            internal_dim=internal_dim,
            defrag_rate=rate,
            seed=42
        )
        result = run_single_test(config)
        result.param_name = "defrag_rate"
        result.param_value = rate
        results.append(result)
    
    return results


def test_across_monogamy(internal_dim: int, budgets: List[float] = None) -> List[TestResult]:
    """Test across different monogamy budgets."""
    if budgets is None:
        budgets = [0.5, 0.75, 1.0, 1.5, 2.0]
    
    results = []
    
    for budget in budgets:
        config = Config(
            n_nodes=48,
            internal_dim=internal_dim,
            monogamy_budget=budget,
            defrag_rate=0.05,
            seed=42
        )
        result = run_single_test(config)
        result.param_name = "monogamy"
        result.param_value = budget
        results.append(result)
    
    return results


def test_across_evolution(internal_dim: int, steps: List[int] = None) -> List[TestResult]:
    """Test across different evolution times."""
    if steps is None:
        steps = [50, 100, 200, 400]
    
    results = []
    
    for n_steps in steps:
        config = Config(
            n_nodes=48,
            internal_dim=internal_dim,
            defrag_rate=0.05,
            seed=42
        )
        result = run_single_test(config, n_steps=n_steps)
        result.param_name = "n_steps"
        result.param_value = n_steps
        results.append(result)
    
    return results


def summarize_results(results: List[TestResult], test_name: str) -> Dict:
    """Summarize a batch of test results."""
    bosonic = [r.bosonic_fraction for r in results]
    fermionic = [r.fermionic_fraction for r in results]
    R_2pi = [r.R_2pi for r in results if r.internal_dim == 2]
    
    summary = {
        'test': test_name,
        'n_trials': len(results),
        'bosonic_mean': np.mean(bosonic),
        'bosonic_std': np.std(bosonic),
        'bosonic_min': np.min(bosonic),
        'fermionic_mean': np.mean(fermionic),
        'all_bosonic': all(r.is_bosonic for r in results),
    }
    
    if R_2pi:
        R_vals = [r.real for r in R_2pi]
        summary['R_2pi_mean'] = np.mean(R_vals)
        summary['R_2pi_std'] = np.std(R_vals)
        summary['all_fermionic_rotation'] = all(
            r.is_fermionic_rotation for r in results if r.internal_dim == 2
        )
    
    return summary


def print_summary(summary: Dict, dim: int):
    """Print summary for one test."""
    print(f"  {summary['test']}: ", end="")
    
    if dim == 1:
        if summary['all_bosonic']:
            print(f"✓ ALL BOSONIC (mean={summary['bosonic_mean']:.3f})")
        else:
            print(f"✗ Not all bosonic (mean={summary['bosonic_mean']:.3f}, min={summary['bosonic_min']:.3f})")
    
    elif dim == 2:
        if summary.get('all_fermionic_rotation', False):
            print(f"✓ R(2π)=-1 (mean={summary.get('R_2pi_mean', 0):.4f} ± {summary.get('R_2pi_std', 0):.4f})")
        else:
            print(f"? R(2π)={summary.get('R_2pi_mean', 0):.4f} ± {summary.get('R_2pi_std', 0):.4f}")
    
    else:
        print(f"bosonic={summary['bosonic_mean']:.3f}, fermionic={summary['fermionic_mean']:.3f}")


def run_universality_tests():
    """Run complete universality test suite."""
    
    print("=" * 60)
    print("UNIVERSALITY TEST")
    print("=" * 60)
    print("""
Testing whether key results hold across:
  - Random initial conditions (20 seeds)
  - System sizes (16 to 96 nodes)
  - Defrag rates (0.01 to 0.2)
  - Monogamy budgets (0.5 to 2.0)
  - Evolution times (50 to 400 steps)

Universal predictions:
  d=1: ALL exchange phases = +1 (bosonic)
  d=2: R(2π) = -1 (fermionic sign)
""")
    
    all_results = {}
    
    for d in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"INTERNAL DIMENSION d = {d}")
        print("=" * 60)
        
        dim_results = {}
        
        # Test across seeds
        print("\n[1] Testing across random seeds...")
        results = test_across_seeds(d, n_seeds=20)
        summary = summarize_results(results, "seeds")
        print_summary(summary, d)
        dim_results['seeds'] = summary
        
        # Test across sizes
        print("\n[2] Testing across system sizes...")
        results = test_across_sizes(d)
        summary = summarize_results(results, "sizes")
        print_summary(summary, d)
        dim_results['sizes'] = summary
        
        # Test across defrag rates
        print("\n[3] Testing across defrag rates...")
        results = test_across_defrag(d)
        summary = summarize_results(results, "defrag")
        print_summary(summary, d)
        dim_results['defrag'] = summary
        
        # Test across monogamy budgets
        print("\n[4] Testing across monogamy budgets...")
        results = test_across_monogamy(d)
        summary = summarize_results(results, "monogamy")
        print_summary(summary, d)
        dim_results['monogamy'] = summary
        
        # Test across evolution times
        print("\n[5] Testing across evolution times...")
        results = test_across_evolution(d)
        summary = summarize_results(results, "evolution")
        print_summary(summary, d)
        dim_results['evolution'] = summary
        
        all_results[d] = dim_results
    
    # Final summary
    print("\n" + "=" * 60)
    print("UNIVERSALITY SUMMARY")
    print("=" * 60)
    
    # d=1 test
    d1_tests = all_results[1]
    d1_universal = all(s['all_bosonic'] for s in d1_tests.values())
    print(f"\nd=1 (U(1)): Exchange = +1")
    print(f"  {'✓ UNIVERSAL' if d1_universal else '✗ NOT UNIVERSAL'}")
    print(f"  Passed: {sum(1 for s in d1_tests.values() if s['all_bosonic'])}/5 tests")
    
    # d=2 test
    d2_tests = all_results[2]
    d2_universal = all(s.get('all_fermionic_rotation', False) for s in d2_tests.values())
    print(f"\nd=2 (SU(2)): R(2π) = -1")
    print(f"  {'✓ UNIVERSAL' if d2_universal else '✗ NOT UNIVERSAL'}")
    r2pi_vals = [s.get('R_2pi_mean', 0) for s in d2_tests.values()]
    print(f"  R(2π) range: [{min(r2pi_vals):.4f}, {max(r2pi_vals):.4f}]")
    
    # d=3 summary
    d3_tests = all_results[3]
    print(f"\nd=3 (SU(3)): Color structure")
    print(f"  Mean bosonic fraction: {np.mean([s['bosonic_mean'] for s in d3_tests.values()]):.3f}")
    print(f"  (Intermediate phases expected due to color mixing)")
    
    return all_results


def run_critical_test():
    """Run the most critical test: is R(2π) = -1 for d=2?"""
    
    print("\n" + "=" * 60)
    print("CRITICAL TEST: FERMIONIC SIGN FROM SU(2)")
    print("=" * 60)
    
    n_trials = 50
    R_values = []
    
    for seed in range(n_trials):
        config = Config(
            n_nodes=32,
            internal_dim=2,
            defrag_rate=0.05,
            seed=seed
        )
        substrate = Substrate(config)
        R = substrate.rotation_phase(2 * np.pi)
        R_values.append(R)
    
    R_real = [r.real for r in R_values]
    
    print(f"\nn = {n_trials} independent trials")
    print(f"R(2π) mean: {np.mean(R_real):.6f}")
    print(f"R(2π) std:  {np.std(R_real):.6f}")
    print(f"R(2π) min:  {np.min(R_real):.6f}")
    print(f"R(2π) max:  {np.max(R_real):.6f}")
    
    all_minus_one = all(np.abs(r + 1) < 0.0001 for r in R_real)
    
    print(f"\nAll exactly -1? {'YES' if all_minus_one else 'NO'}")
    
    if all_minus_one:
        print("""
CONCLUSION: The fermionic sign R(2π) = -1 emerges UNIVERSALLY
from SU(2) internal structure. This is not a parameter choice
or initial condition - it's a mathematical consequence of the
double cover structure of SU(2).
""")
    
    return R_values


if __name__ == "__main__":
    t0 = time.time()
    
    # Run critical test first
    R_values = run_critical_test()
    
    # Run full universality suite
    results = run_universality_tests()
    
    print(f"\nTotal time: {time.time() - t0:.1f}s")