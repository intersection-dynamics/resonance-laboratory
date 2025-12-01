#!/usr/bin/env python3
"""
debug_defrag_effect.py

Minimal sanity check that defrag_rate actually affects the substrate.

We run two substrates with identical initial conditions:
  - cold: defrag_rate = 0.0
  - hot : defrag_rate = 0.4

We then track a simple "node state decorrelation" metric:
  - For each run, pick a fixed list of nodes at t=0.
  - Record their initial direction_amplitudes().
  - Evolve for N steps and measure the average normalized
    complex inner product between the initial and current
    direction_amplitudes for those nodes.

If defrag is doing something, the hot run should decorrelate
those node states faster than the cold run.
"""

import numpy as np
from substrate import Config, Substrate  # type: ignore


def to_numpy(x):
    """Convert a NumPy or CuPy array-like to a NumPy array, preserving complex dtype."""
    if hasattr(x, "get"):
        return x.get()
    return np.array(x)


def node_inner_products(sub: Substrate, ref_states, node_ids):
    """
    Average normalized complex inner product between current and reference
    direction_amplitudes for a small set of nodes.
    """
    vals = []
    for nid, ref in zip(node_ids, ref_states):
        node = sub.nodes[nid]
        cur = to_numpy(node.direction_amplitudes())

        ref = np.asarray(ref)
        n_cur = np.linalg.norm(cur)
        n_ref = np.linalg.norm(ref)
        if n_cur == 0.0 or n_ref == 0.0:
            vals.append(0.0)
        else:
            # Use complex inner product; take real part for reporting
            val = np.vdot(ref / n_ref, cur / n_cur)
            vals.append(float(val.real))
    return float(np.mean(vals))


def make_substrate(defrag_rate: float, seed: int) -> Substrate:
    config = Config(
        n_nodes=32,
        internal_dim=3,
        monogamy_budget=1.0,
        defrag_rate=defrag_rate,
        dt=0.1,
        seed=seed,
    )
    return Substrate(config)


def run_single(defrag_rate: float, n_steps: int = 400) -> np.ndarray:
    seed = 12345
    np.random.seed(seed)

    sub = make_substrate(defrag_rate=defrag_rate, seed=seed)

    node_ids = sorted(list(sub.nodes.keys()))
    node_ids = node_ids[: min(20, len(node_ids))]
    ref_states = [
        to_numpy(sub.nodes[nid].direction_amplitudes()).copy()
        for nid in node_ids
    ]

    overlaps = []
    for step in range(n_steps):
        sub.evolve(dt=0.1, n_steps=1, defrag_rate=defrag_rate)
        overlaps.append(node_inner_products(sub, ref_states, node_ids))

    return np.array(overlaps)


def main():
    cold = run_single(defrag_rate=0.0, n_steps=400)
    hot = run_single(defrag_rate=0.4, n_steps=400)

    print("Step  cold_overlap   hot_overlap")
    for i in range(0, len(cold), 40):
        print(f"{i:4d}  {cold[i]:12.6f}  {hot[i]:12.6f}")

    print("\nMean overlap over last 100 steps:")
    print(f"  cold: {np.mean(cold[-100:]):.6f}")
    print(f"  hot : {np.mean(hot[-100:]):.6f}")


if __name__ == "__main__":
    main()
