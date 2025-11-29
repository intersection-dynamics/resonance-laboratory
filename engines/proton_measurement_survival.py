#!/usr/bin/env python3
"""
proton_measurement_survival.py  (lightweight version)

Long-run repeated-measurement experiment for a single proton-like node
in the batched Hilbert substrate.

Key questions:
  - Does a proton-like excitation survive repeated measurement?
  - Does its internal coherence + localization behave differently
    with strong vs weak measurement?

Design:
  - We avoid expensive per-step entanglement SVDs.
  - We use degree × localization as a "proton score".
  - We compute full entanglement only at the start and end.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
from substrate import Config, Substrate  # type: ignore

# CuPy-aware, but all analysis is done in NumPy
try:
    import cupy as cp
    HAVE_CUPY = True
except Exception:
    cp = None
    HAVE_CUPY = False


# ---------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------

def to_numpy(x) -> np.ndarray:
    """Convert a state to NumPy, handling CuPy if present."""
    if HAVE_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


# ---------------------------------------------------------------------
# Entropy / localization / scoring
# ---------------------------------------------------------------------

def _node_entropy(prob: np.ndarray) -> float:
    """Shannon entropy of a probability vector (natural log)."""
    p = np.asarray(prob, dtype=float)
    total = max(p.sum(), 1e-12)
    p = p / total
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def _get_node_state(node) -> np.ndarray:
    """
    Extract internal pointer amplitudes for a node as a NumPy array.
    """
    psi = None
    if hasattr(node, "direction_amplitudes"):
        try:
            psi = node.direction_amplitudes()
        except Exception:
            psi = None
    if psi is None and hasattr(node, "state"):
        psi = node.state
    if psi is None:
        return np.ones(1, dtype=np.complex128)
    return to_numpy(psi).astype(np.complex128)


def _get_node_degree(node) -> int:
    """Graph degree (no entanglement SVD here)."""
    if hasattr(node, "n_connections"):
        try:
            return int(node.n_connections)
        except Exception:
            pass
    if hasattr(node, "neighbor_ids"):
        try:
            return len(node.neighbor_ids)
        except Exception:
            pass
    return 0


def localization_from_state(psi: np.ndarray) -> float:
    """
    Localization = 1 - (S / log d)
    where S is entropy of |psi|^2.
    """
    d = psi.shape[0]
    probs = np.abs(psi) ** 2
    S = _node_entropy(probs)
    logd = np.log(d) if d > 1 else 1.0
    return float(1.0 - S / logd)


def proton_score_light(node) -> float:
    """
    Lightweight proton score:
        score = degree * localization

    No entanglement SVD inside.
    """
    psi = _get_node_state(node)
    loc = localization_from_state(psi)
    deg = _get_node_degree(node)
    return float(deg * loc)


def find_proton_candidate_light(substrate: Substrate):
    """
    Pick the node with largest degree × localization.
    No entanglement SVD here.
    """
    best_id = None
    best_score = -np.inf

    for node_id, node in substrate.nodes.items():
        try:
            s = proton_score_light(node)
        except Exception:
            continue
        if not np.isfinite(s):
            continue
        if s > best_score:
            best_score = s
            best_id = node_id

    if best_id is None:
        ids = list(substrate.nodes.keys())
        if not ids:
            raise RuntimeError("Substrate has no nodes.")
        best_id = ids[0]
        best_score = 0.0

    return best_id, float(best_score)


# ---------------------------------------------------------------------
# Coherence / pointer projection
# ---------------------------------------------------------------------

def coherence_metric(states: np.ndarray) -> np.ndarray:
    """
    Off-diagonal coherence magnitude for a single node over time.

    states: (T, d) pure state vectors for one node.
    Returns:
        coh: (T,) float64  sum |rho_ij| for i != j
    """
    T, d = states.shape
    coh = np.zeros(T, dtype=float)
    for t in range(T):
        psi = states[t]
        rho = np.outer(psi, np.conjugate(psi))
        off = rho - np.diag(np.diag(rho))
        coh[t] = float(np.sum(np.abs(off)))
    return coh


def make_pointer_state(psi: np.ndarray, axis_index: int | None = None) -> Tuple[np.ndarray, int]:
    """
    Build a pointer state along a chosen basis axis.

    If axis_index is None, pick the dominant axis by |psi_k|^2.
    Keep the phase of psi on that axis.
    """
    psi = np.asarray(psi, dtype=np.complex128)
    d = psi.shape[0]
    if d == 0:
        raise ValueError("Pointer state requested for zero-dimensional psi.")

    probs = np.abs(psi) ** 2
    if axis_index is None:
        k = int(np.argmax(probs))
    else:
        k = int(axis_index)
        if not (0 <= k < d):
            raise ValueError(f"axis_index {k} out of range for dim={d}")

    pointer = np.zeros(d, dtype=np.complex128)
    phase = np.angle(psi[k]) if np.abs(psi[k]) > 1e-14 else 0.0
    pointer[k] = np.exp(1j * phase)

    norm = np.linalg.norm(pointer)
    if norm > 0:
        pointer /= norm
    return pointer, k


# ---------------------------------------------------------------------
# Core survival experiment (lightweight)
# ---------------------------------------------------------------------

def run_measurement_survival(
    config: Config,
    burn_in_steps: int = 300,
    total_steps: int = 2000,
    meas_every: int = 1,
    record_stride: int = 5,
) -> None:
    """
    Run a long survival test for one proton-like node under repeated measurement.

    - Use degree × localization as proton score.
    - Compute total_entanglement only at start and end.
    - Record every `record_stride` steps to cut overhead.
    """

    print("------------------------------------------------------------")
    print(f"Measurement-survival run (meas_every = {meas_every})")
    print(f"  burn_in_steps = {burn_in_steps}, total_steps = {total_steps}, record_stride = {record_stride}")
    print("------------------------------------------------------------")

    substrate = Substrate(config)

    # Burn-in
    print("Burn-in evolution...")
    substrate.evolve(n_steps=burn_in_steps)

    # Proton candidate + initial diagnostics
    proton_id, initial_score = find_proton_candidate_light(substrate)
    proton_node = substrate.nodes[proton_id]
    print(f"Selected proton candidate id={proton_id}, initial light score={initial_score:.3f}")

    # Entanglement only once here
    try:
        ent_initial = float(proton_node.total_entanglement)
    except Exception:
        ent_initial = float("nan")

    # Fix pointer axis and collapse at t=0
    psi0 = _get_node_state(proton_node).copy()
    pointer_state_np, axis_idx = make_pointer_state(psi0, axis_index=None)
    proton_node.state = pointer_state_np.copy()
    print(f"Pointer axis index = {axis_idx}")

    dt = getattr(config, "dt", 0.1)

    # Time series
    times: List[float] = []
    pointer_probs: List[float] = []
    locs: List[float] = []
    light_scores: List[float] = []
    states: List[np.ndarray] = []

    # Degree is structural; treat as constant
    base_degree = _get_node_degree(proton_node)

    def record_step(step_idx: int, t: float):
        node = substrate.nodes[proton_id]
        psi = _get_node_state(node)
        nrm = np.linalg.norm(psi)
        if nrm > 0:
            psi = psi / nrm
            node.state = psi.copy()

        states.append(psi.copy())
        times.append(t)

        if axis_idx < psi.shape[0]:
            p_axis = float(np.abs(psi[axis_idx]) ** 2)
        else:
            p_axis = 0.0
        pointer_probs.append(p_axis)

        loc = localization_from_state(psi)
        score_light = base_degree * loc

        locs.append(loc)
        light_scores.append(score_light)

    # t=0
    t = 0.0
    record_step(step_idx=0, t=t)

    # Main loop
    for step in range(1, total_steps + 1):
        if meas_every > 0 and (step % meas_every == 0):
            node = substrate.nodes[proton_id]
            psi_pre = _get_node_state(node).copy()
            pointer_state_np, _ = make_pointer_state(psi_pre, axis_index=axis_idx)
            node.state = pointer_state_np.copy()

        substrate.evolve(n_steps=1)
        t += dt

        if step % record_stride == 0 or step == total_steps:
            record_step(step_idx=step, t=t)

    # End entanglement (single expensive call)
    proton_node = substrate.nodes[proton_id]
    try:
        ent_final = float(proton_node.total_entanglement)
    except Exception:
        ent_final = float("nan")

    times_arr = np.asarray(times, dtype=float)
    states_arr = np.asarray(states, dtype=np.complex128)
    pointer_probs_arr = np.asarray(pointer_probs, dtype=float)
    locs_arr = np.asarray(locs, dtype=float)
    scores_arr = np.asarray(light_scores, dtype=float)
    coherence_arr = coherence_metric(states_arr)

    # Survival summaries
    frac_pointer_high = float(np.mean(pointer_probs_arr > 0.5))
    score_ratio = float(scores_arr[-1] / scores_arr[0]) if scores_arr[0] != 0 else np.nan
    ent_ratio = (
        float(ent_final / ent_initial)
        if (np.isfinite(ent_initial) and ent_initial != 0)
        else np.nan
    )

    print(f"  Fraction of time pointer_prob > 0.5: {frac_pointer_high:.3f}")
    print(f"  Light score ratio (final/initial):   {score_ratio:.3f}")
    print(f"  Entanglement ratio (final/initial):  {ent_ratio:.3f}")
    print(f"  Coherence range:   {coherence_arr.min():.3e} -> {coherence_arr.max():.3e}")
    print(f"  Localization range:{locs_arr.min():.3e} -> {locs_arr.max():.3e}")

    meta: Dict[str, object] = {
        "config": asdict(config),
        "burn_in_steps": int(burn_in_steps),
        "total_steps": int(total_steps),
        "meas_every": int(meas_every),
        "record_stride": int(record_stride),
        "axis_index": int(axis_idx),
        "proton_id": str(proton_id),
        "initial_light_score": float(scores_arr[0]),
        "final_light_score": float(scores_arr[-1]),
        "initial_entanglement": float(ent_initial),
        "final_entanglement": float(ent_final),
        "light_score_ratio": score_ratio,
        "entanglement_ratio": ent_ratio,
        "fraction_time_pointer_prob_gt_0p5": frac_pointer_high,
    }

    fname = f"proton_measurement_survival_me{meas_every}.npz"
    np.savez(
        fname,
        times=times_arr,
        states=states_arr,
        coherence=coherence_arr,
        pointer_probs=pointer_probs_arr,
        localization=locs_arr,
        light_scores=scores_arr,
        meta_json=json.dumps(meta, indent=2),
    )

    print(f"Saved {fname} (T={len(times_arr)}, dim={states_arr.shape[1]})")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # Reasonable defaults for a run that actually completes
    config = Config(
        n_nodes=64,
        internal_dim=3,
        monogamy_budget=1.0,
        defrag_rate=0.1,
        seed=42,
    )
    if not hasattr(config, "dt"):
        config.dt = 0.1  # type: ignore[attr-defined]

    print("============================================================")
    print("  Proton measurement survival experiment (lightweight)")
    print("============================================================")
    print(
        f"Config: n_nodes={config.n_nodes}, d={config.internal_dim}, "
        f"monogamy={config.monogamy_budget}, defrag={config.defrag_rate}, "
        f"dt={getattr(config, 'dt', 'N/A')}"
    )
    print("============================================================\n")

    burn_in_steps = 300
    total_steps = 2000
    record_stride = 5

    # Minimal set of measurement schedules:
    #  - 0: no measurement (baseline)
    #  - 1: measure every step (max Zeno)
    #  - 8: intermediate
    measurement_intervals = [0, 1, 8]

    for me in measurement_intervals:
        run_measurement_survival(
            config=config,
            burn_in_steps=burn_in_steps,
            total_steps=total_steps,
            meas_every=me,
            record_stride=record_stride,
        )

    print("\nAll measurement-survival runs complete.")


if __name__ == "__main__":
    main()
