#!/usr/bin/env python3
"""
Proton Pointer Scope
====================

Pipeline:
    1. Run the monogamy/defrag substrate (d = 3 → SU(3)-like sector).
    2. Identify a "proton-like" node:
        - high degree (lots of neighbors, ~3D-ish connectivity)
        - high total entanglement
        - high internal entropy (uses all 3 internal directions)
    3. "Probe" it:
        - project its internal state onto a single pointer basis vector
          (the component with the largest |ψ_i|²).
    4. After the probe, evolve the substrate and record:
        - time series of the node's internal state components
        - amplitude maps before/after probe
        - a crude "decoherence" metric (off-diagonal coherence size)
        - an optional wavelet / time–frequency scalogram if SciPy supports it
    5. Save everything to proton_pointer_scope.npz for further analysis.
"""

import os
import json
from typing import Dict, Any, Tuple

import numpy as np

# Try to use SciPy for wavelet scalograms; fall back gracefully if not available
try:
    from scipy import signal as _signal  # type: ignore

    _HAS_SCIPY = hasattr(_signal, "cwt") and hasattr(_signal, "morlet2")
except Exception:
    _HAS_SCIPY = False

# This substrate is the monogamy/defrag engine (Config, Substrate, run_simulation)
from substrate import Config, Substrate, run_simulation  # type: ignore


# =============================================================================
# 1. "PROTON-LIKE" NODE FINDER
# =============================================================================

def _node_entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability vector p (natural log)."""
    p = np.asarray(p, dtype=float)
    p = p / max(p.sum(), 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def find_proton_candidate(substrate: Substrate) -> int:
    """
    Heuristic "proton pointer" finder.

    Score for each node i:
        score_i = degree_i * entanglement_i * (S_i / log(d))

    where:
        - degree_i       = number of neighbors
        - entanglement_i = total entanglement budget used
        - S_i            = entropy of internal amplitudes |ψ|²
        - d              = internal_dim (here: 3 → "color" sector)

    Returns the node id with the largest score.
    """
    d = substrate.config.internal_dim
    logd = np.log(d) if d > 1 else 1.0

    best_id = None
    best_score = -1.0

    for node_id, node in substrate.nodes.items():
        degree = node.n_connections
        ent = node.total_entanglement
        amps = node.direction_amplitudes()  # |ψ_i|²
        S = _node_entropy(amps)
        S_norm = S / logd

        score = degree * ent * S_norm

        if score > best_score:
            best_score = score
            best_id = node_id

    if best_id is None:
        raise RuntimeError("No nodes found in substrate; cannot select proton candidate.")

    return int(best_id)


# =============================================================================
# 2. POINTER "MEASUREMENT"
# =============================================================================

def make_pointer_state(node_state: np.ndarray) -> np.ndarray:
    """
    Construct a pointer state by projecting onto the basis vector
    with the largest |ψ_i|², preserving its phase.

    Result is a normalized vector with a single nonzero component.
    """
    psi = np.asarray(node_state, dtype=complex)
    d = psi.shape[0]

    # Find component with largest probability
    probs = np.abs(psi) ** 2
    k = int(np.argmax(probs))

    pointer = np.zeros(d, dtype=complex)
    # Keep the phase of the dominant component
    pointer[k] = np.exp(1j * np.angle(psi[k]))
    # Normalize (just in case)
    pointer = pointer / np.linalg.norm(pointer)

    return pointer


# =============================================================================
# 3. TIME EVOLUTION & "DECOHERENCE MICROSCOPE"
# =============================================================================

def record_pointer_timeseries(
    substrate: Substrate,
    node_id: int,
    n_steps: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """
    After the pointer projection, evolve the whole substrate and
    record the internal state of the chosen node over time.

    Returns:
        times   : shape (T,)
        states  : shape (T, d) complex array of node internal states
    """
    dt = substrate.config.dt
    times = []
    states = []

    for step in range(n_steps):
        node = substrate.nodes[node_id]
        states.append(node.state.copy())
        times.append(step * dt)

        substrate.evolve(n_steps=1)

    return np.asarray(times, dtype=float), np.asarray(states, dtype=complex)


def decoherence_metric(states: np.ndarray) -> np.ndarray:
    """
    Crude decoherence proxy for a single node:

        coherence_t = sum_{i!=j} |ρ_ij|

    where ρ = |ψ><ψ| is the pure-state density matrix of the node.
    This measures off-diagonal coherence magnitude in the node basis.

    States shape: (T, d)
    Returns: shape (T,)
    """
    T, d = states.shape
    coh = np.zeros(T, dtype=float)

    for t in range(T):
        psi = states[t]
        rho = np.outer(psi, np.conjugate(psi))  # d x d
        off_diag = rho - np.diag(np.diag(rho))
        coh[t] = float(np.sum(np.abs(off_diag)))

    return coh


# =============================================================================
# 4. WAVELET / TIME–FREQUENCY ANALYSIS
# =============================================================================

def compute_scalogram(
    times: np.ndarray,
    signal_1d: np.ndarray,
    n_scales: int = 32
) -> Dict[str, Any]:
    """
    Time–frequency analysis of a 1D complex signal using Morlet CWT if SciPy
    supports it. If not, returns an empty dict and the caller can skip it.

    Returns:
        {
            "widths": widths,        # scales
            "cwt": cwt_matrix        # shape (n_scales, T)
        }
    or {} if unavailable.
    """
    if not _HAS_SCIPY:
        return {}

    T = len(times)
    if T < 4:
        return {}

    # Use real magnitude as the analyzed signal
    x = np.abs(signal_1d)

    # Choose scales from ~1 time step to ~T/5
    widths = np.linspace(1.0, max(T / 5.0, 2.0), n_scales)

    # SciPy continuous wavelet transform with Morlet
    cwt_matrix = _signal.cwt(x, _signal.morlet2, widths, w=5.0)

    return {
        "widths": widths,
        "cwt": cwt_matrix
    }


# =============================================================================
# 5. MAIN PIPELINE
# =============================================================================

def run_proton_pointer_scope(
    n_nodes: int = 64,
    internal_dim: int = 3,
    monogamy_budget: float = 1.0,
    defrag_rate: float = 0.1,
    n_steps_sim: int = 200,
    n_steps_scope: int = 128,
    seed: int = 42,
    output_path: str = "proton_pointer_scope.npz"
) -> None:
    """
    Full pipeline:
        - run substrate simulation
        - pick proton-like node
        - project to pointer state
        - evolve and record time series
        - compute diagnostics
        - save everything to NPZ
    """
    print("=" * 72)
    print("PROTON POINTER SCOPE")
    print("=" * 72)

    # ---------------------------------------------------------------------
    # (1) Run the substrate simulation
    # ---------------------------------------------------------------------
    print("\n[1] Running substrate simulation...")
    config = Config(
        n_nodes=n_nodes,
        internal_dim=internal_dim,
        monogamy_budget=monogamy_budget,
        defrag_rate=defrag_rate,
        seed=seed
    )

    substrate, history = run_simulation(
        config=config,
        n_steps=n_steps_sim,
        record_every=max(1, n_steps_sim // 4)
    )

    print(f"    Nodes: {n_nodes}")
    print(f"    Internal dim: {internal_dim}")
    print(f"    Monogamy budget: {monogamy_budget}")
    print(f"    Defrag rate: {defrag_rate}")
    print(f"    Steps: {n_steps_sim}")

    # ---------------------------------------------------------------------
    # (2) Identify a proton-like candidate node
    # ---------------------------------------------------------------------
    print("\n[2] Locating proton-like candidate...")
    proton_id = find_proton_candidate(substrate)
    proton_node = substrate.nodes[proton_id]

    degree = proton_node.n_connections
    ent = proton_node.total_entanglement
    amps = proton_node.direction_amplitudes()
    probs = amps / max(np.sum(amps), 1e-12)
    S = _node_entropy(amps)
    S_norm = S / np.log(internal_dim) if internal_dim > 1 else 0.0

    print(f"    Selected node id: {proton_id}")
    print(f"    Degree: {degree}")
    print(f"    Total entanglement: {ent:.4f}")
    print(f"    Internal probs: {probs}")
    print(f"    Internal entropy: {S:.4f} (normalized: {S_norm:.4f})")

    # ---------------------------------------------------------------------
    # (3) "Probe" → construct pointer state
    # ---------------------------------------------------------------------
    print("\n[3] Probing node to create pointer state...")
    original_state = proton_node.state.copy()
    pointer_state = make_pointer_state(original_state)

    # Apply the "measurement": collapse node to pointer state
    proton_node.state = pointer_state.copy()

    # Amplitude maps before / after
    amps_original = np.abs(original_state) ** 2
    amps_pointer = np.abs(pointer_state) ** 2

    print("    Original amplitudes:", np.round(amps_original, 4))
    print("    Pointer amplitudes: ", np.round(amps_pointer, 4))

    # ---------------------------------------------------------------------
    # (4) Evolve & record time series after probe
    # ---------------------------------------------------------------------
    print("\n[4] Evolving after probe and recording time series...")
    times, states = record_pointer_timeseries(
        substrate=substrate,
        node_id=proton_id,
        n_steps=n_steps_scope
    )

    # Decoherence metric (off-diagonal coherence in node basis)
    print("\n[5] Computing decoherence metric...")
    coherence = decoherence_metric(states)
    print(f"    Initial coherence: {coherence[0]:.6f}")
    print(f"    Final coherence:   {coherence[-1]:.6f}")

    # ---------------------------------------------------------------------
    # (5) Optional wavelet scalogram on one component
    # ---------------------------------------------------------------------
    print("\n[6] Wavelet / time–frequency analysis (if available)...")
    scalograms: Dict[str, Any] = {}
    if states.shape[1] > 0:
        # Use the magnitude of the dominant pointer component as the signal
        k_dom = int(np.argmax(amps_pointer))
        signal_1d = states[:, k_dom]
        scalograms = compute_scalogram(times, signal_1d)
        if scalograms:
            print("    Wavelet scalogram computed "
                  f"(scales = {len(scalograms['widths'])}).")
        else:
            print("    Wavelet scalogram not available (SciPy or cwt missing).")
    else:
        print("    No internal components to analyze.")

    # ---------------------------------------------------------------------
    # (6) Save all data
    # ---------------------------------------------------------------------
    print(f"\n[7] Saving results to {output_path}...")

    meta: Dict[str, Any] = {
        "config": {
            "n_nodes": n_nodes,
            "internal_dim": internal_dim,
            "monogamy_budget": monogamy_budget,
            "defrag_rate": defrag_rate,
            "n_steps_sim": n_steps_sim,
            "n_steps_scope": n_steps_scope,
            "seed": seed,
        },
        "proton_node_id": int(proton_id),
        "degree": int(degree),
        "total_entanglement": float(ent),
        "internal_probs": probs.tolist(),
        "internal_entropy": float(S),
        "internal_entropy_normalized": float(S_norm),
        "has_scalogram": bool(bool(scalograms)),
    }

    # We store arrays in NPZ and meta as JSON-encoded string
    np.savez(
        output_path,
        times=times,
        states=states,
        original_state=original_state,
        pointer_state=pointer_state,
        amps_original=amps_original,
        amps_pointer=amps_pointer,
        coherence=coherence,
        scalogram_widths=scalograms.get("widths", np.array([])),
        scalogram_cwt=scalograms.get("cwt", np.zeros((0, 0))),
        meta_json=json.dumps(meta, indent=2)
    )

    print("    Done.")
    print("\nSummary:")
    print(f"    Proton candidate: node {proton_id}")
    print(f"    Degree:           {degree}")
    print(f"    Entanglement:     {ent:.4f}")
    print(f"    Pointer basis k*: {int(np.argmax(amps_pointer))}")
    print(f"    Results file:     {os.path.abspath(output_path)}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_proton_pointer_scope()
