#!/usr/bin/env python3
"""
proton_pointer_suite.py

Advanced probes for the monogamy/defrag substrate:

  (A) Axis-sweep pointer probes:
      - Choose a proton-like node.
      - Collapse its internal state onto basis axes 0, 1, 2 separately.
      - Evolve and record amplitudes + coherence vs time.
      - Files: proton_axis_0.npz, proton_axis_1.npz, proton_axis_2.npz

  (B) Multi-probe (1x, 2x, 3x) sequences:
      - Proton-like node collapsed to its dominant axis.
      - Let it heal, then probe again (2x / 3x) and see if healing saturates.
      - Files: proton_multi_probe_1x.npz, proton_multi_probe_2x.npz, proton_multi_probe_3x.npz

  (C) Entanglement wavefront:
      - Single probe on proton node.
      - Record total_entanglement for every node over time.
      - Attempt to group by graph distance from proton node (if neighbor info exists).
      - File: proton_entanglement_wavefront.npz

  (D) Vacuum control:
      - Choose a "vacuum-like" node with low degree, low entanglement, low internal entropy.
      - Run the same single-probe experiment.
      - Expect: coherence does NOT regrow strongly.
      - File: vacuum_pointer_scope.npz
"""

import json
import os
from typing import Dict, Any, Tuple, List

import numpy as np

from substrate import Config, Substrate, run_simulation  # type: ignore


# =============================================================================
# Shared utilities
# =============================================================================

def _node_entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability vector p (natural log)."""
    p = np.asarray(p, dtype=float)
    total = max(p.sum(), 1e-12)
    p = p / total
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def score_node(node) -> float:
    """
    Heuristic "proton-ness" score:
        degree * total_entanglement * (S_internal / log d)
    """
    d = len(node.direction_amplitudes())
    logd = np.log(d) if d > 1 else 1.0
    degree = node.n_connections
    ent = node.total_entanglement
    amps = np.array(node.direction_amplitudes(), dtype=float)
    S = _node_entropy(amps)
    S_norm = S / logd
    return degree * ent * S_norm


def find_proton_candidate(substrate: Substrate) -> int:
    """Select node with largest score_node."""
    best_id = None
    best_score = -1.0

    for node_id, node in substrate.nodes.items():
        s = score_node(node)
        if s > best_score:
            best_score = s
            best_id = node_id

    if best_id is None:
        raise RuntimeError("No nodes found while searching for proton candidate.")

    return int(best_id)


def find_vacuum_candidate(substrate: Substrate) -> int:
    """Select node with *smallest* score_node."""
    worst_id = None
    worst_score = None

    for node_id, node in substrate.nodes.items():
        s = score_node(node)
        if worst_score is None or s < worst_score:
            worst_score = s
            worst_id = node_id

    if worst_id is None:
        raise RuntimeError("No nodes found while searching for vacuum candidate.")

    return int(worst_id)


def make_forced_pointer_state(d: int, axis: int) -> np.ndarray:
    """Return pure basis vector |axis> in C^d."""
    pointer = np.zeros(d, dtype=np.complex128)
    pointer[axis] = 1.0 + 0.0j
    return pointer


def make_dominant_pointer_state(psi: np.ndarray) -> Tuple[np.ndarray, int]:
    """Collapse to the basis with largest probability, preserving phase."""
    psi = np.asarray(psi, dtype=np.complex128)
    d = psi.shape[0]
    probs = np.abs(psi) ** 2
    if probs.sum() <= 0:
        k = 0
    else:
        k = int(np.argmax(probs))
    pointer = np.zeros(d, dtype=np.complex128)
    pointer[k] = np.exp(1j * np.angle(psi[k]))
    norm = np.linalg.norm(pointer)
    if norm > 0:
        pointer /= norm
    return pointer, k


def decoherence_metric(states: np.ndarray) -> np.ndarray:
    """
    Off-diagonal coherence magnitude for a single node over time.
    states: (T, d) pure state vectors.
    """
    T, d = states.shape
    coh = np.zeros(T, dtype=float)
    for t in range(T):
        psi = states[t]
        rho = np.outer(psi, np.conjugate(psi))
        off = rho - np.diag(np.diag(rho))
        coh[t] = float(np.sum(np.abs(off)))
    return coh


def record_node_timeseries(substrate: Substrate, node_id: int, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evolve substrate for n_steps, recording node_id.internal_state at each step.
    Returns:
        times:  (T,)
        states: (T, d)
    """
    dt = substrate.config.dt
    times: List[float] = []
    states: List[np.ndarray] = []
    for step in range(n_steps):
        node = substrate.nodes[node_id]
        states.append(node.state.copy())
        times.append(step * dt)
        substrate.evolve(n_steps=1)
    return np.asarray(times, dtype=float), np.asarray(states, dtype=np.complex128)


def build_distance_map(substrate: Substrate, source_id: int) -> np.ndarray:
    """
    Attempt to compute graph distance from source_id to every node using BFS.
    We try to infer neighbors from node.neighbors (list of ids) if present.
    If that fails, we return all zeros and let the viewer decide how to use it.
    """
    node_ids = list(substrate.nodes.keys())
    node_ids.sort()
    node_ids = list(node_ids)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    dist = np.full(n, np.inf, dtype=float)

    def get_neighbors(node_obj):
        if hasattr(node_obj, "neighbors"):
            try:
                return list(node_obj.neighbors)
            except Exception:
                pass
        if hasattr(node_obj, "neighbor_ids"):
            try:
                return list(node_obj.neighbor_ids)
            except Exception:
                pass
        return []

    any_neighbors = any(get_neighbors(node) for node in substrate.nodes.values())
    if not any_neighbors or source_id not in id_to_idx:
        return np.zeros(n, dtype=float)

    from collections import deque
    q = deque()
    source_idx = id_to_idx[source_id]
    dist[source_idx] = 0.0
    q.append(source_id)

    while q:
        nid = q.popleft()
        d_here = dist[id_to_idx[nid]]
        node = substrate.nodes[nid]
        for nbr_id in get_neighbors(node):
            if nbr_id in id_to_idx:
                j = id_to_idx[nbr_id]
                if np.isinf(dist[j]):
                    dist[j] = d_here + 1.0
                    q.append(nbr_id)

    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return np.zeros(n, dtype=float)
    maxd = float(finite.max())
    dist[np.isinf(dist)] = maxd + 1.0
    return dist


# =============================================================================
# Experiments
# =============================================================================

def run_axis_sweep(config: Config,
                   n_steps_sim: int,
                   n_steps_scope: int) -> None:
    """
    Axis-sweep probe on basis indices 0, 1, 2 for the same proton candidate.
    Each axis run re-simulates from scratch with same config + seed, so the
    pre-probe substrate is identical.
    """
    print("\n=== AXIS SWEEP (0, 1, 2) ===")

    for axis in [0, 1, 2]:
        print(f"\n-- Axis {axis} --")
        substrate, _ = run_simulation(config=config, n_steps=n_steps_sim, record_every=max(1, n_steps_sim // 4))
        proton_id = find_proton_candidate(substrate)
        node = substrate.nodes[proton_id]

        original_state = node.state.copy()
        d = original_state.shape[0]
        pointer_state = make_forced_pointer_state(d, axis)
        node.state = pointer_state.copy()

        times, states = record_node_timeseries(substrate, proton_id, n_steps_scope)
        coherence = decoherence_metric(states)

        amps_original = np.abs(original_state) ** 2
        amps_pointer = np.abs(pointer_state) ** 2

        meta = {
            "experiment": "axis_sweep",
            "axis": axis,
            "proton_node_id": int(proton_id),
            "config": {
                "n_nodes": config.n_nodes,
                "internal_dim": config.internal_dim,
                "monogamy_budget": config.monogamy_budget,
                "defrag_rate": config.defrag_rate,
                "dt": config.dt,
                "n_steps_sim": n_steps_sim,
                "n_steps_scope": n_steps_scope,
                "seed": config.seed,
            },
        }

        fname = f"proton_axis_{axis}.npz"
        np.savez(
            fname,
            times=times,
            states=states,
            coherence=coherence,
            original_state=original_state,
            pointer_state=pointer_state,
            amps_original=amps_original,
            amps_pointer=amps_pointer,
            meta_json=json.dumps(meta, indent=2),
        )
        print(f"Saved axis-sweep results to {os.path.abspath(fname)}")
        print(f"  Initial coherence: {coherence[0]:.6f}, final: {coherence[-1]:.6f}")


def run_multi_probe(config: Config,
                    n_steps_sim: int,
                    n_between: int,
                    max_probes: int = 3) -> None:
    """
    Multi-probe sequences: 1x, 2x, 3x.
    - Collapse proton node to its *dominant* basis axis.
    - Let it heal for n_between steps.
    - Probe again (2x, 3x), continuing the sequence.

    Records:
        - times, states, coherence
        - probe_indices: indices in the time array where probes occur
    """
    print("\n=== MULTI-PROBE SEQUENCES (1x, 2x, 3x) ===")

    for n_probes in [1, 2, 3]:
        print(f"\n-- {n_probes}x probe sequence --")
        substrate, _ = run_simulation(config=config, n_steps=n_steps_sim, record_every=max(1, n_steps_sim // 4))
        proton_id = find_proton_candidate(substrate)
        node = substrate.nodes[proton_id]

        original_state = node.state.copy()
        pointer_state, dom_axis = make_dominant_pointer_state(original_state)
        node.state = pointer_state.copy()

        dt = substrate.config.dt
        states: List[np.ndarray] = []
        times: List[float] = []
        probe_indices: List[int] = []

        t = 0.0
        step_index = 0

        def record_point():
            nonlocal step_index, t
            node_local = substrate.nodes[proton_id]
            states.append(node_local.state.copy())
            times.append(t)
            step_index += 1

        # Record immediately after first probe
        record_point()
        probe_indices.append(0)

        for probe_idx in range(1, n_probes + 1):
            # Evolve between probes (or after last probe as tail)
            steps_here = n_between if probe_idx < n_probes else n_between
            for _ in range(steps_here):
                substrate.evolve(n_steps=1)
                t += dt
                record_point()

            if probe_idx < n_probes:
                # Apply another probe to the same dominant axis
                node = substrate.nodes[proton_id]
                node.state = pointer_state.copy()
                probe_indices.append(step_index - 1)

        states_arr = np.asarray(states, dtype=np.complex128)
        times_arr = np.asarray(times, dtype=float)
        coherence = decoherence_metric(states_arr)

        amps_original = np.abs(original_state) ** 2
        amps_pointer = np.abs(pointer_state) ** 2

        meta = {
            "experiment": "multi_probe",
            "n_probes": n_probes,
            "dominant_axis": int(dom_axis),
            "proton_node_id": int(proton_id),
            "n_between": n_between,
            "probe_indices": probe_indices,
            "config": {
                "n_nodes": config.n_nodes,
                "internal_dim": config.internal_dim,
                "monogamy_budget": config.monogamy_budget,
                "defrag_rate": config.defrag_rate,
                "dt": config.dt,
                "n_steps_sim": n_steps_sim,
                "n_between": n_between,
                "seed": config.seed,
            },
        }

        fname = f"proton_multi_probe_{n_probes}x.npz"
        np.savez(
            fname,
            times=times_arr,
            states=states_arr,
            coherence=coherence,
            original_state=original_state,
            pointer_state=pointer_state,
            amps_original=amps_original,
            amps_pointer=amps_pointer,
            probe_indices=np.array(probe_indices, dtype=int),
            meta_json=json.dumps(meta, indent=2),
        )
        print(f"Saved multi-probe results to {os.path.abspath(fname)}")
        print(f"  Initial coherence: {coherence[0]:.6f}, final: {coherence[-1]:.6f}")


def run_entanglement_wavefront(config: Config,
                               n_steps_sim: int,
                               n_steps_wave: int) -> None:
    """
    Single probe on proton node, then:
      - record total_entanglement for every node at each time step
      - attempt to compute graph distance from proton node to all nodes

    File: proton_entanglement_wavefront.npz
    """
    print("\n=== ENTANGLEMENT WAVEFRONT ===")
    substrate, _ = run_simulation(config=config, n_steps=n_steps_sim, record_every=max(1, n_steps_sim // 4))
    proton_id = find_proton_candidate(substrate)
    proton_node = substrate.nodes[proton_id]

    original_state = proton_node.state.copy()
    pointer_state, dom_axis = make_dominant_pointer_state(original_state)
    proton_node.state = pointer_state.copy()

    node_ids = list(substrate.nodes.keys())
    node_ids.sort()
    node_ids = list(node_ids)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    ent_history = np.zeros((n_steps_wave, n), dtype=float)
    times = np.zeros(n_steps_wave, dtype=float)
    dt = substrate.config.dt

    for step in range(n_steps_wave):
        t = step * dt
        times[step] = t
        for nid in node_ids:
            node = substrate.nodes[nid]
            ent_history[step, id_to_idx[nid]] = node.total_entanglement
        substrate.evolve(n_steps=1)

    distances = build_distance_map(substrate, proton_id)

    meta = {
        "experiment": "entanglement_wavefront",
        "proton_node_id": int(proton_id),
        "dominant_axis": int(dom_axis),
        "node_ids": node_ids,
        "distance_mode": "bfs_if_available_else_flat",
        "config": {
            "n_nodes": config.n_nodes,
            "internal_dim": config.internal_dim,
            "monogamy_budget": config.monogamy_budget,
            "defrag_rate": config.defrag_rate,
            "dt": config.dt,
            "n_steps_sim": n_steps_sim,
            "n_steps_wave": n_steps_wave,
            "seed": config.seed,
        },
    }

    fname = "proton_entanglement_wavefront.npz"
    np.savez(
        fname,
        times=times,
        entanglement=ent_history,
        node_ids=np.array(node_ids, dtype=int),
        distances=distances,
        meta_json=json.dumps(meta, indent=2),
    )
    print(f"Saved entanglement wavefront to {os.path.abspath(fname)}")


def run_vacuum_control(config: Config,
                       n_steps_sim: int,
                       n_steps_scope: int) -> None:
    """
    Vacuum control: choose a low-score node and run the same single-probe
    experiment we used for the proton. Expect no strong coherence regrowth.
    """
    print("\n=== VACUUM CONTROL ===")
    substrate, _ = run_simulation(config=config, n_steps=n_steps_sim, record_every=max(1, n_steps_sim // 4))
    vacuum_id = find_vacuum_candidate(substrate)
    node = substrate.nodes[vacuum_id]

    original_state = node.state.copy()
    pointer_state, dom_axis = make_dominant_pointer_state(original_state)
    node.state = pointer_state.copy()

    times, states = record_node_timeseries(substrate, vacuum_id, n_steps_scope)
    coherence = decoherence_metric(states)

    amps_original = np.abs(original_state) ** 2
    amps_pointer = np.abs(pointer_state) ** 2

    meta = {
        "experiment": "vacuum_control",
        "vacuum_node_id": int(vacuum_id),
        "dominant_axis": int(dom_axis),
        "config": {
            "n_nodes": config.n_nodes,
            "internal_dim": config.internal_dim,
            "monogamy_budget": config.monogamy_budget,
            "defrag_rate": config.defrag_rate,
            "dt": config.dt,
            "n_steps_sim": n_steps_sim,
            "n_steps_scope": n_steps_scope,
            "seed": config.seed,
        },
    }

    fname = "vacuum_pointer_scope.npz"
    np.savez(
        fname,
        times=times,
        states=states,
        coherence=coherence,
        original_state=original_state,
        pointer_state=pointer_state,
        amps_original=amps_original,
        amps_pointer=amps_pointer,
        meta_json=json.dumps(meta, indent=2),
    )
    print(f"Saved vacuum control results to {os.path.abspath(fname)}")
    print(f"  Initial coherence: {coherence[0]:.6f}, final: {coherence[-1]:.6f}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    # You can tweak these defaults if you like
    config = Config(
        n_nodes=64,
        internal_dim=3,       # SU(3)-like "color" sector
        monogamy_budget=1.0,
        defrag_rate=0.1,
        seed=42,
    )

    n_steps_sim = 200    # pre-probe evolution to let structure emerge
    n_steps_scope = 128  # post-probe recording window
    n_steps_wave = 128   # entanglement wavefront duration
    n_between = 64       # steps between probes in multi-probe sequences

    print("================================================================")
    print("PROTON POINTER SUITE")
    print("================================================================")
    print(f"Config: n_nodes={config.n_nodes}, d={config.internal_dim}, "
          f"monogamy={config.monogamy_budget}, defrag={config.defrag_rate}, dt={config.dt}")
    print()

    run_axis_sweep(config, n_steps_sim=n_steps_sim, n_steps_scope=n_steps_scope)
    run_multi_probe(config, n_steps_sim=n_steps_sim, n_between=n_between, max_probes=3)
    run_entanglement_wavefront(config, n_steps_sim=n_steps_sim, n_steps_wave=n_steps_wave)
    run_vacuum_control(config, n_steps_sim=n_steps_sim, n_steps_scope=n_steps_scope)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
