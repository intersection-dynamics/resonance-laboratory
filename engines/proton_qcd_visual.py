#!/usr/bin/env python3
"""
proton_qcd_visual.py

"QCD mode" proton pointer visualization for the monogamy/defrag substrate.

What it does:
  1. Run the substrate with a more QCD-like configuration:
       - internal_dim = 8  (richer internal "color space")
       - monogamy_budget = 0.5
       - defrag_rate = 0.25
       - dt = 0.02
  2. Let structure emerge for n_steps_pre steps.
  3. Find a "proton-like" node (high degree, high entanglement, high internal entropy).
  4. Collapse that node to its dominant internal basis axis (pointer probe).
  5. Evolve for n_steps_post steps.
  6. At each post-probe step, for every node:
       - compute local coherence (off-diagonal magnitude of |psi><psi|)
       - compute a "color charge" RGB from its internal probabilities
       - place nodes on concentric rings by graph distance from the proton
       - draw a scatter plot with:
           * color = internal color charge (RGB)
           * size  = coherence (bubble brightness)
  7. Save frames as qcd_frame_0000.png, qcd_frame_0001.png, ...

You can then assemble these into a GIF/MP4 with your favorite tool.
"""

import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from substrate import Config, Substrate, run_simulation  # type: ignore


# =============================================================================
# Utilities reused from earlier scripts
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
    amps = np.array(node.direction_amplitudes(), dtype=float)
    d = len(amps)
    logd = np.log(d) if d > 1 else 1.0
    degree = node.n_connections
    ent = node.total_entanglement
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


def make_dominant_pointer_state(psi: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Collapse to the basis with largest probability, preserving phase.
    Returns (pointer_state, dominant_axis).
    """
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


def local_coherence(psi: np.ndarray) -> float:
    """
    Off-diagonal coherence magnitude for a single node's pure state |psi>.
    C = sum_{i != j} |rho_ij| where rho = |psi><psi|.
    """
    psi = np.asarray(psi, dtype=np.complex128)
    rho = np.outer(psi, np.conjugate(psi))
    off = rho - np.diag(np.diag(rho))
    return float(np.sum(np.abs(off)))


def build_distance_map(substrate: Substrate, source_id: int) -> Tuple[np.ndarray, List[int]]:
    """
    Compute graph distance from source_id to every node using BFS if neighbor
    info is available. Returns:
        distances: shape (N,)
        node_ids:  list of node ids ordered consistently with distances
    If neighbors are not available, all distances are set to 0.
    """
    node_ids = sorted(substrate.nodes.keys())
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

    # Check if we have any neighbor information at all
    any_neighbors = False
    for node in substrate.nodes.values():
        if get_neighbors(node):
            any_neighbors = True
            break

    if (not any_neighbors) or (source_id not in id_to_idx):
        # Flat distance if we can't do BFS
        return np.zeros(n, dtype=float), node_ids

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
        dist[:] = 0.0
    else:
        maxd = float(finite.max())
        dist[np.isinf(dist)] = maxd + 1.0

    return dist, node_ids


def color_from_internal(psi: np.ndarray) -> Tuple[float, float, float]:
    """
    Map internal probabilities to an RGB color.

    For internal_dim >= 3:
        - take |psi|^2, normalize, and use first three components as RGB
        - if dim < 3, pad with zeros.
    """
    psi = np.asarray(psi, dtype=np.complex128)
    probs = np.abs(psi) ** 2
    if probs.sum() > 0:
        probs = probs / probs.sum()
    d = probs.shape[0]
    r = probs[0] if d > 0 else 0.0
    g = probs[1] if d > 1 else 0.0
    b = probs[2] if d > 2 else 0.0
    # Slight gamma boost for visibility
    gamma = 0.7
    return (float(r**gamma), float(g**gamma), float(b**gamma))


# =============================================================================
# Visualization core
# =============================================================================

def layout_nodes_by_distance(distances: np.ndarray,
                             node_ids: List[int]) -> Dict[int, Tuple[float, float]]:
    """
    Compute a 2D layout where nodes are placed on concentric rings according to
    their distance from the proton node.

    Returns: dict node_id -> (x, y)
    """
    shells: Dict[int, List[int]] = {}
    for nid, d in zip(node_ids, distances):
        shell = int(round(d))
        shells.setdefault(shell, []).append(nid)

    positions: Dict[int, Tuple[float, float]] = {}
    max_shell = max(shells.keys()) if shells else 0
    r_step = 1.0 / max(1, max_shell + 1)  # keep everything in ~unit disk

    for shell, ids in shells.items():
        r = (shell + 1) * r_step
        M = len(ids)
        for k, nid in enumerate(ids):
            angle = 2.0 * np.pi * (k / max(M, 1))
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            positions[nid] = (float(x), float(y))

    return positions


def render_frame(substrate: Substrate,
                 proton_id: int,
                 positions: Dict[int, Tuple[float, float]],
                 t: float,
                 frame_idx: int,
                 out_dir: str) -> None:
    """
    Render a single frame:
      - node size = local coherence
      - node color = internal color charge
      - proton node outlined with a black circle
    """
    node_ids = sorted(substrate.nodes.keys())
    xs = []
    ys = []
    sizes = []
    colors = []

    max_coh = 0.0
    coh_list: List[float] = []

    # First pass: compute coherence, find max
    for nid in node_ids:
        node = substrate.nodes[nid]
        c = local_coherence(node.state)
        coh_list.append(c)
        if c > max_coh:
            max_coh = c

    max_coh = max(max_coh, 1e-6)

    for nid, c in zip(node_ids, coh_list):
        x, y = positions.get(nid, (0.0, 0.0))
        xs.append(x)
        ys.append(y)
        sizes.append(50.0 * (c / max_coh + 0.05))  # base size + coherence scaling
        colors.append(color_from_internal(substrate.nodes[nid].state))

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, s=sizes, c=colors, edgecolors="none")

    # Highlight proton node
    if proton_id in positions:
        px, py = positions[proton_id]
        plt.scatter([px], [py], s=200, facecolors="none", edgecolors="black", linewidths=2)

    plt.title(f"QCD proton pointer: coherence bubble (t = {t:.2f})")
    plt.axis("off")
    plt.axis("equal")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"qcd_frame_{frame_idx:04d}.png")
    plt.savefig(fname, dpi=150)
    plt.close()


# =============================================================================
# Main QCD-mode pipeline
# =============================================================================

def run_qcd_visual(
    n_nodes: int = 64,
    internal_dim: int = 8,
    monogamy_budget: float = 0.5,
    defrag_rate: float = 0.25,
    dt: float = 0.02,
    n_steps_pre: int = 200,
    n_steps_post: int = 200,
    seed: int = 123,
    out_dir: str = "qcd_frames",
) -> None:
    """
    Run the QCD-mode proton pointer visualization.
    """

    print("================================================================")
    print("QCD-MODE PROTON POINTER VISUAL")
    print("================================================================")
    print(f"Config: n_nodes={n_nodes}, d={internal_dim}, "
          f"monogamy={monogamy_budget}, defrag={defrag_rate}, dt={dt}")
    print(f"Pre-evolution steps:  {n_steps_pre}")
    print(f"Post-probe steps:     {n_steps_post}")
    print(f"Output directory:     {os.path.abspath(out_dir)}")
    print("================================================================")

    # Build config
    config = Config(
        n_nodes=n_nodes,
        internal_dim=internal_dim,
        monogamy_budget=monogamy_budget,
        defrag_rate=defrag_rate,
        seed=seed,
    )
    # Override dt if Config allows it via attribute
    if hasattr(config, "dt"):
        config.dt = dt

    # Pre-evolve substrate
    substrate, _ = run_simulation(
        config=config,
        n_steps=n_steps_pre,
        record_every=max(1, n_steps_pre // 5),
    )

    # Find proton candidate
    proton_id = find_proton_candidate(substrate)
    proton_node = substrate.nodes[proton_id]
    original_state = proton_node.state.copy()
    pointer_state, dom_axis = make_dominant_pointer_state(original_state)
    proton_node.state = pointer_state.copy()

    print(f"\nSelected proton-like node: {proton_id}")
    print(f"Dominant internal axis k*:  {dom_axis}")
    print("Original internal probabilities:",
          np.round(np.abs(original_state) ** 2, 4))
    print("Pointer internal probabilities:",
          np.round(np.abs(pointer_state) ** 2, 4))

    # Build distance-based layout
    distances, node_ids = build_distance_map(substrate, proton_id)
    positions = layout_nodes_by_distance(distances, node_ids)

    # Save meta info
    meta = {
        "config": {
            "n_nodes": n_nodes,
            "internal_dim": internal_dim,
            "monogamy_budget": monogamy_budget,
            "defrag_rate": defrag_rate,
            "dt": dt,
            "n_steps_pre": n_steps_pre,
            "n_steps_post": n_steps_post,
            "seed": seed,
        },
        "proton_node_id": int(proton_id),
        "dominant_axis": int(dom_axis),
        "distances": distances.tolist(),
        "node_ids": node_ids,
    }
    meta_path = os.path.join(out_dir, "qcd_meta.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMeta written to {meta_path}")

    # Post-probe evolution & frame rendering
    t = 0.0
    for step in range(n_steps_post):
        print(f"Rendering frame {step+1}/{n_steps_post} ...", end="\r")
        render_frame(substrate, proton_id, positions, t, step, out_dir)
        substrate.evolve(n_steps=1)
        t += dt

    print("\nDone.")
    print(f"Frames written to: {os.path.abspath(out_dir)}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    run_qcd_visual()
