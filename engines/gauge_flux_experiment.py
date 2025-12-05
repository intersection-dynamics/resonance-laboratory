"""
gauge_flux_experiment.py

Emergent-geometry gauge-flux experiment for the Hilbert substrate.

Idea:
- Geometry: given by lr_embedding_3d.npz (must contain graph_dist).
- Degrees of freedom: one qubit per site (dim = 2^n_sites).
- Hamiltonian: Heisenberg-like on edges, but with U(1) link phases
  on the XY (hopping) part.

  For edge (i,j) with phase phi_ij, the XY piece is:

      cos(phi_ij) * (X_i X_j + Y_i Y_j)
    + sin(phi_ij) * (X_i Y_j - Y_i X_j)

  This is equivalent to a Peierls-substituted spin-flip term
  e^{+i phi} S_i^+ S_j^- + e^{-i phi} S_i^- S_j^+.

- Experiment:
  1. Initialize random phases phi_ij on edges.
  2. For each gauge configuration:
       * Build H(phi).
       * Diagonalize -> ground state |psi>.
       * Compute Z-correlation matrix
           C_ij = <Z_i Z_j> - <Z_i><Z_j>.
       * Compute a cost functional:

         C_ent(phi) = sum_{dist>1} w_long(dist) |C_ij|^2
                    - alpha_local_reward * sum_{dist=1} |C_ij|^2

         i.e. punish long-range correlations, reward local ones.

       * Compute square plaquette fluxes Φ_p on all 4-cycles and add

         C_flux(phi) = sum_p min[(Φ_p - π)^2, (Φ_p + π)^2]

         to encourage ±π flux per square plaquette.

     Total cost:

         C_total = C_ent + flux_weight * C_flux

  3. Do gradient descent on phi_ij to minimize C_total via
     finite-difference gradients.

- Output:
  * Optimized phases per edge.
  * Cost evolution.
  * Square plaquette fluxes to see if they cluster near ±π.

Single-file, no imports from the rest of the project.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Data classes and I/O helpers
# ---------------------------------------------------------------------

@dataclass
class GeometryAsset:
    graph_dist: np.ndarray
    extras: Dict[str, np.ndarray]


@dataclass
class GaugeConfig:
    n_sites: int
    J_coupling: float = 1.0
    h_field: float = 0.2
    defrag_zz: float = 0.0  # optional extra ZZ term
    alpha_local_reward: float = 0.5
    w_long_base: float = 1.0
    flux_weight: float = 0.5  # weight for square-plaquette flux term
    n_gauge_steps: int = 30
    phase_step: float = 0.1
    fd_epsilon: float = 1e-2
    random_seed: int = 1234


@dataclass
class RunLayout:
    run_root: str
    data_dir: str
    logs_dir: str
    figures_dir: str
    params_json: str
    metadata_json: str
    summary_json: str
    phases_json: str
    cost_history_json: str
    flux_json: str
    log_path: str


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=False)


def make_run_id(tag: str | None = None) -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if tag:
        tag = tag.strip()
        if tag:
            return f"{ts}_{tag}"
    return ts


def build_layout(output_root: str, tag: str | None = None) -> RunLayout:
    run_id = make_run_id(tag)
    run_root = os.path.join(output_root, "gauge_flux_experiment", run_id)

    data_dir = os.path.join(run_root, "data")
    logs_dir = os.path.join(run_root, "logs")
    figures_dir = os.path.join(run_root, "figures")

    ensure_dir(run_root)
    ensure_dir(data_dir)
    ensure_dir(logs_dir)
    ensure_dir(figures_dir)

    return RunLayout(
        run_root=run_root,
        data_dir=data_dir,
        logs_dir=logs_dir,
        figures_dir=figures_dir,
        params_json=os.path.join(run_root, "params.json"),
        metadata_json=os.path.join(run_root, "metadata.json"),
        summary_json=os.path.join(run_root, "summary.json"),
        phases_json=os.path.join(data_dir, "phases.json"),
        cost_history_json=os.path.join(data_dir, "cost_history.json"),
        flux_json=os.path.join(data_dir, "fluxes.json"),
        log_path=os.path.join(logs_dir, "run.log"),
    )


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_geometry(path: str) -> GeometryAsset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Geometry file not found: {path}")
    npz = np.load(path)
    if "graph_dist" not in npz.files:
        raise ValueError("Geometry npz must contain 'graph_dist'.")
    gd = np.array(npz["graph_dist"], dtype=float)
    extras = {k: npz[k] for k in npz.files if k != "graph_dist"}
    return GeometryAsset(graph_dist=gd, extras=extras)


def adjacency_from_graph_dist(graph_dist: np.ndarray) -> np.ndarray:
    adj = (graph_dist == 1).astype(int)
    np.fill_diagonal(adj, 0)
    return adj


# ---------------------------------------------------------------------
# Pauli ops, Hamiltonian with gauge, and ground state
# ---------------------------------------------------------------------

def pauli_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z


def kron_on_site(op: np.ndarray, site: int, n_sites: int) -> np.ndarray:
    I2 = np.eye(2, dtype=complex)
    mats = []
    for j in range(n_sites):
        mats.append(op if j == site else I2)
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def build_local_ops(n_sites: int) -> Dict[str, List[np.ndarray]]:
    I, X, Y, Z = pauli_matrices()
    Xs: List[np.ndarray] = []
    Ys: List[np.ndarray] = []
    Zs: List[np.ndarray] = []
    for s in range(n_sites):
        Xs.append(kron_on_site(X, s, n_sites))
        Ys.append(kron_on_site(Y, s, n_sites))
        Zs.append(kron_on_site(Z, s, n_sites))
    return {"X": Xs, "Y": Ys, "Z": Zs}


def build_edge_list(adjacency: np.ndarray) -> List[Tuple[int, int]]:
    n = adjacency.shape[0]
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] != 0:
                edges.append((i, j))
    return edges


def build_hamiltonian_with_gauge(
    cfg: GaugeConfig,
    adjacency: np.ndarray,
    local_ops: Dict[str, List[np.ndarray]],
    edges: List[Tuple[int, int]],
    phases: np.ndarray,
) -> np.ndarray:
    """
    H(phi) = sum_edges J * [ cos(phi) (X_i X_j + Y_i Y_j)
                             + sin(phi) (X_i Y_j - Y_i X_j)
                             + defrag_zz * Z_i Z_j ]
             - h_field * sum_i Z_i

    The cos/sin combination is equivalent to a spin-flip term with a U(1)
    link phase, i.e. a discrete U(1) gauge field.
    """
    n_sites = cfg.n_sites
    Xs = local_ops["X"]
    Ys = local_ops["Y"]
    Zs = local_ops["Z"]

    dim = 2 ** n_sites
    H = np.zeros((dim, dim), dtype=complex)

    J = cfg.J_coupling
    h = cfg.h_field
    gzz = cfg.defrag_zz

    for (edge_idx, (i, j)) in enumerate(edges):
        phi = phases[edge_idx]
        c = np.cos(phi)
        s = np.sin(phi)

        XX = Xs[i] @ Xs[j]
        YY = Ys[i] @ Ys[j]
        XY = Xs[i] @ Ys[j]
        YX = Ys[i] @ Xs[j]
        ZZ = Zs[i] @ Zs[j]

        H += J * (c * (XX + YY) + s * (XY - YX))
        if gzz != 0.0:
            H += gzz * ZZ

    for i in range(n_sites):
        H += -h * Zs[i]

    H = 0.5 * (H + H.conj().T)
    return H


def ground_state(H: np.ndarray) -> Tuple[float, np.ndarray]:
    evals, evecs = np.linalg.eigh(H)
    idx = np.argmin(evals)
    E0 = float(evals[idx])
    psi0 = evecs[:, idx]
    return E0, psi0


# ---------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------

def z_expectations(psi: np.ndarray, Z_ops: List[np.ndarray]) -> np.ndarray:
    bra = psi.conj().T
    vals = []
    for Zi in Z_ops:
        vals.append(np.real_if_close(bra @ (Zi @ psi)))
    return np.array(vals, dtype=float)


def z_corr_matrix(
    psi: np.ndarray,
    Z_ops: List[np.ndarray],
) -> np.ndarray:
    """
    C_ij = <Z_i Z_j> - <Z_i> <Z_j>
    """
    n_sites = len(Z_ops)
    z_means = z_expectations(psi, Z_ops)
    C = np.zeros((n_sites, n_sites), dtype=float)
    bra = psi.conj().T
    for i in range(n_sites):
        for j in range(n_sites):
            Zij = Z_ops[i] @ Z_ops[j]
            val = np.real_if_close(bra @ (Zij @ psi))
            C[i, j] = float(val) - float(z_means[i] * z_means[j])
    return C


# ---------------------------------------------------------------------
# Square plaquettes and fluxes
# ---------------------------------------------------------------------

def find_square_plaquettes(adjacency: np.ndarray) -> List[List[int]]:
    """
    Find simple 4-cycles (square plaquettes) in the graph.

    Returns:
        A list of oriented 4-cycles [a,b,c,d] such that edges
        (a,b), (b,c), (c,d), (d,a) exist and a,b,c,d are distinct.
        Each square is returned once (deduplicated by sorted vertex set).
    """
    n = adjacency.shape[0]
    squares: List[List[int]] = []
    seen: set[Tuple[int, int, int, int]] = set()

    for s in range(n):
        for i in range(n):
            if i == s or adjacency[s, i] == 0:
                continue
            for j in range(n):
                if j in (s, i) or adjacency[i, j] == 0:
                    continue
                for k in range(n):
                    if k in (s, i, j) or adjacency[j, k] == 0:
                        continue
                    if adjacency[k, s] == 0:
                        continue
                    nodes = [s, i, j, k]
                    if len(set(nodes)) != 4:
                        continue
                    key = tuple(sorted(nodes))
                    if key in seen:
                        continue
                    seen.add(key)
                    squares.append(nodes)

    return squares


def square_fluxes(
    edges: List[Tuple[int, int]],
    phases: np.ndarray,
    squares: List[List[int]],
) -> List[Dict[str, Any]]:
    """
    Compute flux = sum of phi_ij around each square [a,b,c,d],
    using oriented edges (a->b, b->c, c->d, d->a).

    Returns:
        List of dicts with:
          - "plaquette": [a,b,c,d]
          - "flux_raw": sum of (signed) phases
          - "flux_wrapped": flux wrapped to (-pi, pi]
          - "legs": per-edge info (edge, phase_used)
    """
    edge_index: Dict[Tuple[int, int], int] = {}
    for idx, (i, j) in enumerate(edges):
        if i > j:
            i, j = j, i
        edge_index[(i, j)] = idx

    out: List[Dict[str, Any]] = []
    for sq in squares:
        a, b, c, d = sq
        pairs = [(a, b), (b, c), (c, d), (d, a)]
        flux = 0.0
        legs = []
        for (u, v) in pairs:
            if u < v:
                key = (u, v)
                sign = +1.0
            else:
                key = (v, u)
                sign = -1.0
            idx = edge_index.get(key, None)
            if idx is None:
                phase_uv = 0.0
            else:
                phase_uv = sign * float(phases[idx])
            flux += phase_uv
            legs.append({"edge": [u, v], "phase_used": phase_uv})
        flux_wrapped = (flux + np.pi) % (2.0 * np.pi) - np.pi
        out.append(
            {
                "plaquette": sq,
                "flux_raw": flux,
                "flux_wrapped": flux_wrapped,
                "legs": legs,
            }
        )
    return out


# ---------------------------------------------------------------------
# Cost functional: entanglement locality + flux preference
# ---------------------------------------------------------------------

def entanglement_cost(
    cfg: GaugeConfig,
    graph_dist: np.ndarray,
    C: np.ndarray,
) -> float:
    """
    C_ent = sum_{dist>1} w_long(dist) |C_ij|^2
          - alpha_local_reward * sum_{dist=1} |C_ij|^2

    - dist > 1: penalize long-range correlations.
    - dist == 1: reward local correlations (negative term).
    """
    n = graph_dist.shape[0]
    cost_long = 0.0
    reward_local = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            d = graph_dist[i, j]
            cij2 = float(np.abs(C[i, j]) ** 2)
            if d <= 0:
                continue
            if d == 1.0:
                reward_local += cij2
            elif d > 1.0:
                w = cfg.w_long_base * (d ** 2)
                cost_long += w * cij2

    return cost_long - cfg.alpha_local_reward * reward_local


def flux_cost(
    squares_flux: List[Dict[str, Any]],
) -> float:
    """
    C_flux = sum_p min[(Φ_p - π)^2, (Φ_p + π)^2]

    i.e. prefer square plaquette fluxes near ±π.
    """
    total = 0.0
    for fx in squares_flux:
        phi = float(fx["flux_wrapped"])
        # distance to +pi or -pi
        d_plus = (phi - np.pi) ** 2
        d_minus = (phi + np.pi) ** 2
        total += float(min(d_plus, d_minus))
    return total


def total_cost(
    cfg: GaugeConfig,
    geom: GeometryAsset,
    C: np.ndarray,
    edges: List[Tuple[int, int]],
    phases: np.ndarray,
    squares: List[List[int]],
) -> Tuple[float, float, float]:
    """
    Compute total cost and its components:

        C_total = C_ent + flux_weight * C_flux

    Returns:
        C_total, C_ent, C_flux
    """
    C_ent = entanglement_cost(cfg, geom.graph_dist, C)
    if squares:
        sq_flux_list = square_fluxes(edges, phases, squares)
        C_f = flux_cost(sq_flux_list)
    else:
        C_f = 0.0
    C_total = C_ent + cfg.flux_weight * C_f
    return float(C_total), float(C_ent), float(C_f)


# ---------------------------------------------------------------------
# Gauge optimization (outer loop) via finite-difference gradients
# ---------------------------------------------------------------------

def gauge_gradient_descent(
    geom: GeometryAsset,
    cfg: GaugeConfig,
    layout: RunLayout,
    log,
) -> Dict[str, Any]:
    n_sites = geom.graph_dist.shape[0]
    if cfg.n_sites != n_sites:
        raise ValueError(f"GaugeConfig n_sites={cfg.n_sites}, geometry has {n_sites}.")

    adjacency = adjacency_from_graph_dist(geom.graph_dist)
    local_ops = build_local_ops(n_sites)
    Z_ops = local_ops["Z"]
    edges = build_edge_list(adjacency)
    n_edges = len(edges)
    squares = find_square_plaquettes(adjacency)

    log(f"n_sites = {n_sites}, n_edges = {n_edges}")
    log(f"Edges: {edges}")
    log(f"Found {len(squares)} square plaquettes: {squares}")
    log("-" * 60)

    rng = np.random.default_rng(cfg.random_seed)
    phases = rng.uniform(low=-np.pi, high=np.pi, size=n_edges)

    cost_history: List[Dict[str, float]] = []

    for step in range(cfg.n_gauge_steps):
        # Build H and compute cost components
        H = build_hamiltonian_with_gauge(cfg, adjacency, local_ops, edges, phases)
        E0, psi0 = ground_state(H)
        C = z_corr_matrix(psi0, Z_ops)
        C_total, C_ent, C_f = total_cost(cfg, geom, C, edges, phases, squares)

        cost_history.append(
            {
                "step": float(step),
                "total": C_total,
                "entanglement": C_ent,
                "flux": C_f,
                "E0": E0,
            }
        )

        log(
            f"[step {step:03d}] E0 = {E0:.6f}, "
            f"C_total = {C_total:.6f}, "
            f"C_ent = {C_ent:.6f}, "
            f"C_flux = {C_f:.6f}"
        )

        # Finite-difference gradient wrt each phase
        grad = np.zeros_like(phases)
        eps = cfg.fd_epsilon

        for e_idx in range(n_edges):
            original = phases[e_idx]

            phases[e_idx] = original + eps
            H_plus = build_hamiltonian_with_gauge(cfg, adjacency, local_ops, edges, phases)
            _, psi_plus = ground_state(H_plus)
            C_plus = z_corr_matrix(psi_plus, Z_ops)
            C_tot_plus, _, _ = total_cost(cfg, geom, C_plus, edges, phases, squares)

            phases[e_idx] = original - eps
            H_minus = build_hamiltonian_with_gauge(cfg, adjacency, local_ops, edges, phases)
            _, psi_minus = ground_state(H_minus)
            C_minus = z_corr_matrix(psi_minus, Z_ops)
            C_tot_minus, _, _ = total_cost(cfg, geom, C_minus, edges, phases, squares)

            grad[e_idx] = (C_tot_plus - C_tot_minus) / (2.0 * eps)

            # restore
            phases[e_idx] = original

        # Gradient descent step
        phases = phases - cfg.phase_step * grad
        # Wrap into [-pi, pi)
        phases = (phases + np.pi) % (2.0 * np.pi) - np.pi

    # Final H, psi, correlations, fluxes for summary
    H_final = build_hamiltonian_with_gauge(cfg, adjacency, local_ops, edges, phases)
    E0_final, psi_final = ground_state(H_final)
    C_final = z_corr_matrix(psi_final, Z_ops)
    C_tot_final, C_ent_final, C_f_final = total_cost(cfg, geom, C_final, edges, phases, squares)

    square_flux_list = square_fluxes(edges, phases, squares) if squares else []

    return {
        "phases": phases,
        "edges": edges,
        "cost_history": cost_history,
        "H_final_energy": E0_final,
        "C_final": C_final,
        "cost_total_final": C_tot_final,
        "cost_ent_final": C_ent_final,
        "cost_flux_final": C_f_final,
        "adjacency": adjacency,
        "squares": squares,
        "square_flux_list": square_flux_list,
    }


# ---------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gauge flux experiment on emergent geometry."
    )
    p.add_argument(
        "--geometry",
        type=str,
        default="lr_embedding_3d.npz",
        help="Path to emergent geometry npz (must contain graph_dist).",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory for experiment runs.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag appended to run_id.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for initial phases.",
    )
    p.add_argument(
        "--J-coupling",
        type=float,
        default=1.0,
        help="Heisenberg coupling strength on edges.",
    )
    p.add_argument(
        "--h-field",
        type=float,
        default=0.2,
        help="On-site Z field strength.",
    )
    p.add_argument(
        "--defrag-zz",
        type=float,
        default=0.0,
        help="Extra ZZ coupling on edges (optional).",
    )
    p.add_argument(
        "--alpha-local-reward",
        type=float,
        default=0.5,
        help="Weight for rewarding nearest-neighbor correlations.",
    )
    p.add_argument(
        "--w-long-base",
        type=float,
        default=1.0,
        help="Base weight for long-range correlation penalty.",
    )
    p.add_argument(
        "--flux-weight",
        type=float,
        default=0.5,
        help="Weight for square-plaquette flux term (prefers ±π).",
    )
    p.add_argument(
        "--n-gauge-steps",
        type=int,
        default=30,
        help="Number of gradient-descent steps over link phases.",
    )
    p.add_argument(
        "--phase-step",
        type=float,
        default=0.1,
        help="Gradient-descent step size for phases.",
    )
    p.add_argument(
        "--fd-epsilon",
        type=float,
        default=1e-2,
        help="Finite-difference epsilon for phase derivatives.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Layout and logging
    try:
        layout = build_layout(output_root=args.output_root, tag=args.tag)
    except FileExistsError:
        print(
            f"[ERROR] Run directory already exists. "
            f"Try a different --tag or wait 1s.\n"
            f"output_root={args.output_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    log_f = open(layout.log_path, "w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        print(msg, file=log_f, flush=True)

    log("=" * 60)
    log("  Gauge Flux Experiment on Emergent Geometry (with square flux)")
    log("=" * 60)
    log(f"Run root:      {layout.run_root}")
    log(f"Geometry file: {args.geometry}")
    log(f"Random seed:   {args.seed}")
    log("-" * 60)

    try:
        geom = load_geometry(args.geometry)
    except Exception as e:  # noqa: BLE001
        log(f"[ERROR] Failed to load geometry: {e}")
        log_f.close()
        sys.exit(1)

    n_sites = geom.graph_dist.shape[0]
    log(f"Geometry n_sites: {n_sites}")
    log("-" * 60)

    gcfg = GaugeConfig(
        n_sites=n_sites,
        J_coupling=float(args.J_coupling),
        h_field=float(args.h_field),
        defrag_zz=float(args.defrag_zz),
        alpha_local_reward=float(args.alpha_local_reward),
        w_long_base=float(args.w_long_base),
        flux_weight=float(args.flux_weight),
        n_gauge_steps=int(args.n_gauge_steps),
        phase_step=float(args.phase_step),
        fd_epsilon=float(args.fd_epsilon),
        random_seed=int(args.seed),
    )

    # Save params + metadata
    params_payload = {
        "timestamp": _dt.datetime.now().isoformat(),
        "gauge_config": asdict(gcfg),
    }
    write_json(layout.params_json, params_payload)

    metadata_payload = {
        "script": os.path.basename(__file__),
        "run_root": os.path.abspath(layout.run_root),
        "geometry_file": os.path.abspath(args.geometry),
        "args": vars(args),
        "extras_in_geometry": list(geom.extras.keys()),
    }
    write_json(layout.metadata_json, metadata_payload)

    # Run gauge optimization
    try:
        result = gauge_gradient_descent(geom, gcfg, layout, log)
    except Exception as e:  # noqa: BLE001
        log(f"[ERROR] Gauge optimization failed: {e}")
        log_f.close()
        sys.exit(1)

    phases = result["phases"]
    edges = result["edges"]
    cost_history = result["cost_history"]
    cost_total_final = float(result["cost_total_final"])
    cost_ent_final = float(result["cost_ent_final"])
    cost_flux_final = float(result["cost_flux_final"])
    E0_final = float(result["H_final_energy"])
    squares = result["squares"]
    square_flux_list = result["square_flux_list"]

    # Save phases
    phases_payload = {
        "edges": edges,
        "phases": phases.tolist(),
    }
    write_json(layout.phases_json, phases_payload)

    # Save cost history
    write_json(layout.cost_history_json, cost_history)

    # Save fluxes
    fluxes_payload = {
        "squares": squares,
        "square_fluxes": square_flux_list,
    }
    write_json(layout.flux_json, fluxes_payload)

    # Summary
    summary_payload = {
        "timestamp": _dt.datetime.now().isoformat(),
        "gauge_config": asdict(gcfg),
        "final_cost_total": cost_total_final,
        "final_cost_entanglement": cost_ent_final,
        "final_cost_flux": cost_flux_final,
        "final_ground_state_energy": E0_final,
        "n_steps": len(cost_history),
        "cost_history": cost_history,
        "squares": squares,
        "square_fluxes": square_flux_list,
    }
    write_json(layout.summary_json, summary_payload)

    # Log short flux summary
    log("-" * 60)
    log(f"Final ground-state energy:   {E0_final:.6f}")
    log(f"Final total cost:            {cost_total_final:.6f}")
    log(f"  entanglement part:         {cost_ent_final:.6f}")
    log(f"  flux part (unweighted):    {cost_flux_final:.6f}")
    log("-" * 60)
    if square_flux_list:
        log("Square plaquette fluxes (wrapped to (-pi, pi]):")
        for fx in square_flux_list:
            sq = fx["plaquette"]
            fw = fx["flux_wrapped"]
            log(f"  square {sq}: flux ≈ {fw:.3f} rad")
    else:
        log("No square plaquettes (4-cycles) found in this geometry.")
    log("===============================================")

    log_f.close()


if __name__ == "__main__":
    main()
