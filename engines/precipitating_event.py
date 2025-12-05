"""
precipitating_event.py

Single-file Hilbert-substrate experiment with:

  1) Optional gauge-field self-organization on an emergent geometry.
  2) A Precipitating Event (quench) using that geometry and gauge background.

Philosophy:
- Only Hilbert space + constraints + emergent geometry.
- No ad hoc particle fields or hand-picked particle initial conditions.
- Gauge field = phases on links that track how information transforms
  as it moves through geometry (parallel transport).
- Precipitating Event = quench/cooling where localized "lumps" in <Z_i>
  emerge as proto-particles.

This version always computes a radial "wavefront" diagnostic, even if
no lumps are detected. If no dominant lump appears, the wave center is
chosen as the site whose <Z_i(t)> has the largest variance over time.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, asdict
import datetime as _dt
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    import cupy as cp  # noqa: F401

    HAS_CUPY = True
except Exception:  # noqa: BLE001
    HAS_CUPY = False


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def make_run_id(tag: str | None = None) -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if tag:
        tag = tag.strip()
        if tag:
            return f"{ts}_{tag}"
    return ts


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=False)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


# -----------------------------------------------------------------------------
# Geometry + layout
# -----------------------------------------------------------------------------


@dataclass
class GeometryAsset:
    graph_dist: np.ndarray       # (n_sites, n_sites) shortest-path distances
    coords: np.ndarray | None    # (n_sites, 3) or None


def load_geometry(path: str) -> GeometryAsset:
    arr = np.load(path)
    if "graph_dist" not in arr:
        raise KeyError(f"geometry npz must contain 'graph_dist', found: {arr.files}")
    graph_dist = np.array(arr["graph_dist"], dtype=float)
    coords = np.array(arr["coords"], dtype=float) if "coords" in arr else None
    return GeometryAsset(graph_dist=graph_dist, coords=coords)


def adjacency_from_graph_dist(graph_dist: np.ndarray) -> np.ndarray:
    """Build adjacency matrix: edge if distance == 1."""
    n = graph_dist.shape[0]
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j and abs(graph_dist[i, j] - 1.0) < 1e-9:
                adj[i, j] = 1
    return adj


def build_edge_list(adjacency: np.ndarray) -> List[Tuple[int, int]]:
    n = adjacency.shape[0]
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                edges.append((i, j))
    return edges


@dataclass
class RunLayout:
    run_root: str
    data_dir: str
    logs_dir: str
    figures_dir: str
    log_path: str
    summary_json: str
    timeseries_npz: str
    lump_hist_json: str
    dominant_lump_hist_json: str


def build_layout(output_root: str, tag: str | None = None) -> RunLayout:
    run_id = make_run_id(tag)
    run_root = os.path.join(output_root, "precipitating_event", run_id)

    data_dir = os.path.join(run_root, "data")
    logs_dir = os.path.join(run_root, "logs")
    figures_dir = os.path.join(run_root, "figures")

    ensure_dir(run_root)
    ensure_dir(data_dir)
    ensure_dir(logs_dir)
    ensure_dir(figures_dir)

    log_path = os.path.join(logs_dir, "precipitating_event.log")
    summary_json = os.path.join(run_root, "summary.json")
    timeseries_npz = os.path.join(data_dir, "timeseries.npz")
    lump_hist_json = os.path.join(data_dir, "lump_hist.json")
    dominant_lump_hist_json = os.path.join(data_dir, "dominant_lump_hist.json")

    return RunLayout(
        run_root=run_root,
        data_dir=data_dir,
        logs_dir=logs_dir,
        figures_dir=figures_dir,
        log_path=log_path,
        summary_json=summary_json,
        timeseries_npz=timeseries_npz,
        lump_hist_json=lump_hist_json,
        dominant_lump_hist_json=dominant_lump_hist_json,
    )


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class PrecipitationConfig:
    n_sites: int
    local_dim: int = 2
    J_coupling: float = 1.0
    h_field: float = 0.2
    defrag_hot: float = 0.3
    defrag_cold: float = 1.0
    t_total: float = 10.0
    t_quench: float = 4.0
    n_steps: int = 101
    z_threshold: float = 0.5
    snapshot_stride: int = 0  # if >0, save |psi> snapshots every this many steps


@dataclass
class GaugeConfig:
    n_sites: int
    J_coupling: float = 1.0
    h_field: float = 0.2
    defrag_zz: float = 0.0
    alpha_local_reward: float = 0.5
    w_long_base: float = 1.0
    flux_weight: float = 0.5
    n_gauge_steps: int = 100
    phase_step: float = 0.1
    rng_seed: int = 1234


# -----------------------------------------------------------------------------
# Local operators and Hamiltonian
# -----------------------------------------------------------------------------


def pauli_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return X, Y, Z


def build_local_ops(n_sites: int) -> Dict[str, List[np.ndarray]]:
    X1, Y1, Z1 = pauli_matrices()
    I2 = np.eye(2, dtype=complex)

    X_ops: List[np.ndarray] = []
    Y_ops: List[np.ndarray] = []
    Z_ops: List[np.ndarray] = []

    for site in range(n_sites):
        ops = []
        for which in range(3):
            op = 1
            for i in range(n_sites):
                if i == site:
                    op = np.kron(op, [X1, Y1, Z1][which])
                else:
                    op = np.kron(op, I2)
            ops.append(op)
        X_ops.append(ops[0])
        Y_ops.append(ops[1])
        Z_ops.append(ops[2])

    return {"X": X_ops, "Y": Y_ops, "Z": Z_ops}


def build_parity_operator(Z_ops: List[np.ndarray]) -> np.ndarray:
    """Parity operator: product of all Z_i."""
    P = np.eye(Z_ops[0].shape[0], dtype=complex)
    for Z in Z_ops:
        P = P @ Z
    return P


def diagonalize(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs


def evolve_spectral(evals: np.ndarray, evecs: np.ndarray, psi0: np.ndarray, t: float) -> np.ndarray:
    """Evolve psi0 under H with spectrum (evals, evecs) to time t."""
    coeffs = evecs.conj().T @ psi0
    phase = np.exp(-1j * evals * t)
    coeffs_t = phase * coeffs
    psi_t = evecs @ coeffs_t
    return psi_t


def build_hamiltonian_with_gauge(
    J: float,
    h: float,
    defrag_strength: float,
    adjacency: np.ndarray,
    local_ops: Dict[str, List[np.ndarray]],
    edges: List[Tuple[int, int]],
    edge_phases: Optional[Dict[Tuple[int, int], float]],
    use_plain_heisenberg_if_no_gauge: bool = True,
) -> np.ndarray:
    """
    Build Heisenberg-like Hamiltonian with optional edge phases.

    If edge_phases is None:
      - If use_plain_heisenberg_if_no_gauge: J*(XX+YY+ZZ) + defrag_strength*ZZ.
      - Else: J*(XX+YY) + defrag_strength*ZZ.

    If edge_phases is not None:
      - Implement a minimal gauge-like coupling where phases modulate XX/YY
        couplings between neighbors.
    """
    Xs = local_ops["X"]
    Ys = local_ops["Y"]
    Zs = local_ops["Z"]

    n_sites = len(Xs)
    dim = 2 ** n_sites
    H = np.zeros((dim, dim), dtype=complex)

    use_gauge = edge_phases is not None

    for (i, j) in edges:
        if use_gauge:
            key = (i, j) if i < j else (j, i)
            phi = float(edge_phases.get(key, 0.0))
            c = math.cos(phi)
            s = math.sin(phi)

            XX = Xs[i] @ Xs[j]
            YY = Ys[i] @ Ys[j]
            XY = Xs[i] @ Ys[j]
            YX = Ys[i] @ Xs[j]
            ZZ = Zs[i] @ Zs[j]

            H += J * (c * (XX + YY) + s * (XY - YX))
            H += defrag_strength * ZZ
        else:
            XX = Xs[i] @ Xs[j]
            YY = Ys[i] @ Ys[j]
            ZZ = Zs[i] @ Zs[j]
            if use_plain_heisenberg_if_no_gauge:
                H += J * (XX + YY + ZZ)
                H += defrag_strength * ZZ
            else:
                H += J * (XX + YY)
                H += defrag_strength * ZZ

    for i in range(n_sites):
        H += h * Zs[i]

    return H


# -----------------------------------------------------------------------------
# Gauge optimization (optional, simple hill-climb)
# -----------------------------------------------------------------------------


def initial_edge_phases(edges: List[Tuple[int, int]], rng: np.random.Generator) -> Dict[Tuple[int, int], float]:
    phases: Dict[Tuple[int, int], float] = {}
    for (i, j) in edges:
        key = (i, j) if i < j else (j, i)
        phases[key] = float(rng.uniform(-math.pi, math.pi))
    return phases


def plaquettes_from_adjacency(adjacency: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Very simple square plaquette finder for small graphs:
    look for i-j-k-l-i cycles with all distinct and adjacency=1 along edges.
    """
    n = adjacency.shape[0]
    squares: List[Tuple[int, int, int, int]] = []
    for i in range(n):
        for j in range(n):
            if i == j or not adjacency[i, j]:
                continue
            for k in range(n):
                if k in (i, j) or not adjacency[j, k]:
                    continue
                for l in range(n):
                    if l in (i, j, k) or not adjacency[k, l]:
                        continue
                    if adjacency[l, i]:
                        sq = (i, j, k, l)
                        squares.append(sq)
    unique: List[Tuple[int, int, int, int]] = []
    seen = set()
    for sq in squares:
        variants = [
            sq,
            (sq[1], sq[2], sq[3], sq[0]),
            (sq[2], sq[3], sq[0], sq[1]),
            (sq[3], sq[0], sq[1], sq[2]),
        ]
        variants += [tuple(reversed(v)) for v in variants]
        canon = min(variants)
        if canon not in seen:
            seen.add(canon)
            unique.append(canon)
    return unique


def compute_plaquette_fluxes(
    squares: List[Tuple[int, int, int, int]],
    edge_phases: Dict[Tuple[int, int], float],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for sq in squares:
        i, j, k, l = sq
        loop = [(i, j), (j, k), (k, l), (l, i)]
        flux = 0.0
        legs = []
        for (u, v) in loop:
            if u < v:
                key = (u, v)
                sign = +1.0
            else:
                key = (v, u)
                sign = -1.0
            phase_uv = float(edge_phases.get(key, 0.0))
            flux += sign * phase_uv
            legs.append({"edge": [u, v], "phase_used": sign * phase_uv})
        flux_wrapped = (flux + math.pi) % (2.0 * math.pi) - math.pi
        out.append(
            {
                "plaquette": sq,
                "flux_raw": flux,
                "flux_wrapped": flux_wrapped,
                "legs": legs,
            }
        )
    return out


def gauge_cost_function(
    geom: GeometryAsset,
    adjacency: np.ndarray,
    edges: List[Tuple[int, int]],
    edge_phases: Dict[Tuple[int, int], float],
    gcfg: GaugeConfig,
) -> float:
    """
    Simple cost: combination of flux penalties and long-range phase structure.

    We prefer:
      - plaquette fluxes near ±π (Z2-like structure) if flux_weight > 0,
      - phases that vary more slowly at longer graph distance.
    """
    squares = plaquettes_from_adjacency(adjacency)
    plaquette_info = compute_plaquette_fluxes(squares, edge_phases)

    cost_flux = 0.0
    for info in plaquette_info:
        phi = info["flux_wrapped"]
        cost_flux += (math.cos(phi) + 1.0) / 2.0

    cost_long = 0.0
    n_edges = len(edges)
    for (i, j) in edges:
        key = (i, j) if i < j else (j, i)
        phi = float(edge_phases.get(key, 0.0))
        d = float(geom.graph_dist[i, j])
        w = gcfg.w_long_base * (1.0 + d)
        cost_long += w * (1.0 - math.cos(phi))

    total_cost = gcfg.flux_weight * cost_flux + (1.0 - gcfg.flux_weight) * cost_long
    return float(total_cost / max(n_edges, 1))


def optimize_gauge_phases(
    geom: GeometryAsset,
    adjacency: np.ndarray,
    edges: List[Tuple[int, int]],
    gcfg: GaugeConfig,
) -> Dict[Tuple[int, int], float]:
    rng = np.random.default_rng(gcfg.rng_seed)
    edge_phases = initial_edge_phases(edges, rng)

    best_cost = gauge_cost_function(geom, adjacency, edges, edge_phases, gcfg)
    best_phases = dict(edge_phases)

    for _step in range(gcfg.n_gauge_steps):
        (i, j) = edges[rng.integers(0, len(edges))]
        key = (i, j) if i < j else (j, i)
        old_phi = edge_phases[key]

        proposal = old_phi + rng.normal(0.0, gcfg.phase_step)
        proposal = (proposal + math.pi) % (2.0 * math.pi) - math.pi
        edge_phases[key] = proposal

        cost = gauge_cost_function(geom, adjacency, edges, edge_phases, gcfg)
        if cost < best_cost:
            best_cost = cost
            best_phases = dict(edge_phases)
        else:
            edge_phases[key] = old_phi

    return best_phases


# -----------------------------------------------------------------------------
# Lumps and diagnostics
# -----------------------------------------------------------------------------


@dataclass
class LumpSnapshot:
    n_lumps: int
    lumps: List[List[int]]
    lump_sizes: List[int]


def find_lumps(
    z_profile: np.ndarray,
    adjacency: np.ndarray,
    z_threshold: float,
) -> LumpSnapshot:
    """
    Find connected components (lumps) where |Z_i - mean(Z)| > z_threshold.
    """
    n_sites = len(z_profile)
    z_mean = float(np.mean(z_profile))
    mask = np.abs(z_profile - z_mean) > z_threshold

    visited = np.zeros(n_sites, dtype=bool)
    lumps: List[List[int]] = []

    for i in range(n_sites):
        if not mask[i] or visited[i]:
            continue
        stack = [i]
        component: List[int] = []
        visited[i] = True
        while stack:
            u = stack.pop()
            component.append(u)
            for v in range(n_sites):
                if adjacency[u, v] and mask[v] and not visited[v]:
                    visited[v] = True
                    stack.append(v)
        lumps.append(sorted(component))

    lump_sizes = [len(L) for L in lumps]
    return LumpSnapshot(n_lumps=len(lumps), lumps=lumps, lump_sizes=lump_sizes)


def extract_dominant_lump_timeline(
    times: np.ndarray,
    lump_hist: List[List[List[int]]],
) -> Dict[str, Any]:
    """
    For each timestep, pick the largest lump (if any) as the 'dominant' one.
    """
    n_steps = len(times)
    dominant_sites: List[List[int]] = []
    dominant_sizes: List[int] = []

    for t_idx in range(n_steps):
        lumps_t = lump_hist[t_idx]
        if not lumps_t:
            dominant_sites.append([])
            dominant_sizes.append(0)
            continue
        best = max(lumps_t, key=lambda L: len(L))
        dominant_sites.append(list(best))
        dominant_sizes.append(len(best))

    dom_sizes_arr = np.array(dominant_sizes, dtype=int)
    has_any = dom_sizes_arr > 0
    n_any = int(np.sum(has_any))
    frac_any = float(n_any) / float(n_steps) if n_steps > 0 else 0.0
    max_size = int(np.max(dom_sizes_arr)) if n_steps > 0 else 0
    mean_size = float(np.mean(dom_sizes_arr)) if n_steps > 0 else 0.0

    if n_any > 0:
        first_idx = int(np.argmax(has_any))
        last_idx = int(len(has_any) - 1 - np.argmax(has_any[::-1]))
        first_time = float(times[first_idx])
        last_time = float(times[last_idx])
    else:
        first_idx = -1
        last_idx = -1
        first_time = None
        last_time = None

    metrics = {
        "timesteps_with_any_lump": n_any,
        "fraction_time_with_any_lump": frac_any,
        "max_dominant_lump_size": max_size,
        "mean_dominant_lump_size": mean_size,
        "first_lump_t_index": first_idx,
        "last_lump_t_index": last_idx,
        "first_lump_time": first_time,
        "last_lump_time": last_time,
    }

    return {"dominant_sites": dominant_sites, "dominant_sizes": dominant_sizes, "metrics": metrics}


def local_z_expectations_full(psi: np.ndarray, Z_ops: List[np.ndarray]) -> np.ndarray:
    """
    Compute <Z_i> for all sites given full Hilbert state psi.
    """
    z_vals = []
    for Z in Z_ops:
        z_vals.append(float(np.vdot(psi, Z @ psi).real))
    return np.array(z_vals, dtype=float)


def total_occupation_from_z(z_profile: np.ndarray) -> float:
    """
    Define n_i = (1 - Z_i)/2, interpret as local excitation, sum over sites.
    """
    n_i = 0.5 * (1.0 - z_profile)
    return float(np.sum(n_i))


def parity_expectation(psi: np.ndarray, P_op: np.ndarray) -> float:
    val = np.vdot(psi, P_op @ psi)
    return float(val.real)


def random_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi /= np.linalg.norm(psi)
    return psi


# -----------------------------------------------------------------------------
# Core experiment (with waves + snapshots)
# -----------------------------------------------------------------------------


def run_precipitating_event(
    geom: GeometryAsset,
    cfg: PrecipitationConfig,
    seed: int,
    adjacency: np.ndarray,
    edges: List[Tuple[int, int]],
    local_ops: Dict[str, List[np.ndarray]],
    edge_phases: Optional[Dict[Tuple[int, int], float]],
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_sites = cfg.n_sites

    Z_ops = local_ops["Z"]
    P_op = build_parity_operator(Z_ops)

    H_hot = build_hamiltonian_with_gauge(
        J=cfg.J_coupling,
        h=cfg.h_field,
        defrag_strength=cfg.defrag_hot,
        adjacency=adjacency,
        local_ops=local_ops,
        edges=edges,
        edge_phases=edge_phases,
        use_plain_heisenberg_if_no_gauge=True,
    )
    H_cold = build_hamiltonian_with_gauge(
        J=cfg.J_coupling,
        h=cfg.h_field,
        defrag_strength=cfg.defrag_cold,
        adjacency=adjacency,
        local_ops=local_ops,
        edges=edges,
        edge_phases=edge_phases,
        use_plain_heisenberg_if_no_gauge=True,
    )

    evals_hot, evecs_hot = diagonalize(H_hot)
    evals_cold, evecs_cold = diagonalize(H_cold)

    times = np.linspace(0.0, cfg.t_total, cfg.n_steps)
    if not (0.0 < cfg.t_quench < cfg.t_total):
        raise ValueError("Require 0 < t_quench < t_total.")
    k_quench = int(np.argmin(np.abs(times - cfg.t_quench)))
    t_quench_eff = float(times[k_quench])

    dim = cfg.local_dim ** cfg.n_sites
    psi0 = random_state(dim, rng)
    psi_quench = evolve_spectral(evals_hot, evecs_hot, psi0, t_quench_eff)

    local_z_t = np.zeros((cfg.n_steps, n_sites), dtype=float)
    lump_counts = np.zeros(cfg.n_steps, dtype=int)
    lump_hist: List[List[List[int]]] = []
    parity_t = np.zeros(cfg.n_steps, dtype=float)
    total_N_t = np.zeros(cfg.n_steps, dtype=float)

    # Internal-path diagnostics
    occupancy_t = np.zeros((cfg.n_steps, n_sites), dtype=int)
    hamming_t = np.zeros(cfg.n_steps, dtype=int)

    # Optional full-state snapshots
    snapshot_indices: List[int] = []
    snapshot_psi: List[np.ndarray] = []

    for idx, t in enumerate(times):
        if t <= t_quench_eff + 1e-12:
            psi_t = evolve_spectral(evals_hot, evecs_hot, psi0, t)
        else:
            dt = float(t - t_quench_eff)
            psi_t = evolve_spectral(evals_cold, evecs_cold, psi_quench, dt)

        # Save snapshot if requested
        if cfg.snapshot_stride > 0 and (idx % cfg.snapshot_stride == 0):
            snapshot_indices.append(int(idx))
            snapshot_psi.append(psi_t.astype(complex).ravel())

        z_profile = local_z_expectations_full(psi_t, Z_ops)
        local_z_t[idx, :] = z_profile
        total_N_t[idx] = total_occupation_from_z(z_profile)
        parity_t[idx] = parity_expectation(psi_t, P_op)

        # Find lumps in Z profile
        snap = find_lumps(
            z_profile=z_profile,
            adjacency=adjacency,
            z_threshold=cfg.z_threshold,
        )
        lump_counts[idx] = snap.n_lumps
        lump_hist.append(snap.lumps)

        # Coarse occupancy pattern from Z:
        n_i = 0.5 * (1.0 - z_profile)
        occ = (n_i >= 0.5).astype(int)
        occupancy_t[idx, :] = occ
        if idx > 0:
            hamming_t[idx] = int(np.sum(np.bitwise_xor(occupancy_t[idx - 1, :], occ)))

    final_z = local_z_t[-1, :]
    final_snap = find_lumps(
        z_profile=final_z,
        adjacency=adjacency,
        z_threshold=cfg.z_threshold,
    )
    final_lump_sizes = final_snap.lump_sizes

    metrics_basic: Dict[str, Any] = {
        "t_quench_index": int(k_quench),
        "t_quench_effective": t_quench_eff,
        "final_n_lumps": int(final_snap.n_lumps),
        "final_lump_sizes": final_lump_sizes,
        "mean_lump_count": float(np.mean(lump_counts)),
        "has_particle_candidates": bool(final_snap.n_lumps > 0),
    }

    dom_info = extract_dominant_lump_timeline(times, lump_hist)
    dominant_metrics = dom_info["metrics"]

    # Internal-change detection:
    centers = np.full(cfg.n_steps, -1, dtype=int)
    for t_idx, sites in enumerate(dom_info["dominant_sites"]):
        if sites:
            centers[t_idx] = int(sites[0])

    internal_change_t = np.zeros(cfg.n_steps, dtype=bool)
    for t_idx in range(1, cfg.n_steps):
        if centers[t_idx] != -1 and centers[t_idx] == centers[t_idx - 1]:
            if hamming_t[t_idx] > 0:
                internal_change_t[t_idx] = True

    internal_change_count = int(np.sum(internal_change_t))
    denom = max(cfg.n_steps - 1, 1)
    internal_change_fraction = float(internal_change_count) / float(denom)

    # Finalize snapshots
    if snapshot_indices:
        snapshot_indices_arr = np.array(snapshot_indices, dtype=int)
        snapshot_psi_arr = np.stack(snapshot_psi, axis=0)
    else:
        snapshot_indices_arr = np.zeros((0,), dtype=int)
        snapshot_psi_arr = np.zeros((0, dim), dtype=complex)

    parity_mean = float(np.mean(parity_t))
    parity_std = float(np.std(parity_t))
    N_mean = float(np.mean(total_N_t))
    N_std = float(np.std(total_N_t))

    # Wavefront diagnostics (ALWAYS compute, with a sensible center)
    # 1) Prefer first time a dominant lump appears
    first_dom_idx = dominant_metrics.get("first_lump_t_index", -1)
    center_site: int

    if first_dom_idx is not None and int(first_dom_idx) >= 0:
        first_dom_idx = int(first_dom_idx)
        dom_sites_at_first = dom_info["dominant_sites"][first_dom_idx]
        if dom_sites_at_first:
            center_site = int(dom_sites_at_first[0])
        else:
            # Fallback to max-variance site
            var_per_site = local_z_t.var(axis=0)
            center_site = int(np.argmax(var_per_site))
    else:
        # No dominant lump at any time: choose the site with maximal Z variance.
        var_per_site = local_z_t.var(axis=0)
        center_site = int(np.argmax(var_per_site))

    dist_from_center = np.asarray(geom.graph_dist[center_site], dtype=float)
    finite_mask = np.isfinite(dist_from_center)
    if not np.any(finite_mask):
        # degenerate geometry; just treat all as shell 0
        unique_shells = [0]
        shell_index_lists: List[np.ndarray] = [np.arange(n_sites)]
    else:
        unique_shells = sorted({int(round(d)) for d in dist_from_center[finite_mask]})
        shell_index_lists = [
            np.where(np.round(dist_from_center) == s)[0] for s in unique_shells
        ]

    wave_intensity = np.zeros((cfg.n_steps, len(unique_shells)), dtype=float)
    for t_idx in range(1, cfg.n_steps):
        dz = local_z_t[t_idx, :] - local_z_t[t_idx - 1, :]
        abs_dz = np.abs(dz)
        for j_shell, idx_array in enumerate(shell_index_lists):
            if idx_array.size > 0:
                wave_intensity[t_idx, j_shell] = float(abs_dz[idx_array].mean())

    wave_info: Dict[str, Any] = {
        "center_site": center_site,
        "shells": [int(s) for s in unique_shells],
        "wave_intensity": wave_intensity,
    }

    metrics = {
        **metrics_basic,
        **{
            "timesteps_with_any_lump": dominant_metrics["timesteps_with_any_lump"],
            "fraction_time_with_any_lump": dominant_metrics["fraction_time_with_any_lump"],
            "max_dominant_lump_size": dominant_metrics["max_dominant_lump_size"],
            "mean_dominant_lump_size": dominant_metrics["mean_dominant_lump_size"],
            "first_lump_t_index": dominant_metrics["first_lump_t_index"],
            "last_lump_t_index": dominant_metrics["last_lump_t_index"],
            "first_lump_time": dominant_metrics["first_lump_time"],
            "last_lump_time": dominant_metrics["last_lump_time"],
            "parity_mean": parity_mean,
            "parity_std": parity_std,
            "N_mean": N_mean,
            "N_std": N_std,
            "uses_gauge_phases": bool(edge_phases is not None),
            "internal_change_count": internal_change_count,
            "internal_change_fraction": internal_change_fraction,
            "has_internal_dynamics": bool(internal_change_count > 0),
        },
    }

    return {
        "config": asdict(cfg),
        "times": times,
        "local_z_t": local_z_t,
        "lump_counts": lump_counts,
        "lump_hist": lump_hist,
        "dominant_lump_sizes": dom_info["dominant_sizes"],
        "dominant_lump_sites": dom_info["dominant_sites"],
        "parity_t": parity_t,
        "total_N_t": total_N_t,
        # internal-path diagnostics:
        "occupancy_t": occupancy_t,
        "hamming_t": hamming_t,
        "internal_change_t": internal_change_t.tolist(),
        # full-state snapshots:
        "snapshot_indices": snapshot_indices_arr,
        "snapshot_psi": snapshot_psi_arr,
        # wavefront diagnostics (always present):
        "wave_info": wave_info,
        "metrics": metrics,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Hilbert-substrate Precipitating Event on emergent geometry "
            "(fermion-ready, with optional internal gauge optimization, CUDA-aware)."
        )
    )

    # Paths / run organisation
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
        help="Root directory to hold all runs.",
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
        default=117,
        help="Random seed used for both gauge and event.",
    )

    # Precipitating Event parameters
    p.add_argument("--t-total", type=float, default=10.0)
    p.add_argument("--t-quench", type=float, default=4.0)
    p.add_argument("--n-steps", type=int, default=101)
    p.add_argument("--J-coupling", type=float, default=1.0)
    p.add_argument("--h-field", type=float, default=0.2)
    p.add_argument("--defrag-hot", type=float, default=0.3)
    p.add_argument("--defrag-cold", type=float, default=1.0)
    p.add_argument("--z-threshold", type=float, default=0.5)
    p.add_argument(
        "--snapshot-stride",
        type=int,
        default=0,
        help=(
            "If >0, save full |psi> snapshots every this many steps into snapshots.npz "
            "for downstream diagnostics."
        ),
    )

    # Gauge control
    p.add_argument(
        "--use-gauge",
        action="store_true",
        help="If set, run internal gauge optimization before the event.",
    )
    p.add_argument(
        "--gauge-flux-weight",
        type=float,
        default=0.5,
        help="Weight for square-plaquette flux term (prefers ±π).",
    )
    p.add_argument(
        "--gauge-alpha-local-reward",
        type=float,
        default=0.5,
        help="Weight on local edge reward term.",
    )
    p.add_argument(
        "--gauge-w-long-base",
        type=float,
        default=1.0,
        help="Base weight for long-range phase-smoothing term.",
    )
    p.add_argument(
        "--gauge-defrag-zz",
        type=float,
        default=0.0,
        help="ZZ defrag strength used in gauge-evaluation Hamiltonian.",
    )
    p.add_argument(
        "--gauge-n-steps",
        type=int,
        default=200,
        help="Number of gauge hill-climb steps.",
    )
    p.add_argument(
        "--gauge-phase-step",
        type=float,
        default=0.2,
        help="Std dev of random phase proposals in hill-climb.",
    )

    return p.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Layout + logging
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
    log("  Hilbert-Substrate Precipitating Event (single-file, gauge-ready, CUDA-aware)")
    log("=" * 60)
    log(f"Run root:      {layout.run_root}")
    log(f"Geometry file: {args.geometry}")
    log(f"Seed:          {args.seed}")
    log(f"Use gauge:     {args.use_gauge}")
    log(f"CuPy/CUDA:     {'ENABLED' if HAS_CUPY else 'DISABLED'}")
    log("-" * 60)

    # Load geometry
    try:
        geom = load_geometry(args.geometry)
    except Exception as e:  # noqa: BLE001
        log(f"[ERROR] Failed to load geometry: {e}")
        log_f.close()
        sys.exit(1)

    n_sites = geom.graph_dist.shape[0]
    adjacency = adjacency_from_graph_dist(geom.graph_dist)
    edges = build_edge_list(adjacency)
    local_ops = build_local_ops(n_sites)

    log(f"Geometry n_sites: {n_sites}")
    log(f"Edges:            {edges}")
    log("-" * 60)

    # Precipitation config
    pcfg = PrecipitationConfig(
        n_sites=n_sites,
        local_dim=2,
        J_coupling=float(args.J_coupling),
        h_field=float(args.h_field),
        defrag_hot=float(args.defrag_hot),
        defrag_cold=float(args.defrag_cold),
        t_total=float(args.t_total),
        t_quench=float(args.t_quench),
        n_steps=int(args.n_steps),
        z_threshold=float(args.z_threshold),
        snapshot_stride=int(args.snapshot_stride),
    )

    # Gauge config (if used)
    if args.use_gauge:
        gcfg = GaugeConfig(
            n_sites=n_sites,
            J_coupling=float(args.J_coupling),
            h_field=float(args.h_field),
            defrag_zz=float(args.gauge_defrag_zz),
            alpha_local_reward=float(args.gauge_alpha_local_reward),
            w_long_base=float(args.gauge_w_long_base),
            flux_weight=float(args.gauge_flux_weight),
            n_gauge_steps=int(args.gauge_n_steps),
            phase_step=float(args.gauge_phase_step),
            rng_seed=int(args.seed),
        )
        log("[Gauge] Starting internal gauge-field optimization...")
        edge_phases = optimize_gauge_phases(geom, adjacency, edges, gcfg)
        log("[Gauge] Optimization complete.")
    else:
        edge_phases = None
        log("[Gauge] Skipping gauge optimization; using plain Heisenberg-like coupling.")

    # Run the actual Precipitating Event
    log("[Event] Running precipitating event...")
    results = run_precipitating_event(
        geom=geom,
        cfg=pcfg,
        seed=int(args.seed),
        adjacency=adjacency,
        edges=edges,
        local_ops=local_ops,
        edge_phases=edge_phases,
    )
    log("[Event] Done.")

    # Save timeseries and lump history
    np.savez_compressed(
        layout.timeseries_npz,
        times=results["times"],
        local_z_t=results["local_z_t"],
        lump_counts=results["lump_counts"],
        dominant_lump_sizes=np.array(results["dominant_lump_sizes"], dtype=int),
        parity_t=results["parity_t"],
        total_N_t=results["total_N_t"],
        occupancy_t=results["occupancy_t"],
        hamming_t=results["hamming_t"],
        internal_change_t=np.array(results["internal_change_t"], dtype=bool),
    )
    # Also save a copy of timeseries.npz at the run root for diagnostics
    np.savez_compressed(
        os.path.join(layout.run_root, "timeseries.npz"),
        times=results["times"],
        occupancy_t=results["occupancy_t"],
        hamming_t=results["hamming_t"],
        internal_change_t=np.array(results["internal_change_t"], dtype=bool),
        lump_counts=results["lump_counts"],
        dominant_lump_sizes=np.array(results["dominant_lump_sizes"], dtype=int),
        parity_t=results["parity_t"],
        total_N_t=results["total_N_t"],
        local_z_t=results["local_z_t"],
    )

    # Lumps history (per-timestep membership)
    write_json(layout.lump_hist_json, results["lump_hist"])
    write_json(layout.dominant_lump_hist_json, results["dominant_lump_sites"])

    # Also save a compact lumps.json at the run root for exchange/phase diagnostics
    lumps_payload = {
        "times": [float(t) for t in results["times"]],
        "lump_counts": [int(c) for c in results["lump_counts"]],
        "lump_memberships": results["lump_hist"],
        "z_threshold": float(pcfg.z_threshold),
    }
    write_json(os.path.join(layout.run_root, "lumps.json"), lumps_payload)

    # Save snapshots if any
    if results["snapshot_indices"].size > 0:
        np.savez_compressed(
            os.path.join(layout.run_root, "snapshots.npz"),
            indices=np.array(results["snapshot_indices"], dtype=int),
            psi=np.array(results["snapshot_psi"], dtype=complex),
        )

    # Save wavefront diagnostics (ALWAYS written now)
    wave = results["wave_info"]
    np.savez_compressed(
        os.path.join(layout.data_dir, "wave_series.npz"),
        center_site=int(wave["center_site"]),
        shells=np.array(wave["shells"], dtype=int),
        wave_intensity=np.array(wave["wave_intensity"], dtype=float),
    )

    # Build and save summary
    metrics = results["metrics"]
    summary_payload = {
        "timestamp": _dt.datetime.now().isoformat(),
        "precipitation_config": dataclasses.asdict(pcfg),
        "metrics": metrics,
    }
    write_json(layout.summary_json, summary_payload)

    log("==== Precipitating Event Summary ====")
    log(f"t_quench_effective:      {metrics['t_quench_effective']:.6f}")
    log(f"final_n_lumps:           {metrics['final_n_lumps']}")
    log(f"final_lump_sizes:        {metrics['final_lump_sizes']}")
    log(f"mean_lump_count:         {metrics['mean_lump_count']:.3f}")
    log(f"has_candidates:          {metrics['has_particle_candidates']}")
    log(f"timesteps_with_any_lump: {metrics['timesteps_with_any_lump']}")
    log(f"fraction_time_with_any:  {metrics['fraction_time_with_any_lump']:.3f}")
    log(f"max_dominant_size:       {metrics['max_dominant_lump_size']}")
    log(f"mean_dominant_size:      {metrics['mean_dominant_lump_size']:.3f}")
    log(f"first_lump_time:         {metrics['first_lump_time']}")
    log(f"last_lump_time:          {metrics['last_lump_time']}")
    log(f"parity_mean:             {metrics['parity_mean']:.6f}")
    log(f"parity_std:              {metrics['parity_std']:.6f}")
    log(f"N_mean:                  {metrics['N_mean']:.6f}")
    log(f"N_std:                   {metrics['N_std']:.6f}")
    log(f"internal_change_count:   {metrics['internal_change_count']}")
    log(f"internal_change_frac:    {metrics['internal_change_fraction']:.6f}")
    log(f"has_internal_dynamics:   {metrics['has_internal_dynamics']}")
    log("======================================")

    log_f.close()


if __name__ == "__main__":
    main()
