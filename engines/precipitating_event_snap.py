"""
precipitating_event_snap.py

Standalone "Precipitating Event" engine with:
  - geometry from NPZ
      * coords from 'coords' or 'X'
      * edges from 'edges' or inferred from 'graph_dist == 1'
  - simple spin-1/2 substrate on that graph
  - quench + defrag-like longitudinal field
  - lump detection from <Z_i(t)>
  - full-state snapshots: snapshots.npz

This is a fresh engine; it does NOT depend on your old
precipitating_event.py. You can run them side-by-side.

Model (toy but coherent):
  - Spins live on graph nodes
  - H(t) = J sum_edges σ_z σ_z + h_x sum_i σ_x + λ(t) sum_i σ_z
  - λ(t) switches from defrag_hot to defrag_cold at t_quench

Time evolution:
  - First-order Suzuki–Trotter:
      U(dt) ≈ exp(-i dt H_zz+defrag) exp(-i dt H_x)
  - H_zz+defrag is diagonal, applied as phases in computational basis
  - H_x acts per-site using 2×2 rotations in σ_x

Outputs inside run_root:
  - params.json    : run configuration
  - summary.json   : lump stats
  - z_series.npz   : times, Z[i,t] expectation values
  - lumps.json     : lump counts & memberships over time
  - snapshots.npz  : indices[], psi[] full state snapshots
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False
    cp = None  # type: ignore


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------


@dataclass
class Config:
    geometry: str
    output_root: str
    tag: str
    seed: int
    J_coupling: float
    h_field_x: float
    defrag_hot: float
    defrag_cold: float
    t_total: float
    t_quench: float
    n_steps: int
    z_threshold: float
    use_cuda: bool
    snapshot_indices: List[int]
    snapshot_every: int


@dataclass
class Geometry:
    coords: np.ndarray  # (n_sites, 3)
    edges: np.ndarray   # (n_edges, 2)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def load_geometry(path: str) -> Geometry:
    """
    Load geometry NPZ robustly.

    Supports:
      - Original LR files:      X, graph_dist
      - Upscaled grid files:    coords, edges
      - Mixed:                  X, edges

    Edges will be:
      - data['edges'] if present, OR
      - inferred as { (i,j) | graph_dist[i,j] == 1 } if graph_dist present.
    """
    data = np.load(path)
    keys = set(data.keys())

    # coords
    if "coords" in keys:
        coords = data["coords"]
    elif "X" in keys:
        coords = data["X"]
    else:
        raise ValueError(
            f"Geometry NPZ must contain either 'coords' or 'X' for coordinates: {path}"
        )

    # edges
    if "edges" in keys:
        edges = data["edges"]
    elif "graph_dist" in keys:
        gd = data["graph_dist"]
        if gd.ndim != 2 or gd.shape[0] != gd.shape[1]:
            raise ValueError("graph_dist must be a square matrix if used to infer edges.")
        n = gd.shape[0]
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(gd[i, j] - 1.0) < 1e-8:
                    edge_list.append((int(i), int(j)))
        edges = np.array(edge_list, dtype=int)
    else:
        raise ValueError(
            f"Geometry NPZ must contain 'edges' or 'graph_dist' to infer edges: {path}"
        )

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (n_edges, 2)")

    return Geometry(coords=coords, edges=edges.astype(int))


def build_basis_info(n_sites: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute σ_z eigenvalues for each (site, basis_state).
    basis_state index k corresponds to spins given by bits of k:
      bit=0 -> +1 eigenvalue (up)
      bit=1 -> -1 eigenvalue (down)
    """
    dim = 1 << n_sites
    z_eigs = np.empty((n_sites, dim), dtype=np.int8)
    for i in range(n_sites):
        bit = 1 << i
        for k in range(dim):
            z_eigs[i, k] = 1 if ((k & bit) == 0) else -1
    return z_eigs, np.arange(dim, dtype=np.uint32)


def apply_zz_defrag_phase(
    psi: np.ndarray,
    z_eigs: np.ndarray,
    edges: np.ndarray,
    lam: float,
    J: float,
    dt: float,
) -> None:
    """
    Apply diagonal phase from:
      H_zz+defrag = J sum_{(i,j)} σ_z^i σ_z^j + lam sum_i σ_z^i
    directly in computational basis.
    """
    n_sites, dim = z_eigs.shape
    s = z_eigs

    sum_z = s.sum(axis=0)  # (dim,)
    zz = np.zeros(dim, dtype=np.int32)
    for (i, j) in edges:
        zz += s[int(i)] * s[int(j)]
    E = J * zz + lam * sum_z
    phase = np.exp(-1j * dt * E.astype(np.float64))
    psi *= phase


def apply_hx(
    psi: np.ndarray,
    n_sites: int,
    h_x: float,
    dt: float,
) -> None:
    """
    Apply exp(-i dt h_x sum_i σ_x^i) via per-site 2×2 rotations.
    U_x = cos(h_x dt) I - i sin(h_x dt) σ_x
        = [[c, -i s],
           [-i s, c]]
    """
    dim = psi.shape[0]
    c = math.cos(h_x * dt)
    s = math.sin(h_x * dt)

    for i in range(n_sites):
        bit = 1 << i
        for base in range(dim):
            if base & bit:
                continue
            up = base
            dn = base | bit
            a = psi[up]
            b = psi[dn]
            psi[up] = c * a - 1j * s * b
            psi[dn] = -1j * s * a + c * b


def expectation_z(
    psi: np.ndarray,
    z_eigs: np.ndarray,
) -> np.ndarray:
    """
    Compute <σ_z^i> for each site i given state psi.
    psi is 1D complex array of length dim.
    z_eigs[i,k] = ±1 is eigenvalue for site i, basis k.
    """
    probs = np.abs(psi) ** 2
    return (z_eigs * probs[np.newaxis, :]).sum(axis=1)


def detect_lumps(
    z_vals: np.ndarray,
    threshold: float,
    edges: np.ndarray,
) -> List[List[int]]:
    """
    Simple lump finder:
      - threshold on |Z_i - mean(Z)| >= threshold
      - group above-threshold sites into connected components
        via the geometry edges.
    """
    n_sites = z_vals.shape[0]
    mean_z = float(z_vals.mean())
    dev = np.abs(z_vals - mean_z)
    active = dev >= threshold

    adj = [[] for _ in range(n_sites)]
    for (i, j) in edges:
        i = int(i)
        j = int(j)
        adj[i].append(j)
        adj[j].append(i)

    visited = np.zeros(n_sites, dtype=bool)
    lumps: List[List[int]] = []

    for i in range(n_sites):
        if not active[i] or visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp: List[int] = [i]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if active[v] and not visited[v]:
                    visited[v] = True
                    stack.append(v)
                    comp.append(int(v))
        lumps.append(sorted(comp))

    return lumps


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Hilbert-Substrate Precipitating Event (snapshot-enabled, self-contained)."
    )

    p.add_argument("--geometry", type=str, required=True, help="Path to NPZ geometry.")
    p.add_argument("--output-root", type=str, default="outputs", help="Root directory for runs.")
    p.add_argument("--tag", type=str, default="hilbert_quench", help="Tag name for this run.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")

    p.add_argument("--J-coupling", type=float, default=1.0, help="Ising ZZ coupling J.")
    p.add_argument("--h-field-x", type=float, default=1.0, help="Transverse field strength (σ_x).")

    p.add_argument("--defrag-hot", type=float, default=0.3, help="Pre-quench longitudinal field (λ_hot).")
    p.add_argument("--defrag-cold", type=float, default=1.0, help="Post-quench longitudinal field (λ_cold).")

    p.add_argument("--t-total", type=float, default=10.0, help="Total simulation time.")
    p.add_argument("--t-quench", type=float, default=4.0, help="Quench time (switch defrag hot→cold).")
    p.add_argument("--n-steps", type=int, default=101, help="Number of time steps (including t=0).")

    p.add_argument("--z-threshold", type=float, default=0.5, help="Deviation threshold for lumps.")

    p.add_argument(
        "--use-cuda",
        action="store_true",
        help="Use CuPy on GPU if available (state & observables; evolution is still done in Python loops).",
    )

    p.add_argument(
        "--snapshot-indices",
        type=str,
        default="",
        help="Comma-separated list of integer time indices to save full-state snapshots.",
    )
    p.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="If >0, also save snapshots every K steps (0 disables).",
    )

    args = p.parse_args()

    snapshot_indices: List[int] = []
    if args.snapshot_indices:
        for tok in args.snapshot_indices.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                snapshot_indices.append(int(tok))
            except ValueError:
                raise ValueError(f"Invalid entry in --snapshot-indices: {tok!r}")
    snapshot_every = max(0, int(args.snapshot_every))

    use_cuda = bool(args.use_cuda and HAS_CUPY)

    cfg = Config(
        geometry=args.geometry,
        output_root=args.output_root,
        tag=args.tag,
        seed=int(args.seed),
        J_coupling=float(args.__dict__["J_coupling"]),
        h_field_x=float(args.h_field_x),
        defrag_hot=float(args.defrag_hot),
        defrag_cold=float(args.defrag_cold),
        t_total=float(args.t_total),
        t_quench=float(args.t_quench),
        n_steps=int(args.n_steps),
        z_threshold=float(args.z_threshold),
        use_cuda=use_cuda,
        snapshot_indices=sorted(set(snapshot_indices)),
        snapshot_every=snapshot_every,
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    np.random.seed(cfg.seed)

    # Run directory
    tstamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(
        cfg.output_root,
        "precipitating_event_snap",
        f"{tstamp}_{cfg.tag}",
    )
    os.makedirs(run_root, exist_ok=True)

    params = asdict(cfg)
    params["run_root"] = run_root
    with open(os.path.join(run_root, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print("============================================================")
    print("  Hilbert-Substrate Precipitating Event (snapshot engine)")
    print("============================================================")
    print(f"Run root:      {run_root}")
    print(f"Geometry file: {cfg.geometry}")
    print(f"Seed:          {cfg.seed}")
    print(f"Use CUDA:      {cfg.use_cuda} (CuPy available: {HAS_CUPY})")
    print("------------------------------------------------------------")

    # Geometry
    geom = load_geometry(cfg.geometry)
    n_sites = geom.coords.shape[0]
    print(f"Geometry n_sites: {n_sites}")
    print(f"Edges (n={geom.edges.shape[0]}): {geom.edges.tolist()}")
    print("------------------------------------------------------------")

    # Basis
    z_eigs, basis_indices = build_basis_info(n_sites)
    dim = 1 << n_sites
    print(f"Hilbert dim: {dim}")
    print("------------------------------------------------------------")

    # Initial random state
    rng = np.random.default_rng(cfg.seed)
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi = psi / np.linalg.norm(psi)

    psi_gpu = None
    if cfg.use_cuda:
        psi_gpu = cp.asarray(psi)

    times = np.linspace(0.0, cfg.t_total, cfg.n_steps)
    dt = times[1] - times[0] if cfg.n_steps > 1 else cfg.t_total

    Z_series = np.zeros((cfg.n_steps, n_sites), dtype=np.float64)
    lump_counts: List[int] = []
    lump_sizes: List[List[int]] = []
    lump_memberships: List[List[List[int]]] = []

    snapshot_states: Dict[int, np.ndarray] = {}

    print("Starting Precipitating Event evolution...")
    print("------------------------------------------------------------")

    for step_idx, t in enumerate(times):
        if step_idx > 0:
            lam = cfg.defrag_hot if t < cfg.t_quench else cfg.defrag_cold

            apply_zz_defrag_phase(
                psi=psi,
                z_eigs=z_eigs,
                edges=geom.edges,
                lam=lam,
                J=cfg.J_coupling,
                dt=dt / 2.0,
            )

            apply_hx(
                psi=psi,
                n_sites=n_sites,
                h_x=cfg.h_field_x,
                dt=dt,
            )

            apply_zz_defrag_phase(
                psi=psi,
                z_eigs=z_eigs,
                edges=geom.edges,
                lam=lam,
                J=cfg.J_coupling,
                dt=dt / 2.0,
            )

            psi = psi / np.linalg.norm(psi)

            if cfg.use_cuda:
                psi_gpu = cp.asarray(psi)

        # Observables
        Z = expectation_z(psi, z_eigs)
        Z_series[step_idx] = Z

        # Lumps
        lumps = detect_lumps(Z, cfg.z_threshold, geom.edges)
        lump_counts.append(int(len(lumps)))
        lump_sizes.append([int(len(L)) for L in lumps])
        lump_memberships.append([[int(v) for v in L] for L in lumps])

        # Snapshots
        take_snap = False
        if step_idx in cfg.snapshot_indices:
            take_snap = True
        if cfg.snapshot_every > 0 and (step_idx % cfg.snapshot_every == 0):
            take_snap = True
        if take_snap:
            snapshot_states[int(step_idx)] = psi.copy()

    print("Evolution complete.")
    print("------------------------------------------------------------")

    # Save Z-series
    z_path = os.path.join(run_root, "z_series.npz")
    np.savez_compressed(z_path, times=times, Z=Z_series)
    print(f"Saved Z(t) series to {z_path}")

    # Convert lumps to pure Python types for JSON
    lumps_payload = {
        "times": [float(t) for t in times],
        "lump_counts": [int(c) for c in lump_counts],
        "lump_sizes": [[int(s) for s in sizes] for sizes in lump_sizes],
        "lump_memberships": [
            [[int(v) for v in comp] for comp in comps] for comps in lump_memberships
        ],
    }
    with open(os.path.join(run_root, "lumps.json"), "w", encoding="utf-8") as f:
        json.dump(lumps_payload, f, indent=2)

    # Summary
    total_lumps = sum(lumps_payload["lump_counts"])
    nonzero_times = sum(1 for c in lumps_payload["lump_counts"] if c > 0)
    max_lumps = max(lumps_payload["lump_counts"]) if lumps_payload["lump_counts"] else 0
    max_lump_size = max(
        (max(s) for s in lumps_payload["lump_sizes"] if s),
        default=0,
    )

    summary = {
        "run_root": run_root,
        "n_sites": n_sites,
        "dim": dim,
        "t_total": float(cfg.t_total),
        "n_steps": int(cfg.n_steps),
        "z_threshold": float(cfg.z_threshold),
        "total_lumps_over_time": int(total_lumps),
        "fraction_time_with_lumps": (
            float(nonzero_times) / float(cfg.n_steps) if cfg.n_steps > 0 else 0.0
        ),
        "max_lumps_per_step": int(max_lumps),
        "max_lump_size": int(max_lump_size),
        "snapshot_indices": [int(i) for i in cfg.snapshot_indices],
        "snapshot_every": int(cfg.snapshot_every),
    }
    with open(os.path.join(run_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Snapshots
    if snapshot_states:
        snap_indices = sorted(snapshot_states.keys())
        psi_list = [snapshot_states[k] for k in snap_indices]
        psi_array = np.stack(psi_list, axis=0)
        snap_path = os.path.join(run_root, "snapshots.npz")
        np.savez_compressed(
            snap_path,
            indices=np.array(snap_indices, dtype=int),
            psi=psi_array,
        )
        print(f"Saved {len(snap_indices)} snapshots to {snap_path}")
    else:
        print("No snapshots requested; snapshots.npz not written.")

    print("============================================================")
    print("  Precipitating Event snapshot run complete")
    print("============================================================")


if __name__ == "__main__":
    main()
