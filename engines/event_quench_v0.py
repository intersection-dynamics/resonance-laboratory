#!/usr/bin/env python3
"""
event_quench_v0.py — Substrate v0.1 quench protocol

No lumps. No thresholds. No gauge.
Just:
  1. Graph → Hamiltonian (hot)
  2. Graph → Hamiltonian (cold)
  3. Evolve psi(t) step by step
  4. Save n_i(t) = (1 - <Z_i>)/2

Geometry NPZ requirements:
  - EITHER 'edges' (list of (i,j))
  - OR 'adj'   (NxN adjacency matrix)
  - OR 'graph_dist' (NxN, with graph_dist[i,j] == 1 for neighbors)
"""

import argparse
import os
import json
import numpy as np

from substrate_core import (
    random_state,
    build_hamiltonian,
    diagonalize,
    evolve,
    local_excitation,
)


def load_geometry_edges(path: str):
    """
    Load edges from a geometry npz.

    Priority:
      1) 'edges' (array of shape (E,2))
      2) 'adj'   (NxN adjacency matrix)
      3) 'graph_dist' (NxN, neighbors have distance ~1)
    """
    data = np.load(path, allow_pickle=True)
    files = set(data.files)

    # 1) explicit edge list
    if "edges" in files:
        raw = data["edges"]
        edges = [tuple(map(int, e)) for e in raw]
        return edges

    # 2) adjacency matrix
    if "adj" in files:
        adj = np.asarray(data["adj"])
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError("adj must be a square matrix.")
        N = adj.shape[0]
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i, j] != 0:
                    edges.append((i, j))
        return edges

    # 3) graph_dist: neighbors have distance 1
    if "graph_dist" in files:
        gd = np.asarray(data["graph_dist"], dtype=float)
        if gd.ndim != 2 or gd.shape[0] != gd.shape[1]:
            raise ValueError("graph_dist must be a square matrix.")
        N = gd.shape[0]
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if abs(gd[i, j] - 1.0) < 1e-9:
                    edges.append((i, j))
        if not edges:
            raise ValueError(
                "No edges inferred from graph_dist==1. "
                "Check that your geometry encodes nearest neighbors with distance 1."
            )
        return edges

    raise ValueError(
        f"Geometry NPZ {path} must contain 'edges', 'adj', or 'graph_dist'. "
        f"Found keys: {sorted(files)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Minimal hot→cold quench on a Hilbert substrate (v0.1)."
    )
    ap.add_argument("--geometry", required=True, help="Geometry npz (edges/adj/graph_dist).")
    ap.add_argument("--nsteps", type=int, default=200, help="Number of time steps.")
    ap.add_argument("--tmax", type=float, default=10.0, help="Total evolution time.")
    ap.add_argument("--tquench", type=float, default=4.0, help="Quench time.")
    ap.add_argument("--Jhot", type=float, default=1.0, help="J coupling (hot).")
    ap.add_argument("--Jcold", type=float, default=1.0, help="J coupling (cold).")
    ap.add_argument("--hhot", type=float, default=0.1, help="Z-field (hot).")
    ap.add_argument("--hcold", type=float, default=1.0, help="Z-field (cold).")
    ap.add_argument("--seed", type=int, default=117, help="Random seed for psi0.")
    ap.add_argument(
        "--outdir",
        type=str,
        default="quench_output",
        help="Output directory for timeseries.npz + metadata.",
    )
    args = ap.parse_args()

    geom_path = os.path.abspath(args.geometry)
    edges = load_geometry_edges(geom_path)
    N = max(max(i, j) for (i, j) in edges) + 1

    print("============================================================")
    print("  Substrate v0.1 — Hot→Cold Quench")
    print("============================================================")
    print(f"geometry:   {geom_path}")
    print(f"N sites:    {N}")
    print(f"#edges:     {len(edges)}")
    print(f"nsteps:     {args.nsteps}")
    print(f"tmax:       {args.tmax}")
    print(f"tquench:    {args.tquench}")
    print(f"J_hot, h_hot   = {args.Jhot}, {args.hhot}")
    print(f"J_cold, h_cold = {args.Jcold}, {args.hcold}")
    print("------------------------------------------------------------")

    # Build Hamiltonians
    H_hot = build_hamiltonian(N, edges, J=args.Jhot, h=args.hhot)
    H_cold = build_hamiltonian(N, edges, J=args.Jcold, h=args.hcold)

    evals_hot, evecs_hot = diagonalize(H_hot)
    evals_cold, evecs_cold = diagonalize(H_cold)

    # Initial state
    psi = random_state(N, seed=args.seed)

    # Time grid
    times = np.linspace(0.0, args.tmax, args.nsteps)
    dt = times[1] - times[0] if args.nsteps > 1 else args.tmax

    n_t = np.zeros((args.nsteps, N), dtype=float)

    # Evolve step-by-step with local H (no signaling)
    for k, t in enumerate(times):
        # Record observables at current state
        n_t[k, :] = local_excitation(psi, N)

        # Advance to next time (except final step)
        if k == args.nsteps - 1:
            break
        if t < args.tquench:
            psi = evolve(psi, dt, evals_hot, evecs_hot)
        else:
            psi = evolve(psi, dt, evals_cold, evecs_cold)

    os.makedirs(args.outdir, exist_ok=True)
    ts_path = os.path.join(args.outdir, "timeseries.npz")
    np.savez_compressed(ts_path, times=times, n_t=n_t)

    meta = {
        "geometry": geom_path,
        "N_sites": N,
        "n_edges": len(edges),
        "nsteps": int(args.nsteps),
        "tmax": float(args.tmax),
        "tquench": float(args.tquench),
        "Jhot": float(args.Jhot),
        "Jcold": float(args.Jcold),
        "hhot": float(args.hhot),
        "hcold": float(args.hcold),
        "seed": int(args.seed),
    }
    with open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {ts_path}")
    print(f"Wrote {os.path.join(args.outdir, 'meta.json')}")
    print("Done.")


if __name__ == "__main__":
    main()
