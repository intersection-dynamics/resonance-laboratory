#!/usr/bin/env python3
"""
fermion_exchange_diagnostic.py

Exchange experiment on emergent two-lump states from a
precipitating_event_snap run (or any run that writes compatible
lumps.json and snapshots.npz).

What this does:

  1. Load:
       - run_root/lumps.json
       - run_root/snapshots.npz
       - geometry NPZ (coords + edges or graph_dist)
  2. Choose a snapshot time step that has:
       - exactly 2 lumps
       - each with size >= --min-lump-size
       - centers separated by graph distance >= --min-center-distance
  3. Build a site-permutation that EXCHANGES either:
       - just the two lump centers   (exchange-mode = "center"), OR
       - the entire support of both lumps via a greedy geometric
         matching (exchange-mode = "full-lumps", requires equal
         lump sizes).
  4. Optionally apply a simple parity "dressing" operator W:
       dressing-mode = "none" (default)
         -> do nothing
       dressing-mode = "lump0-parity"
         -> multiply each basis state by (-1)^{(# of 1-bits on sites in lump 0)}
         before doing the exchange.
     Then implement the exchange in Hilbert space:
       psi_dressed = W |psi>
       psi_exch    = U_exchange psi_dressed
  5. Compute:
       overlap = <psi_dressed | psi_exch>
     and report:
       |overlap| and arg(overlap) (in radians and as a multiple of pi).

If |overlap| ≈ 1 and phase ≈ π (mod 2π), the state is approximately
antisymmetric under that exchange (fermion-like). If phase ≈ 0, it's
symmetric (boson-like). Anything in between is more mixed/anyonic.

Usage example (center exchange, no dressing):

  python fermion_exchange_diagnostic.py ^
    --run-root outputs\\precipitating_event_snap\\20251204_145839_hilbert_quench_12_snap_2lumps ^
    --geometry lr_embedding_3d_12.npz ^
    --min-lump-size 2 ^
    --min-center-distance 2

Full-lump exchange with parity dressing on lump 0:

  python fermion_exchange_diagnostic.py ^
    --run-root outputs\\precipitating_event_snap\\20251204_145839_hilbert_quench_12_snap_2lumps ^
    --geometry lr_embedding_3d_12.npz ^
    --min-lump-size 2 ^
    --min-center-distance 2 ^
    --exchange-mode full-lumps ^
    --dressing-mode lump0-parity
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------
# Geometry & data loading helpers
# ---------------------------------------------------------------------


def load_geometry(path: str) -> Dict[str, np.ndarray]:
    """
    Load geometry NPZ robustly.

    Supports:
      - coords + edges
      - X + graph_dist
      - X + edges

    Returns a dict with:
      - "coords": (n_sites, dim)
      - "edges": (n_edges, 2)
      - "graph_dist": (n_sites, n_sites) or None
    """
    data = np.load(path)
    keys = set(data.keys())

    # coords
    if "coords" in keys:
        coords = data["coords"]
    elif "X" in keys:
        coords = data["X"]
    else:
        raise ValueError(f"Geometry NPZ must contain 'coords' or 'X': {path}")

    # edges / graph_dist
    if "edges" in keys:
        edges = data["edges"].astype(int)
        graph_dist = data["graph_dist"] if "graph_dist" in keys else None
    elif "graph_dist" in keys:
        gd = data["graph_dist"]
        if gd.ndim != 2 or gd.shape[0] != gd.shape[1]:
            raise ValueError("graph_dist must be a square matrix.")
        n = gd.shape[0]
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(gd[i, j] - 1.0) < 1e-8:
                    edge_list.append((i, j))
        edges = np.array(edge_list, dtype=int)
        graph_dist = gd
    else:
        raise ValueError(f"Geometry NPZ must contain 'edges' or 'graph_dist': {path}")

    return {"coords": coords, "edges": edges, "graph_dist": graph_dist}


def bfs_graph_dist(n_sites: int, edges: np.ndarray) -> np.ndarray:
    """
    If there is no graph_dist in the geometry file, build one by BFS.
    dist[i,j] = shortest number of edges between i and j.
    """
    adj = [[] for _ in range(n_sites)]
    for i, j in edges:
        i = int(i)
        j = int(j)
        adj[i].append(j)
        adj[j].append(i)

    dist = np.full((n_sites, n_sites), np.inf, dtype=float)
    for src in range(n_sites):
        dist[src, src] = 0.0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            du = dist[src, u]
            for v in adj[u]:
                if dist[src, v] > du + 1.0:
                    dist[src, v] = du + 1.0
                    queue.append(v)
    return dist


def load_lumps(run_root: str) -> Dict:
    path = os.path.join(run_root, "lumps.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"lumps.json not found in run_root: {run_root}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_snapshots(run_root: str) -> Tuple[List[int], np.ndarray]:
    path = os.path.join(run_root, "snapshots.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"snapshots.npz not found in run_root: {run_root}")
    data = np.load(path)
    indices = [int(i) for i in data["indices"]]
    psi = data["psi"]
    return indices, psi


def center_of_lump(lump: List[int], coords: np.ndarray) -> int:
    """
    Define the "center" of a lump as the site in the lump whose
    coords are closest to the mean coordinates of the lump.
    """
    if len(lump) == 1:
        return int(lump[0])
    pts = coords[lump]  # shape (len(lump), dim)
    mean = pts.mean(axis=0)
    d2 = np.sum((pts - mean) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return int(lump[idx])


# ---------------------------------------------------------------------
# Exchange operators and dressing
# ---------------------------------------------------------------------


def build_site_permutation_for_exchange(
    n_sites: int,
    site_a: int,
    site_b: int,
) -> np.ndarray:
    """
    Build a permutation of site indices that swaps site_a and site_b
    and leaves all others fixed.

    Returns an array p of length n_sites such that:
      new_site_index = p[old_site_index]
    """
    p = np.arange(n_sites, dtype=int)
    p[site_a], p[site_b] = p[site_b], p[site_a]
    return p


def build_lump_exchange_permutation(
    n_sites: int,
    lump0: List[int],
    lump1: List[int],
    coords: np.ndarray,
) -> np.ndarray:
    """
    Build a site permutation that exchanges the ENTIRE support of
    two lumps.

    We require len(lump0) == len(lump1). For each site i in lump0,
    we greedily match to the closest unused site j in lump1 (in
    coordinate space). The permutation then swaps each matched pair:
      p[i] = j, p[j] = i
    and leaves all other sites fixed.
    """
    if len(lump0) != len(lump1):
        raise ValueError(
            f"Full-lump exchange requires equal-sized lumps, "
            f"got sizes {len(lump0)} and {len(lump1)}."
        )

    p = np.arange(n_sites, dtype=int)

    # Greedy bipartite matching based on coordinate distance
    remaining1 = set(int(x) for x in lump1)
    pairs: List[Tuple[int, int]] = []

    for i in lump0:
        i = int(i)
        if not remaining1:
            raise RuntimeError("Internal error: no remaining targets in lump1.")
        best_j = None
        best_d2 = None
        for j in remaining1:
            d2 = float(np.sum((coords[i] - coords[j]) ** 2))
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_j = j
        pairs.append((i, int(best_j)))
        remaining1.remove(int(best_j))

    for i, j in pairs:
        p[i] = j
        p[j] = i

    return p


def apply_site_permutation_to_basis_index(
    idx: int,
    site_perm: np.ndarray,
) -> int:
    """
    Given:
      - idx: integer label of a basis state in the spin-1/2 Hilbert space
      - site_perm: permutation of site indices, length n_sites

    Interpret idx as a bitstring b_0 ... b_{n-1} where bit i belongs to site i.
    Build new bitstring where the bit that used to live on site i now lives
    on site_perm[i].

    That is:
      new_bits[site_perm[i]] = old_bits[i]

    Return the integer index corresponding to this permuted bitstring.
    """
    n_sites = site_perm.shape[0]
    new_idx = 0
    for i in range(n_sites):
        bit = (idx >> i) & 1
        if bit:
            j = int(site_perm[i])
            new_idx |= (1 << j)
    return new_idx


def build_exchange_state(
    psi: np.ndarray,
    n_sites: int,
    site_perm: np.ndarray,
) -> np.ndarray:
    """
    Construct psi_exch = U_exchange psi, where U_exchange implements
    the site permutation given by site_perm.

    psi is 1D complex array of length 2**n_sites.
    """
    dim = psi.shape[0]
    psi_exch = np.empty_like(psi)
    for old_idx in range(dim):
        new_idx = apply_site_permutation_to_basis_index(old_idx, site_perm)
        psi_exch[new_idx] = psi[old_idx]
    return psi_exch


def apply_parity_dressing(
    psi: np.ndarray,
    n_sites: int,
    sites: List[int],
) -> np.ndarray:
    """
    Apply a simple Z2 "parity string" dressing:

      (W psi)[idx] = (-1)^{ (# of 1-bits on 'sites') } * psi[idx].

    Here we interpret each basis index as a bitstring over sites
    0..n_sites-1, consistent with the exchange permutation logic.
    """
    dim = psi.shape[0]
    out = np.empty_like(psi)
    sites_int = [int(s) for s in sites]

    for idx in range(dim):
        parity = 0
        for s in sites_int:
            parity ^= ((idx >> s) & 1)
        if parity:
            out[idx] = -psi[idx]
        else:
            out[idx] = psi[idx]
    return out


# ---------------------------------------------------------------------
# Main snapshot selection logic
# ---------------------------------------------------------------------


def pick_snapshot_step(
    lumps_data: Dict,
    snapshot_indices: List[int],
    coords: np.ndarray,
    graph_dist: np.ndarray,
    min_lump_size: int,
    min_center_distance: float,
    user_step: Optional[int] = None,
) -> Tuple[int, List[int], List[int], int, int, float]:
    """
    Choose a snapshot step for the exchange experiment.

    If user_step is not None:
      - Validate that:
          * user_step is in snapshot_indices
          * at that step, there are exactly 2 lumps, each size >= min_lump_size
          * centers are separated by at least min_center_distance
      - If not valid, raise.

    If user_step is None:
      - Iterate over snapshot_indices in order and pick the first
        that satisfies the same criteria.

    Returns:
      (step, lump0_sites, lump1_sites, center0, center1, center_distance)
    """
    times = lumps_data["times"]
    lump_counts = lumps_data["lump_counts"]
    lump_memberships = lumps_data["lump_memberships"]

    candidates = snapshot_indices if user_step is None else [user_step]

    for step in candidates:
        if step < 0 or step >= len(times):
            continue

        c = int(lump_counts[step])
        if c != 2:
            continue

        lumps = lump_memberships[step]
        if len(lumps) != 2:
            continue

        L0 = [int(v) for v in lumps[0]]
        L1 = [int(v) for v in lumps[1]]

        if len(L0) < min_lump_size or len(L1) < min_lump_size:
            continue

        c0 = center_of_lump(L0, coords)
        c1 = center_of_lump(L1, coords)
        d01 = float(graph_dist[c0, c1])

        if not np.isfinite(d01) or d01 < min_center_distance:
            continue

        return step, L0, L1, c0, c1, d01

    if user_step is not None:
        raise RuntimeError(
            f"User-specified snapshot-step {user_step} does not meet two-lump criteria."
        )
    else:
        raise RuntimeError(
            "No snapshot index meets the two-lump criteria. "
            "Try relaxing min_lump_size or min_center_distance, "
            "or saving snapshots at different times."
        )


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Exchange experiment on two emergent lumps from precipitating_event_snap "
            "snapshots (with optional full-lump exchange and parity dressing)."
        )
    )
    ap.add_argument(
        "--run-root",
        required=True,
        help="Run root directory (contains lumps.json, snapshots.npz).",
    )
    ap.add_argument(
        "--geometry",
        required=True,
        help="Geometry NPZ used for the run (e.g. lr_embedding_3d_12.npz).",
    )
    ap.add_argument(
        "--min-lump-size",
        type=int,
        default=2,
        help="Minimum size (number of sites) for each lump.",
    )
    ap.add_argument(
        "--min-center-distance",
        type=float,
        default=2.0,
        help="Minimum graph distance between lump centers.",
    )
    ap.add_argument(
        "--snapshot-step",
        type=int,
        default=None,
        help=(
            "Optional explicit snapshot step index to use "
            "(must be in snapshots.npz and meet two-lump criteria)."
        ),
    )
    ap.add_argument(
        "--exchange-mode",
        type=str,
        default="center",
        choices=["center", "full-lumps"],
        help=(
            "How to define the exchange operator: "
            "'center' = swap only lump centers; "
            "'full-lumps' = swap entire lumps by greedy geometric matching "
            "(requires equal-sized lumps)."
        ),
    )
    ap.add_argument(
        "--dressing-mode",
        type=str,
        default="none",
        choices=["none", "lump0-parity"],
        help=(
            "Optional parity dressing applied before exchange: "
            "'none' = no dressing; "
            "'lump0-parity' = multiply basis states by (-1)^(# of 1-bits "
            "on sites in lump 0)."
        ),
    )
    ap.add_argument(
        "--results-json",
        type=str,
        default=None,
        help=(
            "Optional path for JSON output with exchange details. "
            "Defaults to run_root/exchange_results.json."
        ),
    )
    args = ap.parse_args()

    run_root = os.path.abspath(args.run_root)
    geom_path = os.path.abspath(args.geometry)
    results_json_path = (
        os.path.abspath(args.results_json)
        if args.results_json is not None
        else os.path.join(run_root, "exchange_results.json")
    )

    print("============================================================")
    print("  Two-Particle Exchange from Snapshot (diagnostic)")
    print("============================================================")
    print(f"run_root:       {run_root}")
    print(f"geometry:       {geom_path}")
    print(f"exchange mode:  {args.exchange_mode}")
    print(f"dressing mode:  {args.dressing_mode}")
    print("------------------------------------------------------------")

    # Load geometry
    g = load_geometry(geom_path)
    coords = g["coords"]
    edges = g["edges"]
    graph_dist = g["graph_dist"]
    n_sites = coords.shape[0]

    if graph_dist is None:
        print("No graph_dist in geometry; building BFS-based distance matrix...")
        graph_dist = bfs_graph_dist(n_sites, edges)
    else:
        print("Using graph_dist from geometry file.")

    # Load lumps and snapshots
    lumps_data = load_lumps(run_root)
    snap_indices, psi_snap = load_snapshots(run_root)

    print(f"Available snapshot steps: {snap_indices}")
    print("------------------------------------------------------------")

    # Pick snapshot step
    user_step = args.snapshot_step
    if user_step is not None:
        print(f"User requested snapshot-step: {user_step}")

    step, L0, L1, c0, c1, d01 = pick_snapshot_step(
        lumps_data=lumps_data,
        snapshot_indices=snap_indices,
        coords=coords,
        graph_dist=graph_dist,
        min_lump_size=int(args.min_lump_size),
        min_center_distance=float(args.min_center_distance),
        user_step=user_step,
    )

    # Locate psi for that step
    if step not in snap_indices:
        raise RuntimeError(
            f"Chosen step {step} not found in snapshots.npz indices {snap_indices}."
        )
    snap_pos = snap_indices.index(step)
    psi_raw = psi_snap[snap_pos].astype(complex)

    print("Chosen snapshot:")
    print(f"  step:              {step}")
    print(f"  time:              {lumps_data['times'][step]:.6f}")
    print(f"  lump0 sites:       {L0}")
    print(f"  lump1 sites:       {L1}")
    print(f"  lump0 size:        {len(L0)}")
    print(f"  lump1 size:        {len(L1)}")
    print(f"  center0 site:      {c0}")
    print(f"  center1 site:      {c1}")
    print(f"  center dist:       {d01:.3f} (graph distance)")
    print("------------------------------------------------------------")

    # Build exchange permutation
    if args.exchange_mode == "center":
        print(f"Building exchange permutation by swapping centers {c0} <-> {c1}...")
        site_perm = build_site_permutation_for_exchange(n_sites, c0, c1)
    else:  # full-lumps
        print("Building full-lump exchange permutation (greedy geometric matching)...")
        if len(L0) != len(L1):
            raise RuntimeError(
                f"full-lumps exchange requires equal-sized lumps, "
                f"but sizes are {len(L0)} and {len(L1)}."
            )
        site_perm = build_lump_exchange_permutation(
            n_sites=n_sites,
            lump0=L0,
            lump1=L1,
            coords=coords,
        )

    print("Site permutation for exchange:")
    print(f"  perm: {site_perm.tolist()}")
    print("------------------------------------------------------------")

    # Sanity check on psi dimension
    dim_expected = 1 << n_sites
    if psi_raw.shape[0] != dim_expected:
        raise RuntimeError(
            f"psi has dimension {psi_raw.shape[0]}, but for n_sites={n_sites} "
            f"expected {dim_expected}."
        )

    # Optional parity dressing
    psi_dressed = psi_raw
    dressing_used = "none"
    if args.dressing_mode == "lump0-parity":
        print("Applying lump0-parity dressing: W = (-1)^(# of 1-bits on lump 0 sites).")
        psi_dressed = apply_parity_dressing(psi_raw, n_sites=n_sites, sites=L0)
        dressing_used = "lump0-parity"

    # Apply exchange in Hilbert space
    print("Applying exchange permutation in Hilbert space...")
    psi_exch = build_exchange_state(psi_dressed, n_sites, site_perm)

    # Compute overlap <psi_dressed | psi_exch>
    overlap = np.vdot(psi_dressed, psi_exch)  # conjugate(psi_dressed) * psi_exch
    mag = float(np.abs(overlap))
    phase = float(np.angle(overlap))  # in radians
    phase_over_pi = phase / np.pi

    print("------------------------------------------------------------")
    print("Exchange experiment result:")
    print(f"  <psi_dressed | Exchanged psi> = "
          f"{overlap.real:+.6f}{overlap.imag:+.6f}i")
    print(f"  |overlap|                    = {mag:.6f}")
    print(f"  phase (radians)              = {phase:.6f}")
    print(f"  phase / pi                   = {phase_over_pi:.6f}")
    print("------------------------------------------------------------")

    # Simple interpretation
    eps = 0.1  # loose tolerance
    if mag > 0.9:
        # State is almost invariant up to a phase
        if abs((phase_over_pi - 1.0) % 2.0) < eps:
            print(
                "Interpretation: approximately fermionic under this exchange "
                "(phase ≈ π mod 2π)."
            )
        elif abs((phase_over_pi) % 2.0) < eps:
            print(
                "Interpretation: approximately bosonic under this exchange "
                "(phase ≈ 0 mod 2π)."
            )
        else:
            print(
                "Interpretation: high overlap but with a nontrivial exchange "
                "phase (anyonic / mixed-like)."
            )
    else:
        print("Interpretation: large deformation under exchange (|overlap| << 1);")
        print("               the emergent state is not simply an eigenstate "
              "of the chosen exchange operator.")
    print("============================================================")

    # Write JSON results for later comparison / sweeps
    results_payload = {
        "run_root": run_root,
        "geometry": geom_path,
        "snapshot_step": int(step),
        "snapshot_time": float(lumps_data["times"][step]),
        "lump0_sites": [int(x) for x in L0],
        "lump1_sites": [int(x) for x in L1],
        "lump0_size": int(len(L0)),
        "lump1_size": int(len(L1)),
        "center0": int(c0),
        "center1": int(c1),
        "center_distance_graph": float(d01),
        "exchange_mode": str(args.exchange_mode),
        "dressing_mode": str(dressing_used),
        "site_permutation": [int(x) for x in site_perm.tolist()],
        "overlap_real": float(overlap.real),
        "overlap_imag": float(overlap.imag),
        "overlap_mag": mag,
        "overlap_phase": phase,
        "overlap_phase_over_pi": phase_over_pi,
    }

    try:
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results_payload, f, indent=2, sort_keys=True)
        print(f"Saved exchange results to: {results_json_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] Could not write results JSON: {exc}")


if __name__ == "__main__":
    main()
