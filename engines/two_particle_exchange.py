"""
two_particle_exchange.py

Adiabatic two-particle exchange experiment on the emergent geometry.

Philosophy:
- Only Hilbert space + constraints, no hand-planted "particles".
- We create two localized excitations by pinning two sites with local Z fields.
- A base Hamiltonian H0 encodes the substrate (Heisenberg+defrag, optionally
  including gauge phases on edges).
- We then define two time-dependent Hamiltonian paths:

    1) Trivial path: pinning wells stay fixed at the initial two sites.
    2) Exchange path: the wells move along user-specified site paths
       (path A and path B) so that their positions are exchanged.

- We start from the ground state of H_init = H0 + strong pinning at the
  initial wells, then evolve under each time-dependent Hamiltonian and
  compare the final states.

Goal:
- Measure the overlap <psi_trivial | psi_exchange>.
  The magnitude tells us how adiabatic we were;
  the phase tells us the exchange statistics / Berry phase.

Outputs:
- A run directory: <output_root>/two_particle_exchange/<timestamp>_<tag>/
  with:
    - logs/run.log
    - params.json, metadata.json, summary.json
    - data/time_series.npz:
        * times
        * local_z_exchange (T, n_sites)
        * local_z_trivial (T, n_sites)
    - data/z_paths.json:
        * paths of <Z_i> for each site and evolution
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional CUDA via CuPy
try:
    import cupy as cp  # type: ignore[attr-defined]
    HAS_CUPY = True
except Exception:  # noqa: BLE001
    cp = None  # type: ignore[assignment]
    HAS_CUPY = False


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclass
class GeometryAsset:
    graph_dist: np.ndarray
    extras: Dict[str, np.ndarray]


@dataclass
class ExchangeConfig:
    n_sites: int
    local_dim: int = 2

    J_coupling: float = 2.0
    h_field: float = 0.2
    defrag_zz: float = 2.0

    # Time-discretization for adiabatic evolution
    total_time: float = 20.0
    n_steps: int = 200

    # Bias wells
    bias_strength: float = 1.0  # amplitude of moving pinning field
    init_pin_strength: float = 4.0  # strong pinning for initial state

    # Paths (lists of site indices) for wells A and B
    path_A: List[int] = None  # type: ignore[assignment]
    path_B: List[int] = None  # type: ignore[assignment]


@dataclass
class RunLayout:
    run_root: str
    data_dir: str
    logs_dir: str
    figures_dir: str
    params_json: str
    metadata_json: str
    summary_json: str
    timeseries_npz: str
    z_paths_json: str
    log_path: str


# ---------------------------------------------------------------------
# Utilities: filesystem, logging
# ---------------------------------------------------------------------


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


def build_layout(output_root: str, tag: str | None = None) -> RunLayout:
    run_id = make_run_id(tag)
    run_root = os.path.join(output_root, "two_particle_exchange", run_id)

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
        timeseries_npz=os.path.join(data_dir, "time_series.npz"),
        z_paths_json=os.path.join(data_dir, "z_paths.json"),
        log_path=os.path.join(logs_dir, "run.log"),
    )


# ---------------------------------------------------------------------
# Geometry and adjacency
# ---------------------------------------------------------------------


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


def build_edge_list(adjacency: np.ndarray) -> List[Tuple[int, int]]:
    n = adjacency.shape[0]
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] != 0:
                edges.append((i, j))
    return edges


# ---------------------------------------------------------------------
# Pauli ops & Hamiltonian building
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


def build_hamiltonian_heisenberg_gauge(
    J: float,
    h: float,
    defrag_zz: float,
    adjacency: np.ndarray,
    local_ops: Dict[str, List[np.ndarray]],
    edges: List[Tuple[int, int]],
    edge_phases: Optional[Dict[Tuple[int, int], float]] = None,
) -> np.ndarray:
    """
    H0 with optional gauge phases on edges.

    For edge (i,j) with phase phi_ij:

      H_ij = J[ cos(phi) (X_i X_j + Y_i Y_j)
                + sin(phi) (X_i Y_j - Y_i X_j) ]
             + defrag_zz Z_i Z_j

    If edge_phases is None, phi_ij=0 and we get plain Heisenberg+ZZ.
    """
    Xs = local_ops["X"]
    Ys = local_ops["Y"]
    Zs = local_ops["Z"]

    n_sites = adjacency.shape[0]
    dim = 2 ** n_sites
    H = np.zeros((dim, dim), dtype=complex)

    use_gauge = edge_phases is not None

    for (i, j) in edges:
        XX = Xs[i] @ Xs[j]
        YY = Ys[i] @ Ys[j]
        XY = Xs[i] @ Ys[j]
        YX = Ys[i] @ Xs[j]
        ZZ = Zs[i] @ Zs[j]

        if use_gauge:
            key = (i, j) if i < j else (j, i)
            phi = float(edge_phases.get(key, 0.0))
            c = math.cos(phi)
            s = math.sin(phi)
            H += J * (c * (XX + YY) + s * (XY - YX))
        else:
            H += J * (XX + YY)
        H += defrag_zz * ZZ

    # local field term
    for i in range(n_sites):
        H += -h * Zs[i]

    H = 0.5 * (H + H.conj().T)
    return H


def add_local_bias(H: np.ndarray, Z_ops: List[np.ndarray], bias: np.ndarray) -> np.ndarray:
    """
    Add sum_i bias[i] * Z_i to Hamiltonian H and re-Hermitize.
    """
    H_b = H.copy()
    for i, b in enumerate(bias):
        if abs(b) > 0.0:
            H_b += float(b) * Z_ops[i]
    H_b = 0.5 * (H_b + H_b.conj().T)
    return H_b


# ---------------------------------------------------------------------
# CUDA-aware eigensolver & evolution
# ---------------------------------------------------------------------


def eigh_hermitian(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if HAS_CUPY:
        H_gpu = cp.asarray(H)  # type: ignore[call-arg]
        evals_gpu, evecs_gpu = cp.linalg.eigh(H_gpu)  # type: ignore[union-attr]
        evals = cp.asnumpy(evals_gpu)  # type: ignore[union-attr]
        evecs = cp.asnumpy(evecs_gpu)  # type: ignore[union-attr]
        return evals, evecs
    else:
        evals, evecs = np.linalg.eigh(H)
        return evals, evecs


def ground_state(H: np.ndarray) -> Tuple[float, np.ndarray]:
    evals, evecs = eigh_hermitian(H)
    idx = int(np.argmin(evals))
    E0 = float(evals[idx])
    psi0 = evecs[:, idx]
    return E0, psi0


def evolve_step(
    H: np.ndarray,
    psi: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Single time-step evolution under piecewise constant H: psi -> exp(-i H dt) psi.
    """
    evals, evecs = eigh_hermitian(H)
    alpha = evecs.conj().T @ psi
    phase = np.exp(-1j * evals * dt)
    alpha_t = phase * alpha
    psi_t = evecs @ alpha_t
    return psi_t


def z_expectations(psi: np.ndarray, Z_ops: List[np.ndarray]) -> np.ndarray:
    bra = psi.conj().T
    vals = []
    for Zi in Z_ops:
        vals.append(np.real_if_close(bra @ (Zi @ psi)))
    return np.array(vals, dtype=float)


# ---------------------------------------------------------------------
# Gauge phase loading
# ---------------------------------------------------------------------


def load_gauge_phases_from_run(run_root: str) -> Dict[Tuple[int, int], float]:
    """
    Load gauge phases from a precipitating_event run root that has
    data/gauge_phases.json with fields:
      - edges: list[[i,j], ...]
      - phases: list[float]
    """
    data_dir = os.path.join(run_root, "data")
    gpath = os.path.join(data_dir, "gauge_phases.json")
    if not os.path.exists(gpath):
        raise FileNotFoundError(
            f"gauge_phases.json not found at {gpath} "
            "(provide a precipitating_event run root with gauge enabled)."
        )
    with open(gpath, "r", encoding="utf-8") as f:
        payload = json.load(f)
    edges_list = payload["edges"]
    phases_list = payload["phases"]
    if len(edges_list) != len(phases_list):
        raise ValueError("edges and phases lengths mismatch in gauge_phases.json.")
    edge_phases: Dict[Tuple[int, int], float] = {}
    for (i, j), phi in zip(edges_list, phases_list):
        i_int = int(i)
        j_int = int(j)
        key = (i_int, j_int) if i_int < j_int else (j_int, i_int)
        edge_phases[key] = float(phi)
    return edge_phases


# ---------------------------------------------------------------------
# Paths & bias schedule
# ---------------------------------------------------------------------


def parse_path_string(s: str) -> List[int]:
    """
    Parse comma-separated list of integers, e.g. "0,1,3,2".
    """
    s = s.strip()
    if not s:
        return []
    parts = s.split(",")
    out: List[int] = []
    for p in parts:
        out.append(int(p.strip()))
    return out


def build_bias_schedule(
    cfg: ExchangeConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build bias schedules for trivial and exchange paths.

    Returns:
        bias_trivial (T, n_sites)
        bias_exchange (T, n_sites)
    where T = cfg.n_steps.
    """
    n_steps = cfg.n_steps
    n_sites = cfg.n_sites
    bias_strength = cfg.bias_strength

    if not cfg.path_A or not cfg.path_B:
        raise ValueError("path_A and path_B must be non-empty lists of site indices.")

    path_A = cfg.path_A
    path_B = cfg.path_B

    # Make sure paths start distinct
    if path_A[0] == path_B[0]:
        raise ValueError("path_A and path_B must start at different sites.")

    # Trivial: wells stay fixed at the initial sites
    bias_trivial = np.zeros((n_steps, n_sites), dtype=float)
    for t in range(n_steps):
        bias_trivial[t, path_A[0]] -= bias_strength
        bias_trivial[t, path_B[0]] -= bias_strength

    # Exchange path:
    # - We linearly step through path_A and path_B in parallel over n_steps.
    # - At the end, the wells have reached path_A[-1], path_B[-1].
    # It is up to the user to choose paths such that those endpoints are
    # exchanged positions in geometry.
    bias_exchange = np.zeros((n_steps, n_sites), dtype=float)

    def index_for_step(step: int, path: List[int]) -> int:
        # Map step 0..n_steps-1 onto indices 0..len(path)-1
        # using integer division.
        if len(path) == 1:
            return 0
        frac = step / max(1, n_steps - 1)
        idx = int(round(frac * (len(path) - 1)))
        idx = max(0, min(len(path) - 1, idx))
        return idx

    for t in range(n_steps):
        idxA = index_for_step(t, path_A)
        idxB = index_for_step(t, path_B)
        siteA = path_A[idxA]
        siteB = path_B[idxB]
        bias_exchange[t, siteA] -= bias_strength
        bias_exchange[t, siteB] -= bias_strength

    return bias_trivial, bias_exchange


# ---------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Two-particle exchange experiment on the emergent geometry, "
            "with time-dependent pinning wells."
        )
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
        help="Root directory for all outputs.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="exchange",
        help="Optional tag for the run ID.",
    )

    p.add_argument(
        "--gauge-run-root",
        type=str,
        default="",
        help=(
            "Optional: path to a precipitating_event run root containing "
            "data/gauge_phases.json. If given, we will import these phases "
            "into H0."
        ),
    )

    # Base Hamiltonian
    p.add_argument("--J-coupling", type=float, default=2.0)
    p.add_argument("--h-field", type=float, default=0.2)
    p.add_argument("--defrag-zz", type=float, default=2.0)

    # Exchange evolution
    p.add_argument("--total-time", type=float, default=20.0)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--bias-strength", type=float, default=1.0)
    p.add_argument("--init-pin-strength", type=float, default=4.0)

    p.add_argument(
        "--path-A",
        type=str,
        default="0,1,3,2",
        help="Comma-separated path for well A (e.g. '0,1,3,2').",
    )
    p.add_argument(
        "--path-B",
        type=str,
        default="4,5,7,6",
        help="Comma-separated path for well B (e.g. '4,5,7,6').",
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Layout & logging
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

    log("=" * 70)
    log("  Two-Particle Exchange Experiment (Hilbert substrate, CUDA-aware)")
    log("=" * 70)
    log(f"Run root:  {layout.run_root}")
    log(f"Geometry:  {args.geometry}")
    log(f"Gauge run: {args.gauge_run_root or '(none)'}")
    log(f"CuPy/CUDA: {'ENABLED' if HAS_CUPY else 'DISABLED'}")
    log("-" * 70)

    # Geometry
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
    log(f"Edges:           {edges}")
    log("-" * 70)

    # Exchange configuration
    cfg = ExchangeConfig(
        n_sites=n_sites,
        local_dim=2,
        J_coupling=float(args.J_coupling),
        h_field=float(args.h_field),
        defrag_zz=float(args.defrag_zz),
        total_time=float(args.total_time),
        n_steps=int(args.n_steps),
        bias_strength=float(args.bias_strength),
        init_pin_strength=float(args.init_pin_strength),
        path_A=parse_path_string(args.path_A),
        path_B=parse_path_string(args.path_B),
    )

    # Gauge phases (optional)
    edge_phases: Optional[Dict[Tuple[int, int], float]] = None
    if args.gauge_run_root.strip():
        gauge_run_root = args.gauge_run_root.strip()
        try:
            edge_phases = load_gauge_phases_from_run(gauge_run_root)
            log(f"Loaded gauge phases from {gauge_run_root}")
        except Exception as e:  # noqa: BLE001
            log(f"[WARN] Failed to load gauge phases: {e}")
            log("[WARN] Proceeding with plain Heisenberg+ZZ (no gauge).")
            edge_phases = None
    else:
        log("[GAUGE] No gauge-run-root provided; using plain Heisenberg+ZZ.")

    # Save params & metadata
    params_payload = {
        "timestamp": _dt.datetime.now().isoformat(),
        "exchange_config": asdict(cfg),
        "gauge_source": args.gauge_run_root or None,
    }
    write_json(layout.params_json, params_payload)

    metadata_payload = {
        "script": os.path.basename(__file__),
        "run_root": os.path.abspath(layout.run_root),
        "geometry_file": os.path.abspath(args.geometry),
        "gauge_run_root": (
            os.path.abspath(args.gauge_run_root) if args.gauge_run_root else None
        ),
        "args": vars(args),
        "extras_in_geometry": list(geom.extras.keys()),
    }
    write_json(layout.metadata_json, metadata_payload)

    # Base Hamiltonian H0
    H0 = build_hamiltonian_heisenberg_gauge(
        J=cfg.J_coupling,
        h=cfg.h_field,
        defrag_zz=cfg.defrag_zz,
        adjacency=adjacency,
        local_ops=local_ops,
        edges=edges,
        edge_phases=edge_phases,
    )
    Z_ops = local_ops["Z"]

    # Bias schedules
    try:
        bias_trivial, bias_exchange = build_bias_schedule(cfg)
    except Exception as e:  # noqa: BLE001
        log(f"[ERROR] Failed to build bias schedules: {e}")
        log_f.close()
        sys.exit(1)

    # Initial pinning for ground state:
    # Strong wells at the initial sites of path_A and path_B.
    init_bias = np.zeros(n_sites, dtype=float)
    init_bias[cfg.path_A[0]] -= cfg.init_pin_strength
    init_bias[cfg.path_B[0]] -= cfg.init_pin_strength
    H_init = add_local_bias(H0, Z_ops, init_bias)
    E_init, psi0 = ground_state(H_init)

    log(f"Initial pinned ground state energy E_init = {E_init:.6f}")
    log("-" * 70)

    # Time stepping
    n_steps = cfg.n_steps
    dt = cfg.total_time / float(max(1, n_steps - 1))
    times = np.linspace(0.0, cfg.total_time, n_steps)

    local_z_trivial = np.zeros((n_steps, n_sites), dtype=float)
    local_z_exchange = np.zeros((n_steps, n_sites), dtype=float)

    # Start with the same initial state for both paths
    psi_trivial = psi0.copy()
    psi_exchange = psi0.copy()

    # Record t=0 expectations
    local_z_trivial[0, :] = z_expectations(psi_trivial, Z_ops)
    local_z_exchange[0, :] = z_expectations(psi_exchange, Z_ops)

    log("Starting time evolution...")
    log(f"  total_time = {cfg.total_time}, n_steps = {n_steps}, dt = {dt:.4f}")
    log(f"  bias_strength = {cfg.bias_strength}, init_pin_strength = {cfg.init_pin_strength}")
    log("-" * 70)

    for k in range(1, n_steps):
        # Trivial path: fixed wells
        H_triv = add_local_bias(H0, Z_ops, bias_trivial[k, :])
        psi_trivial = evolve_step(H_triv, psi_trivial, dt)
        local_z_trivial[k, :] = z_expectations(psi_trivial, Z_ops)

        # Exchange path: moving wells
        H_exch = add_local_bias(H0, Z_ops, bias_exchange[k, :])
        psi_exchange = evolve_step(H_exch, psi_exchange, dt)
        local_z_exchange[k, :] = z_expectations(psi_exchange, Z_ops)

        if k % max(1, n_steps // 10) == 0 or k == n_steps - 1:
            log(
                f"[step {k:4d}/{n_steps-1}] t={times[k]:.3f} "
                f"Z_triv[0]={local_z_trivial[k, 0]:+.3f}, "
                f"Z_exch[0]={local_z_exchange[k, 0]:+.3f}"
            )

    log("-" * 70)
    log("Time evolution complete.")
    log("-" * 70)

    # Final overlap and phase
    overlap = np.vdot(psi_trivial, psi_exchange)
    mag = float(np.abs(overlap))
    phase = float(np.angle(overlap))

    log(f"Final overlap <psi_trivial|psi_exchange> = {overlap.real:+.6f}{overlap.imag:+.6f}i")
    log(f"|overlap| = {mag:.6f}")
    log(f"arg(overlap) [radians] = {phase:.6f}")
    log(f"arg(overlap) [degrees] = {math.degrees(phase):.3f}")
    log("-" * 70)

    # Save time-series
    np.savez_compressed(
        layout.timeseries_npz,
        times=times,
        local_z_trivial=local_z_trivial,
        local_z_exchange=local_z_exchange,
    )

    # Also save a JSON version of Z trajectories (for quick inspection)
    z_paths_payload = {
        "times": times.tolist(),
        "local_z_trivial": local_z_trivial.tolist(),
        "local_z_exchange": local_z_exchange.tolist(),
    }
    write_json(layout.z_paths_json, z_paths_payload)

    # Summary JSON
    summary_payload = {
        "timestamp": _dt.datetime.now().isoformat(),
        "exchange_config": asdict(cfg),
        "gauge_used": bool(edge_phases is not None),
        "gauge_source": args.gauge_run_root or None,
        "overlap": {
            "real": float(overlap.real),
            "imag": float(overlap.imag),
            "magnitude": mag,
            "phase_radians": phase,
            "phase_degrees": math.degrees(phase),
        },
    }
    write_json(layout.summary_json, summary_payload)

    log("==== Exchange Summary ====")
    log(f"Overlap magnitude: {mag:.6f}")
    log(f"Overlap phase:     {phase:.6f} rad  ({math.degrees(phase):.3f} deg)")
    log("======================================")

    log_f.close()


if __name__ == "__main__":
    main()
