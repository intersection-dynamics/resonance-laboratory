"""
geometry_upscale_denoise.py

Upscale an emergent 3D geometry (few nodes) to a dense regular grid
and run a diffusion-based denoise over the resulting scalar field.

Workflow:
- Auto-select latest precipitating_event run under a root, optionally
  filtered by a tag substring (e.g. "flux_event_J2_intpaths").
- Load geometry npz:
    - graph_dist (N x N)
    - coords (N x 3) if present; otherwise use cube/circle fallback
- Load shapes npz from that run:
    - coords (N x 3)
    - mean_occ_all, mean_occ_internal
- Choose a scalar per node:
    - ones:            1.0 at each node
    - degree:          node degree in graph
    - mean_occ_all:    from shapes.npz
    - mean_occ_internal: from shapes.npz
- Embed those node values into a regular 3D grid (Nx x Ny x Nz)
  by mapping coords into [0, Nx-1]x[0,Ny-1]x[0,Nz-1] and accumulating.
- Run simple 3D diffusion ("heat equation"):
    phi_{t+1} = phi_t + dt * Laplacian(phi_t)
- Save:
    - upscaled_field.npz:
        * phi_initial (Nx, Ny, Nz)
        * phi_denoised (Nx, Ny, Nz)
    - central slices as PNGs:
        * slice_xy_initial.png / slice_xy_denoised.png (z midplane)
        * slice_xz_initial.png / slice_xz_denoised.png (y midplane)
        * slice_yz_initial.png / slice_yz_denoised.png (x midplane)
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------
# Run discovery (same style as your other analysis tools)
# --------------------------------------------------------------------


def find_run_roots(root: str) -> List[str]:
    run_roots: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "precipitating_event" not in dirpath.replace("\\", "/"):
            continue
        data_dir = os.path.join(dirpath, "data")
        ts_path = os.path.join(data_dir, "time_series.npz")
        if os.path.isfile(ts_path):
            run_roots.append(dirpath)
    return run_roots


def select_run_root(root: str, tag_substr: str | None = None) -> str:
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Root directory does not exist: {root}")

    candidates = find_run_roots(root)
    if not candidates:
        raise FileNotFoundError(
            f"No precipitating_event runs with time_series.npz found under {root}"
        )

    if tag_substr:
        tag_norm = tag_substr.replace("\\", "/")
        filtered: List[str] = []
        for c in candidates:
            if tag_norm in c.replace("\\", "/"):
                filtered.append(c)
        candidates = filtered
        if not candidates:
            raise FileNotFoundError(
                f"No runs under {root} contain tag substring '{tag_substr}'."
            )

    candidates_sorted = sorted(
        candidates,
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    return candidates_sorted[0]


# --------------------------------------------------------------------
# Geometry & shapes loading
# --------------------------------------------------------------------


def load_geometry(geometry_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(geometry_path):
        raise FileNotFoundError(f"Geometry file not found: {geometry_path}")
    npz = np.load(geometry_path)
    if "graph_dist" not in npz.files:
        raise ValueError("Geometry npz must contain 'graph_dist'.")
    graph_dist = np.array(npz["graph_dist"], dtype=float)
    n_sites = graph_dist.shape[0]

    if "coords" in npz.files:
        coords = np.array(npz["coords"], dtype=float)
        if coords.shape != (n_sites, 3):
            raise ValueError(
                f"coords has shape {coords.shape}, expected ({n_sites}, 3)."
            )
    else:
        # same fallback as our other tools
        if n_sites == 8:
            coords = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1],
                ],
                dtype=float,
            )
        else:
            theta = np.linspace(0, 2 * np.pi, n_sites, endpoint=False)
            coords = np.stack(
                [
                    np.cos(theta),
                    np.sin(theta),
                    np.zeros_like(theta),
                ],
                axis=1,
            )

    return {"graph_dist": graph_dist, "coords": coords}


def load_shapes(run_root: str) -> Dict[str, np.ndarray]:
    analysis_dir = os.path.join(run_root, "analysis_microstates")
    shapes_path = os.path.join(analysis_dir, "shapes.npz")
    if not os.path.exists(shapes_path):
        raise FileNotFoundError(
            f"shapes.npz not found at {shapes_path}. "
            "Run fermion_microstate_shapes.py first."
        )
    npz = np.load(shapes_path)
    out = {"coords": npz["coords"]}
    if "mean_occ_all" in npz.files:
        out["mean_occ_all"] = npz["mean_occ_all"]
    if "mean_occ_internal" in npz.files:
        out["mean_occ_internal"] = npz["mean_occ_internal"]
    return out


# --------------------------------------------------------------------
# Field construction & diffusion
# --------------------------------------------------------------------


def choose_node_field(
    field_source: str,
    geom: Dict[str, np.ndarray],
    shapes: Dict[str, np.ndarray],
) -> np.ndarray:
    graph_dist = geom["graph_dist"]
    n_sites = graph_dist.shape[0]

    field_source = field_source.lower()
    if field_source == "ones":
        return np.ones(n_sites, dtype=float)
    if field_source == "degree":
        # degree from 1-step adjacency
        adj = (graph_dist == 1).astype(int)
        np.fill_diagonal(adj, 0)
        deg = np.sum(adj, axis=1).astype(float)
        return deg

    if field_source == "mean_occ_all":
        if "mean_occ_all" not in shapes:
            raise ValueError("mean_occ_all not found in shapes.npz.")
        return np.array(shapes["mean_occ_all"], dtype=float)
    if field_source == "mean_occ_internal":
        if "mean_occ_internal" not in shapes:
            raise ValueError("mean_occ_internal not found in shapes.npz.")
        return np.array(shapes["mean_occ_internal"], dtype=float)

    raise ValueError(
        f"Unknown field_source '{field_source}'. "
        "Use one of: ones, degree, mean_occ_all, mean_occ_internal."
    )


def embed_to_grid(
    coords: np.ndarray,
    node_values: np.ndarray,
    Nx: int,
    Ny: int,
    Nz: int,
) -> np.ndarray:
    """
    Map node values into a regular 3D grid by nearest-cell splatting.
    """
    n_sites = coords.shape[0]
    if node_values.shape[0] != n_sites:
        raise ValueError("node_values length must match coords length.")

    # Normalize coords into [0,1] in each dimension
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = maxs - mins
    # avoid divide by zero if geometry is flat in some direction
    span[span == 0.0] = 1.0
    norm = (coords - mins) / span

    grid = np.zeros((Nx, Ny, Nz), dtype=float)
    count = np.zeros((Nx, Ny, Nz), dtype=float)

    for i in range(n_sites):
        x, y, z = norm[i]
        ix = int(round(x * (Nx - 1)))
        iy = int(round(y * (Ny - 1)))
        iz = int(round(z * (Nz - 1)))
        ix = max(0, min(Nx - 1, ix))
        iy = max(0, min(Ny - 1, iy))
        iz = max(0, min(Nz - 1, iz))

        grid[ix, iy, iz] += float(node_values[i])
        count[ix, iy, iz] += 1.0

    # Average where we deposited multiple nodes
    mask = count > 0
    grid_avg = np.zeros_like(grid)
    grid_avg[mask] = grid[mask] / count[mask]
    return grid_avg


def diffuse_3d(phi: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
    """
    Simple 3D diffusion (heat equation) with 6-point Laplacian and Neumann-like
    boundary (via np.roll wrapping, but we clamp by copying boundary after each step).
    """
    phi = phi.copy()
    Nx, Ny, Nz = phi.shape

    for step in range(n_steps):
        lap = (
            np.roll(phi, +1, axis=0)
            + np.roll(phi, -1, axis=0)
            + np.roll(phi, +1, axis=1)
            + np.roll(phi, -1, axis=1)
            + np.roll(phi, +1, axis=2)
            + np.roll(phi, -1, axis=2)
            - 6.0 * phi
        )
        phi = phi + dt * lap

        # Optional: clamp boundaries to their immediate neighbors to avoid
        # wrap-around artifacts (approx. Neumann BC).
        phi[0, :, :] = phi[1, :, :]
        phi[-1, :, :] = phi[-2, :, :]
        phi[:, 0, :] = phi[:, 1, :]
        phi[:, -1, :] = phi[:, -2, :]
        phi[:, :, 0] = phi[:, :, 1]
        phi[:, :, -1] = phi[:, :, -2]

    return phi


def save_slices(phi_initial: np.ndarray, phi_denoised: np.ndarray, out_dir: str) -> None:
    Nx, Ny, Nz = phi_initial.shape
    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2

    def _save_pair(
        slice_initial: np.ndarray,
        slice_denoised: np.ndarray,
        plane_name: str,
    ) -> None:
        vmin = min(slice_initial.min(), slice_denoised.min())
        vmax = max(slice_initial.max(), slice_denoised.max())

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        im0 = axes[0].imshow(
            slice_initial.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
        )
        axes[0].set_title(f"{plane_name} initial")
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        im1 = axes[1].imshow(
            slice_denoised.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
        )
        axes[1].set_title(f"{plane_name} denoised")
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        for ax in axes:
            ax.set_xlabel("i")
            ax.set_ylabel("j")

        plt.tight_layout()
        fname = os.path.join(out_dir, f"slice_{plane_name}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)

    # XY plane at mid z
    _save_pair(
        phi_initial[:, :, cz],
        phi_denoised[:, :, cz],
        "xy",
    )
    # XZ plane at mid y
    _save_pair(
        phi_initial[:, cy, :],
        phi_denoised[:, cy, :],
        "xz",
    )
    # YZ plane at mid x
    _save_pair(
        phi_initial[cx, :, :],
        phi_denoised[cx, :, :],
        "yz",
    )


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Upscale emergent geometry to a 3D grid and run a diffusion "
            "denoise over a scalar field defined on that grid."
        )
    )
    p.add_argument(
        "--root",
        type=str,
        required=True,
        help=(
            "Root directory to search for runs, e.g. 'outputs'. "
            "The script will pick the most recent precipitating_event run "
            "under this root (optionally filtered by --tag-substr)."
        ),
    )
    p.add_argument(
        "--tag-substr",
        type=str,
        default="",
        help=(
            "Optional substring to filter run paths by tag. "
            "For example 'flux_event_J2_intpaths' restricts to those runs."
        ),
    )
    p.add_argument(
        "--geometry",
        type=str,
        default="lr_embedding_3d.npz",
        help="Path to geometry npz (must contain graph_dist; coords optional).",
    )
    p.add_argument(
        "--field-source",
        type=str,
        default="mean_occ_internal",
        help=(
            "Node scalar field to use. One of: "
            "'ones', 'degree', 'mean_occ_all', 'mean_occ_internal'. "
            "The 'mean_occ_*' choices require shapes.npz in the run."
        ),
    )
    p.add_argument("--Nx", type=int, default=32, help="Grid size in x.")
    p.add_argument("--Ny", type=int, default=32, help="Grid size in y.")
    p.add_argument("--Nz", type=int, default=32, help="Grid size in z.")
    p.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="Number of diffusion steps.",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Diffusion time step.",
    )
    return p.parse_args()


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    root = args.root
    tag_substr = args.tag_substr.strip() or None
    geom_path = args.geometry
    field_source = args.field_source
    Nx, Ny, Nz = int(args.Nx), int(args.Ny), int(args.Nz)
    n_steps = int(args.n_steps)
    dt = float(args.dt)

    # Pick run root automatically
    run_root = select_run_root(root, tag_substr=tag_substr)
    print(f"Selected run_root: {os.path.abspath(run_root)}")

    # Where to drop outputs
    out_dir = os.path.join(run_root, "analysis_geometry_upscaled")
    os.makedirs(out_dir, exist_ok=True)

    # Load geometry & shapes
    geom = load_geometry(geom_path)
    shapes = load_shapes(run_root)

    coords_geom = geom["coords"]
    coords_shapes = shapes["coords"]
    if coords_geom.shape != coords_shapes.shape:
        print(
            "[WARN] coords in geometry and shapes differ; using geometry coords."
        )

    node_values = choose_node_field(field_source, geom, shapes)
    print(f"Using field_source='{field_source}' for node scalars.")

    # Embed into grid
    phi_initial = embed_to_grid(coords_geom, node_values, Nx, Ny, Nz)
    print(f"Embedded node field into grid of shape {phi_initial.shape}.")

    # Diffuse / denoise
    phi_denoised = diffuse_3d(phi_initial, n_steps=n_steps, dt=dt)
    print(f"Diffusion complete: n_steps={n_steps}, dt={dt}.")

    # Save volumes
    npz_out = os.path.join(out_dir, "upscaled_field.npz")
    np.savez_compressed(
        npz_out,
        phi_initial=phi_initial,
        phi_denoised=phi_denoised,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        field_source=field_source,
        n_steps=n_steps,
        dt=dt,
    )
    print(f"Saved upscaled fields to {npz_out}")

    # Save central slices
    save_slices(phi_initial, phi_denoised, out_dir)
    print("Saved central slice PNGs in", out_dir)


if __name__ == "__main__":
    main()
