"""
fermion_microstate_shapes.py

Post-analysis tool for the Precipitating Event runs.

Goal:
- Treat each time slice as a "microstate" of the excitation.
- Group microstates by:
    * dominant lump center site
    * binary occupancy pattern over sites
- Use the emergent geometry asset to project these patterns into 3D
  and see what "shapes" appear.

Inputs (from a precipitating_event run):
- <run_root>/data/time_series.npz
    * times
    * local_z_t
    * lump_counts
    * dominant_lump_sizes
    * parity_t
    * total_N_t
    * occupancy_t        (int, 0/1)
    * hamming_t          (int)
    * internal_change_t  (bool)
- <run_root>/data/dominant_lump_hist.json
    * list[time][list of site indices] for dominant clusters
- geometry npz file (e.g. lr_embedding_3d.npz)
    * must contain 'graph_dist'
    * may contain 'coords' with shape (n_sites, 3)

Outputs (into <run_root>/analysis_microstates/):
- microstates.json
    * per unique microstate:
        - center_site
        - pattern_bits (string, e.g. "11001010")
        - count (how many times it appears)
        - times (list of times where it appears, truncated for brevity)
- shapes.npz
    * coords: (n_sites, 3) node positions
    * mean_occ_all: (n_sites,) time-averaged occupancy over all times
    * mean_occ_internal: (n_sites,) time-averaged occupancy over
      internal-change timesteps (where the center is stable but pattern moves)
- Two PNGs:
    * shape_all.png
    * shape_internal.png

Run selection:
- You pass a ROOT directory (e.g. "outputs" or "outputs/precipitating_event").
- Script walks under it looking for precipitating_event run roots that contain:
    * data/time_series.npz
    * data/dominant_lump_hist.json
- If multiple are found:
    * optional --tag-substr filters to paths containing that substring
    * then the MOST RECENT (by directory mtime) is chosen.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass
class MicrostateRecord:
    center_site: int
    pattern_bits: str
    count: int
    times: List[float]


# ---------------------------------------------------------------------
# Run-root discovery
# ---------------------------------------------------------------------


def find_run_roots(root: str) -> List[str]:
    """
    Walk under `root` and find precipitating_event run roots:
    directories that contain data/time_series.npz and data/dominant_lump_hist.json.
    """
    run_roots: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Quick cheap filter: only look at paths that contain 'precipitating_event'
        if "precipitating_event" not in dirpath.replace("\\", "/"):
            continue
        data_dir = os.path.join(dirpath, "data")
        ts_path = os.path.join(data_dir, "time_series.npz")
        dom_path = os.path.join(data_dir, "dominant_lump_hist.json")
        if os.path.isfile(ts_path) and os.path.isfile(dom_path):
            run_roots.append(dirpath)
    return run_roots


def select_run_root(root: str, tag_substr: str | None = None) -> str:
    """
    Given a root directory (e.g. 'outputs' or 'outputs/precipitating_event'),
    find all valid run roots under it and select the most recent.
    Optionally filter by tag_substr (substring match in path).
    """
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Root directory does not exist: {root}")

    candidates = find_run_roots(root)
    if not candidates:
        raise FileNotFoundError(
            f"No precipitating_event runs with time_series.npz found under {root}"
        )

    if tag_substr:
        tag_substr_norm = tag_substr.replace("\\", "/")
        filtered: List[str] = []
        for c in candidates:
            if tag_substr_norm in c.replace("\\", "/"):
                filtered.append(c)
        candidates = filtered
        if not candidates:
            raise FileNotFoundError(
                f"No runs under {root} contain tag substring '{tag_substr}'."
            )

    # Pick the most recent by directory mtime
    candidates_sorted = sorted(
        candidates,
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    chosen = candidates_sorted[0]
    return chosen


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def load_time_series(run_root: str) -> Dict[str, Any]:
    data_dir = os.path.join(run_root, "data")
    npz_path = os.path.join(data_dir, "time_series.npz")
    dom_hist_path = os.path.join(data_dir, "dominant_lump_hist.json")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"time_series.npz not found at {npz_path}")
    if not os.path.exists(dom_hist_path):
        raise FileNotFoundError(f"dominant_lump_hist.json not found at {dom_hist_path}")

    npz = np.load(npz_path)
    with open(dom_hist_path, "r", encoding="utf-8") as f:
        dominant_lump_sites = json.load(f)

    out = {
        "times": npz["times"],
        "occupancy_t": npz["occupancy_t"],          # (T, n_sites) int 0/1
        "internal_change_t": npz["internal_change_t"],  # (T,) bool
        "dominant_lump_sites": dominant_lump_sites,
    }
    return out


def load_geometry(geometry_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(geometry_path):
        raise FileNotFoundError(f"Geometry file not found: {geometry_path}")

    npz = np.load(geometry_path)
    if "graph_dist" not in npz.files:
        raise ValueError("Geometry npz must contain 'graph_dist'.")

    graph_dist = np.array(npz["graph_dist"], dtype=float)
    n_sites = graph_dist.shape[0]

    # Try to get coords if present
    if "coords" in npz.files:
        coords = np.array(npz["coords"], dtype=float)
        if coords.shape != (n_sites, 3):
            raise ValueError(
                f"coords has shape {coords.shape}, expected ({n_sites}, 3)."
            )
    else:
        # Fallback: embed nodes in a cube for n_sites=8,
        # otherwise place them on a circle in 3D.
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


def pattern_from_row(row: np.ndarray) -> str:
    """
    Convert a 0/1 int row (shape (n_sites,)) to a bitstring, e.g. '11001010'.
    """
    bits = "".join(str(int(x)) for x in row.tolist())
    return bits


def build_microstates(
    times: np.ndarray,
    occupancy_t: np.ndarray,
    dominant_lump_sites: List[List[int]],
) -> List[MicrostateRecord]:
    """
    Build unique microstates keyed by (center_site, bitstring).
    center_site = first site of dominant lump; if no lump, center_site = -1.
    """
    T, n_sites = occupancy_t.shape
    if len(dominant_lump_sites) != T:
        raise ValueError(
            f"dominant_lump_sites length {len(dominant_lump_sites)} "
            f"does not match time dimension {T}"
        )

    store: Dict[Tuple[int, str], MicrostateRecord] = {}

    for t_idx in range(T):
        t = float(times[t_idx])
        occ_row = occupancy_t[t_idx, :]
        pattern_bits = pattern_from_row(occ_row)

        sites = dominant_lump_sites[t_idx]
        center_site = sites[0] if sites else -1

        key = (center_site, pattern_bits)
        if key not in store:
            store[key] = MicrostateRecord(
                center_site=center_site,
                pattern_bits=pattern_bits,
                count=0,
                times=[],
            )
        rec = store[key]
        rec.count += 1
        rec.times.append(t)

    # Sort for nicer output: by center_site, then by descending count
    microstates = list(store.values())
    microstates.sort(key=lambda r: (r.center_site, -r.count, r.pattern_bits))
    return microstates


def compute_shapes(
    occupancy_t: np.ndarray,
    internal_change_t: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute time-averaged occupancy for:
    - all times
    - internal-change times only
    """
    T, n_sites = occupancy_t.shape

    mean_occ_all = occupancy_t.mean(axis=0)

    internal_mask = internal_change_t.astype(bool)
    if np.any(internal_mask):
        occ_internal = occupancy_t[internal_mask, :]
        mean_occ_internal = occ_internal.mean(axis=0)
    else:
        mean_occ_internal = np.zeros(n_sites, dtype=float)

    return {
        "mean_occ_all": mean_occ_all,
        "mean_occ_internal": mean_occ_internal,
    }


def plot_shape(
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    out_path: str,
) -> None:
    """
    3D scatter of nodes, colored/sized by 'values'.
    """
    if coords.shape[0] != values.shape[0]:
        raise ValueError("coords and values must share first dimension.")

    # Normalize sizes for plotting
    v = values.astype(float)
    v_min = float(v.min())
    v_max = float(v.max())
    if v_max > v_min:
        v_norm = (v - v_min) / (v_max - v_min)
    else:
        v_norm = np.zeros_like(v)

    sizes = 50 + 250 * v_norm  # base size + scaled

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=sizes,
        c=v,
        cmap="viridis",
        depthshade=True,
    )

    for i, (x, y, z) in enumerate(coords):
        ax.text(x, y, z, str(i), fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="mean occupancy")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Analyze Precipitating Event microstates and project their "
            "pointer states into a 3D shape using the geometry asset."
        )
    )
    p.add_argument(
        "--root",
        type=str,
        required=True,
        help=(
            "Root directory to search for runs, e.g. 'outputs' or "
            "'outputs/precipitating_event'. The script will find the most "
            "recent precipitating_event run under this root automatically."
        ),
    )
    p.add_argument(
        "--tag-substr",
        type=str,
        default="",
        help=(
            "Optional substring to filter run paths by tag. For example, "
            "'flux_event_J2_intpaths' will restrict to runs whose path "
            "contains that substring, then pick the most recent."
        ),
    )
    p.add_argument(
        "--geometry",
        type=str,
        default="lr_embedding_3d.npz",
        help="Path to emergent geometry npz (must contain graph_dist; "
             "coords optional).",
    )
    p.add_argument(
        "--max-times-per-state",
        type=int,
        default=10,
        help="Maximum number of times to store per microstate in JSON "
             "(for readability).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = args.root
    tag_substr = args.tag_substr.strip() or None
    geom_path = args.geometry

    # -----------------------
    # Resolve run_root automatically
    # -----------------------
    run_root = select_run_root(root, tag_substr=tag_substr)
    print(f"Selected run_root: {os.path.abspath(run_root)}")

    # Where we'll store analysis outputs
    analysis_dir = os.path.join(run_root, "analysis_microstates")
    os.makedirs(analysis_dir, exist_ok=True)

    # -----------------------
    # Load data
    # -----------------------
    ts = load_time_series(run_root)
    times = ts["times"]
    occupancy_t = ts["occupancy_t"]
    internal_change_t = ts["internal_change_t"]
    dominant_lump_sites = ts["dominant_lump_sites"]

    geom = load_geometry(geom_path)
    coords = geom["coords"]

    T, n_sites = occupancy_t.shape
    if coords.shape[0] != n_sites:
        raise ValueError(
            f"coords has {coords.shape[0]} sites but occupancy_t has {n_sites}."
        )

    # -----------------------
    # Build microstates
    # -----------------------
    microstates = build_microstates(times, occupancy_t, dominant_lump_sites)

    micro_json_path = os.path.join(analysis_dir, "microstates.json")
    micro_json_serializable: List[Dict[str, Any]] = []
    for rec in microstates:
        micro_json_serializable.append(
            {
                "center_site": rec.center_site,
                "pattern_bits": rec.pattern_bits,
                "count": rec.count,
                "times": rec.times[: args.max_times_per_state],
            }
        )

    with open(micro_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_root": os.path.abspath(run_root),
                "geometry": os.path.abspath(geom_path),
                "n_sites": n_sites,
                "n_times": int(T),
                "n_unique_microstates": len(microstates),
                "microstates": micro_json_serializable,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    # -----------------------
    # Compute shapes
    # -----------------------
    shapes = compute_shapes(occupancy_t, internal_change_t)
    mean_occ_all = shapes["mean_occ_all"]
    mean_occ_internal = shapes["mean_occ_internal"]

    shapes_npz_path = os.path.join(analysis_dir, "shapes.npz")
    np.savez_compressed(
        shapes_npz_path,
        coords=coords,
        mean_occ_all=mean_occ_all,
        mean_occ_internal=mean_occ_internal,
    )

    # -----------------------
    # Plot shapes
    # -----------------------
    plot_shape(
        coords,
        mean_occ_all,
        title="Time-averaged occupancy (all times)",
        out_path=os.path.join(analysis_dir, "shape_all.png"),
    )

    if np.any(internal_change_t):
        plot_shape(
            coords,
            mean_occ_internal,
            title="Time-averaged occupancy (internal-change times only)",
            out_path=os.path.join(analysis_dir, "shape_internal.png"),
        )

    print("Analysis complete.")
    print(f"  Microstates JSON: {micro_json_path}")
    print(f"  Shapes NPZ:        {shapes_npz_path}")
    print(f"  Shape plots:       {os.path.join(analysis_dir, 'shape_all.png')}")
    if np.any(internal_change_t):
        print(f"                      {os.path.join(analysis_dir, 'shape_internal.png')}")


if __name__ == "__main__":
    main()
