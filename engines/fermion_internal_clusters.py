"""
fermion_internal_clusters.py

Dive deeper into internal microstates of a Precipitating Event run.

What it does:
- Automatically finds the most recent precipitating_event run under a root,
  optionally filtered by a tag substring (like "flux_event_J2_intpaths").
- Loads:
    * data/time_series.npz
        - times
        - occupancy_t (T, n_sites) int 0/1
        - internal_change_t (T,) bool
    * data/dominant_lump_hist.json
        - dominant_lump_sites[t] = list of site indices for the dominant lump
- Reconstructs microstates:
    * center_site = first site of dominant lump, or -1 if no lump
    * pattern_bits = occupancy bitstring, e.g. "01011001"
- Assigns each unique (center_site, pattern_bits) a state ID.
- Builds:
    1) Per-center-site clusters in occupancy space using Hamming distance
       (greedy clustering with configurable radius).
    2) A transition count matrix over microstates: state_t -> state_{t+1}.

Outputs inside <run_root>/analysis_microstates/:
- internal_clusters.json
    {
      "run_root": ...,
      "n_sites": int,
      "n_times": int,
      "n_states": int,
      "cluster_radius": int,
      "center_sites": {
        "<center_site>": {
          "total_occurrences": int,
          "clusters": [
            {
              "cluster_id": int,
              "representative_pattern": "01011001",
              "total_count": int,
              "fraction_of_center_site": float,
              "fraction_of_all_times": float,
              "fraction_internal_change": float,
              "members": [
                {
                  "pattern_bits": "01011001",
                  "state_id": int,
                  "count": int
                },
                ...
              ]
            },
            ...
          ]
        },
        ...
      },
      "transitions": {
        "states": [
          { "state_id": int,
            "center_site": int,
            "pattern_bits": "01011001",
            "count": int },
          ...
        ],
        "edges": [
          {
            "from_state": int,
            "to_state": int,
            "count": int
          },
          ...
        ]
      }
    }

This gives you:
- Internal "basis states" (clusters) of the excitation per center site.
- A transition graph of how the system moves through these internal states.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass
class Microstate:
    center_site: int
    pattern_bits: str
    count: int = 0
    internal_change_count: int = 0


# ---------------------------------------------------------------------
# Run-root discovery (same style as fermion_microstate_shapes)
# ---------------------------------------------------------------------


def find_run_roots(root: str) -> List[str]:
    """
    Walk under `root` and find precipitating_event run roots:
    directories that contain data/time_series.npz and data/dominant_lump_hist.json.
    """
    run_roots: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
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
    chosen = candidates_sorted[0]
    return chosen


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def load_time_series_and_centers(run_root: str) -> Dict[str, Any]:
    data_dir = os.path.join(run_root, "data")
    npz_path = os.path.join(data_dir, "time_series.npz")
    dom_path = os.path.join(data_dir, "dominant_lump_hist.json")

    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"time_series.npz not found at {npz_path}")
    if not os.path.isfile(dom_path):
        raise FileNotFoundError(f"dominant_lump_hist.json not found at {dom_path}")

    npz = np.load(npz_path)
    times = npz["times"]
    occupancy_t = npz["occupancy_t"]            # (T, n_sites) int 0/1
    internal_change_t = npz["internal_change_t"]  # (T,) bool

    with open(dom_path, "r", encoding="utf-8") as f:
        dominant_lump_sites = json.load(f)

    if len(dominant_lump_sites) != occupancy_t.shape[0]:
        raise ValueError(
            "dominant_lump_hist length does not match time dimension in occupancy_t."
        )

    # Compute center_site per time: first site of dominant lump or -1 if no lump
    T = occupancy_t.shape[0]
    centers = np.full(T, -1, dtype=int)
    for t in range(T):
        sites = dominant_lump_sites[t]
        if sites:
            centers[t] = int(sites[0])

    return {
        "times": times,
        "occupancy_t": occupancy_t,
        "internal_change_t": internal_change_t,
        "centers": centers,
    }


def pattern_from_row(row: np.ndarray) -> str:
    """
    Convert a 0/1 int row (shape (n_sites,)) to a bitstring, e.g. '11001010'.
    """
    return "".join(str(int(x)) for x in row.tolist())


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("Bitstrings must have same length for Hamming distance.")
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


# ---------------------------------------------------------------------
# Microstate construction
# ---------------------------------------------------------------------


def build_microstates(
    occupancy_t: np.ndarray,
    centers: np.ndarray,
    internal_change_t: np.ndarray,
) -> Tuple[Dict[Tuple[int, str], Microstate], List[int]]:
    """
    Build microstates and map each time index to a state_id.

    Returns:
        microstate_map: dict keyed by (center_site, pattern_bits) -> Microstate
        state_ids: list of state_id[t] for each time t
    """
    T, n_sites = occupancy_t.shape
    if centers.shape[0] != T or internal_change_t.shape[0] != T:
        raise ValueError("Mismatched time dimension in inputs.")

    microstate_map: Dict[Tuple[int, str], Microstate] = {}
    state_ids: List[int] = [-1] * T

    # We'll assign state IDs in order of first appearance
    id_lookup: Dict[Tuple[int, str], int] = {}
    next_id = 0

    for t in range(T):
        center_site = int(centers[t])
        pattern_bits = pattern_from_row(occupancy_t[t, :])
        key = (center_site, pattern_bits)

        if key not in microstate_map:
            microstate_map[key] = Microstate(
                center_site=center_site,
                pattern_bits=pattern_bits,
                count=0,
                internal_change_count=0,
            )
            id_lookup[key] = next_id
            next_id += 1

        rec = microstate_map[key]
        rec.count += 1
        if bool(internal_change_t[t]):
            rec.internal_change_count += 1

        state_ids[t] = id_lookup[key]

    return microstate_map, state_ids


# ---------------------------------------------------------------------
# Clustering per center site
# ---------------------------------------------------------------------


def cluster_microstates_for_center(
    states_for_center: List[Microstate],
    radius: int,
) -> List[Dict[str, Any]]:
    """
    Greedy clustering in Hamming space.

    states_for_center: list of Microstate with same center_site
    radius: max Hamming distance within a cluster

    Returns: list of cluster dicts with:
        - cluster_id
        - representative_pattern
        - total_count
        - fraction_of_center_site
        - fraction_of_all_times (to be filled by caller)
        - fraction_internal_change
        - members: [ {pattern_bits, state_id, count, internal_change_count}, ... ]
    """
    # We'll work on a copy so we can sort
    states_sorted = sorted(
        states_for_center, key=lambda s: -s.count
    )  # most frequent first

    used: Dict[str, bool] = {}
    clusters: List[Dict[str, Any]] = []

    # Precompute total count for this center
    total_count_center = sum(s.count for s in states_sorted)
    if total_count_center == 0:
        return []

    cluster_id = 0

    for s in states_sorted:
        if used.get(s.pattern_bits, False):
            continue

        # This state seeds a new cluster
        rep = s.pattern_bits
        members = []
        total_count = 0
        total_internal_change = 0

        for s2 in states_sorted:
            if used.get(s2.pattern_bits, False):
                continue
            d = hamming_distance(rep, s2.pattern_bits)
            if d <= radius:
                used[s2.pattern_bits] = True
                members.append(s2)
                total_count += s2.count
                total_internal_change += s2.internal_change_count

        if total_count == 0:
            continue

        frac_center = total_count / float(total_count_center)
        frac_internal = (
            total_internal_change / float(total_count)
            if total_count > 0
            else 0.0
        )

        cluster_dict = {
            "cluster_id": cluster_id,
            "representative_pattern": rep,
            "total_count": int(total_count),
            "fraction_of_center_site": float(frac_center),
            # fraction_of_all_times will be filled in by caller
            "fraction_of_all_times": None,
            "fraction_internal_change": float(frac_internal),
            "members": [
                {
                    "pattern_bits": m.pattern_bits,
                    "state_id": None,  # to be patched by caller
                    "count": int(m.count),
                    "internal_change_count": int(m.internal_change_count),
                }
                for m in members
            ],
        }
        clusters.append(cluster_dict)
        cluster_id += 1

    return clusters


# ---------------------------------------------------------------------
# Transition graph
# ---------------------------------------------------------------------


def build_transition_edges(state_ids: List[int]) -> List[Dict[str, int]]:
    """
    Build sparse list of transitions "from_state" -> "to_state" with counts.
    """
    T = len(state_ids)
    if T < 2:
        return []

    trans_counts: Dict[Tuple[int, int], int] = {}

    for t in range(T - 1):
        a = int(state_ids[t])
        b = int(state_ids[t + 1])
        key = (a, b)
        trans_counts[key] = trans_counts.get(key, 0) + 1

    edges = []
    for (a, b), c in trans_counts.items():
        edges.append(
            {"from_state": int(a), "to_state": int(b), "count": int(c)}
        )

    # Sort for a bit of readability
    edges.sort(key=lambda e: (-e["count"], e["from_state"], e["to_state"]))
    return edges


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cluster internal microstates of a Precipitating Event run and "
            "build a transition graph over them."
        )
    )
    p.add_argument(
        "--root",
        type=str,
        required=True,
        help=(
            "Root directory to search for runs, e.g. 'outputs' or "
            "'outputs/precipitating_event'. The script finds the most "
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
        "--cluster-radius",
        type=int,
        default=2,
        help=(
            "Maximum Hamming distance within a cluster (default: 2). "
            "Smaller = more, tighter clusters; larger = fewer, looser clusters."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    root = args.root
    tag_substr = args.tag_substr.strip() or None
    radius = int(args.cluster_radius)

    # Select run root
    run_root = select_run_root(root, tag_substr=tag_substr)
    print(f"Selected run_root: {os.path.abspath(run_root)}")

    analysis_dir = os.path.join(run_root, "analysis_microstates")
    os.makedirs(analysis_dir, exist_ok=True)

    ts = load_time_series_and_centers(run_root)
    occupancy_t = ts["occupancy_t"]
    centers = ts["centers"]
    internal_change_t = ts["internal_change_t"]
    times = ts["times"]

    T, n_sites = occupancy_t.shape
    print(f"n_times = {T}, n_sites = {n_sites}")

    # Build microstates and state IDs
    microstate_map, state_ids = build_microstates(
        occupancy_t=occupancy_t,
        centers=centers,
        internal_change_t=internal_change_t,
    )
    n_states = len(microstate_map)
    print(f"n_unique_microstates = {n_states}")

    # Index microstates by state_id
    # Recall: state_ids were assigned in order of first appearance
    # We reconstruct that mapping
    #   key -> Microstate, but we also need key -> state_id
    key_to_id: Dict[Tuple[int, str], int] = {}
    next_id = 0
    for t in range(T):
        center_site = int(centers[t])
        pattern_bits = pattern_from_row(occupancy_t[t, :])
        key = (center_site, pattern_bits)
        if key not in key_to_id:
            key_to_id[key] = next_id
            next_id += 1

    # Sanity check
    if next_id != n_states:
        print(
            "[WARN] Internal state_id counting mismatch: "
            f"next_id={next_id}, n_states={n_states}"
        )

    # Organize microstates by center_site
    by_center: Dict[int, List[Microstate]] = {}
    total_occurrences_all = 0
    for key, ms in microstate_map.items():
        c_site, pattern_bits = key
        if c_site not in by_center:
            by_center[c_site] = []
        by_center[c_site].append(ms)
        total_occurrences_all += ms.count

    # Cluster per center_site
    center_sites_sorted = sorted(by_center.keys())
    center_clusters: Dict[str, Any] = {}

    for c_site in center_sites_sorted:
        states_for_center = by_center[c_site]
        clusters = cluster_microstates_for_center(
            states_for_center=states_for_center,
            radius=radius,
        )

        # Fill in fraction_of_all_times and patch state_ids for members
        for cl in clusters:
            total_count = cl["total_count"]
            cl["fraction_of_all_times"] = (
                total_count / float(total_occurrences_all)
                if total_occurrences_all > 0
                else 0.0
            )
            # patch member state_ids
            for mem in cl["members"]:
                key = (c_site, mem["pattern_bits"])
                mem["state_id"] = int(key_to_id.get(key, -1))

        total_count_center = sum(ms.count for ms in states_for_center)
        center_clusters[str(c_site)] = {
            "center_site": c_site,
            "total_occurrences": int(total_count_center),
            "clusters": clusters,
        }

    # Build transition edges
    edges = build_transition_edges(state_ids)

    # Build state list metadata
    states_meta: List[Dict[str, Any]] = []
    for (c_site, pattern_bits), ms in microstate_map.items():
        sid = key_to_id[(c_site, pattern_bits)]
        states_meta.append(
            {
                "state_id": int(sid),
                "center_site": int(c_site),
                "pattern_bits": pattern_bits,
                "count": int(ms.count),
                "internal_change_count": int(ms.internal_change_count),
            }
        )

    states_meta.sort(key=lambda s: s["state_id"])

    # Assemble payload
    payload = {
        "run_root": os.path.abspath(run_root),
        "n_sites": int(n_sites),
        "n_times": int(T),
        "n_states": int(n_states),
        "cluster_radius": int(radius),
        "center_sites": center_clusters,
        "transitions": {
            "states": states_meta,
            "edges": edges,
        },
    }

    out_path = os.path.join(analysis_dir, "internal_clusters.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print("Internal microstate clustering complete.")
    print(f"  Output JSON: {out_path}")


if __name__ == "__main__":
    main()
