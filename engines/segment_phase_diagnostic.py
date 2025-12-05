#!/usr/bin/env python
"""
segment_phase_diagnostic.py

Track the phase history of a coarse-grained segment (a lump or a single site)
in a precipitating_event run.

Inputs (in run_root):
  - timeseries.npz : times, occupancy_t, internal_change_t, ...
  - lumps.json     : times, lump_counts, lump_memberships (per t index)
  - snapshots.npz  : indices, psi (subset of time steps with full |psi>)

What this script does:

  1. Load timeseries + (optionally) lumps + snapshots.

  2. Choose a reference segment S:

     a) First try to find a lump of size >= min_segment_size at one of the
        snapshot steps. If found, pick the LARGEST such lump.

     b) Failing that, scan ALL time steps for such a lump and use its sites.

     c) If no lumps of the requested size exist anywhere, fall back to a
        purely data-driven choice: pick the single site whose occupancy_t
        has the largest variance over time and define S = {that_site}.

  3. For each snapshot step k in snapshots.npz:
       - Read coarse occupancy pattern on S: occ_S(k) from occupancy_t[k, S].
       - Build the corresponding branch amplitude:

           A_S(k) = sum_{basis states with those bits on S} psi_k

       - Record |A_S(k)|^2 and arg(A_S(k)).
       - Look up internal_change_t[k] (from precipitating_event diagnostics).

  4. Print a table:

       step, time, pattern_on_S, |A|^2, phase, Δphase, internal_change

This is a direct probe of "information-history phase bookkeeping" for a
chosen segment: how the phase of the coarse-grained pattern on S evolves
as the substrate reconfigures.

The fallback to a single high-variance site ensures you always get a
diagnostic, even if the lump finder never reports any lumps at snapshot
times for the chosen z_threshold.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def load_timeseries(run_root: str) -> Dict[str, np.ndarray]:
    path = os.path.join(run_root, "timeseries.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"timeseries.npz not found in run_root: {run_root}")
    data = np.load(path)
    required = ["times", "occupancy_t", "internal_change_t"]
    for key in required:
        if key not in data:
            raise KeyError(f"{key} not found in timeseries.npz at {path}")
    return {
        "times": data["times"],
        "occupancy_t": data["occupancy_t"],
        "internal_change_t": data["internal_change_t"],
        # optional extras if you want them later
        "lump_counts": data["lump_counts"] if "lump_counts" in data else None,
        "dominant_lump_sizes": data["dominant_lump_sizes"]
        if "dominant_lump_sizes" in data
        else None,
        "parity_t": data["parity_t"] if "parity_t" in data else None,
        "total_N_t": data["total_N_t"] if "total_N_t" in data else None,
        "local_z_t": data["local_z_t"] if "local_z_t" in data else None,
        "hamming_t": data["hamming_t"] if "hamming_t" in data else None,
    }


def load_lumps(run_root: str) -> Optional[Dict]:
    path = os.path.join(run_root, "lumps.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_snapshots(run_root: str) -> Tuple[List[int], np.ndarray]:
    path = os.path.join(run_root, "snapshots.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"snapshots.npz not found in run_root: {run_root}")
    data = np.load(path)
    if "indices" not in data or "psi" not in data:
        raise KeyError(
            f"snapshots.npz at {path} must contain 'indices' and 'psi' arrays."
        )
    indices = [int(i) for i in data["indices"]]
    psi = data["psi"].astype(complex)
    return indices, psi


# ---------------------------------------------------------------------
# Segment selection
# ---------------------------------------------------------------------


def choose_segment_from_lumps(
    lumps_data: Dict,
    n_steps: int,
    min_size: int,
    snapshot_indices: Optional[List[int]] = None,
) -> Tuple[int, List[int]]:
    """
    Choose a reference segment S using the lump data.

    Strategy:
      1) If snapshot_indices is provided and nonempty:
           Scan only those steps first, looking for any lump of size >= min_size.
           If found, pick the LARGEST lump at the first such step.
      2) If not found, scan ALL timesteps 0..n_steps-1 similarly.
      3) If STILL not found, raise RuntimeError.

    Returns:
      (step_index, list_of_sites_in_segment)
    """
    times = lumps_data["times"]
    lump_counts = lumps_data["lump_counts"]
    lump_memberships = lumps_data["lump_memberships"]

    # 1) Prefer snapshot steps if given
    if snapshot_indices:
        for step in sorted(snapshot_indices):
            if step < 0 or step >= n_steps:
                continue
            c = int(lump_counts[step])
            if c <= 0:
                continue
            lumps_t = lump_memberships[step]
            if not lumps_t:
                continue
            # find largest lump satisfying min_size
            sorted_lumps = sorted(lumps_t, key=lambda L: len(L), reverse=True)
            for L in sorted_lumps:
                if len(L) >= min_size:
                    best = [int(v) for v in L]
                    return step, best

    # 2) Otherwise scan all timesteps
    for step in range(n_steps):
        c = int(lump_counts[step])
        if c <= 0:
            continue
        lumps_t = lump_memberships[step]
        if not lumps_t:
            continue
        sorted_lumps = sorted(lumps_t, key=lambda L: len(L), reverse=True)
        for L in sorted_lumps:
            if len(L) >= min_size:
                best = [int(v) for v in L]
                return step, best

    # 3) No luck
    raise RuntimeError(
        "No lumps with size >= min_size exist at any timestep according to lumps.json."
    )


def choose_fallback_segment(
    occupancy_t: np.ndarray,
) -> Tuple[int, List[int]]:
    """
    Fallback segment choice based only on occupancy data:

      - Compute per-site variance of occupancy_t[:, i].
      - Choose the site with maximum variance as S = {i_*}.
      - Use step 0 as a nominal reference step.

    Returns:
      (ref_step, segment_sites)
    """
    n_steps, n_sites = occupancy_t.shape
    var_per_site = occupancy_t.astype(float).var(axis=0)
    best_site = int(np.argmax(var_per_site))
    ref_step = 0
    return ref_step, [best_site]


# ---------------------------------------------------------------------
# Branch amplitude extraction
# ---------------------------------------------------------------------


def build_segment_mask_and_target(
    segment_sites: List[int], pattern_bits: np.ndarray
) -> Tuple[int, int]:
    """
    Given:
      - segment_sites: list of site indices in S
      - pattern_bits: array of 0/1 occupancy bits on S (same length)

    Build:
      - mask:   bitmask that selects bits on S
      - target: bitmask that encodes the required 0/1 bits on S

    Convention: bit i in the integer basis index corresponds to site i.
    """
    if len(segment_sites) != len(pattern_bits):
        raise ValueError("segment_sites and pattern_bits must have same length.")

    mask = 0
    target = 0
    for site, bit in zip(segment_sites, pattern_bits):
        site = int(site)
        b = int(bit)
        mask |= (1 << site)
        if b == 1:
            target |= (1 << site)
    return mask, target


def segment_branch_amplitude(
    psi: np.ndarray, n_sites: int, segment_sites: List[int], pattern_bits: np.ndarray
) -> complex:
    """
    Compute the branch amplitude A_S:

      A_S = sum_{basis states with given bits on S} psi[idx]

    where the bits on S are specified by pattern_bits (0/1 array).
    """
    dim_expected = 1 << n_sites
    if psi.shape[0] != dim_expected:
        raise ValueError(
            f"psi has dimension {psi.shape[0]}, but expected 2**n_sites={dim_expected}"
        )

    mask, target = build_segment_mask_and_target(segment_sites, pattern_bits)

    amp = 0.0 + 0.0j
    # brute force over basis states (fine for n_sites ~ 12)
    for idx in range(dim_expected):
        if (idx & mask) == target:
            amp += psi[idx]
    return amp


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Phase-history diagnostic on a segment (lump or high-variance site) "
            "in a precipitating_event run."
        )
    )
    ap.add_argument(
        "--run-root",
        required=True,
        help="Run root directory for precipitating_event (timeseries.npz, lumps.json, snapshots.npz).",
    )
    ap.add_argument(
        "--min-segment-size",
        type=int,
        default=2,
        help="Minimum size of the reference lump used to define the segment. "
             "If no such lump exists, a 1-site high-variance fallback is used.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra details about the chosen segment.",
    )
    args = ap.parse_args()

    run_root = os.path.abspath(args.run_root)

    print("============================================================")
    print("  Segment Phase-History Diagnostic")
    print("============================================================")
    print(f"run_root:       {run_root}")
    print("------------------------------------------------------------")

    # Load data
    ts = load_timeseries(run_root)
    lumps_data = load_lumps(run_root)
    snap_indices, psi_snap = load_snapshots(run_root)

    times = np.asarray(ts["times"])
    occupancy_t = np.asarray(ts["occupancy_t"])
    internal_change_t = np.asarray(ts["internal_change_t"]).astype(bool)

    n_steps, n_sites = occupancy_t.shape

    print(f"n_steps (timeseries): {n_steps}")
    print(f"n_sites:              {n_sites}")
    print(f"snapshot indices:     {snap_indices}")
    print("------------------------------------------------------------")

    # Choose reference segment S
    segment_chosen_from_lumps = False
    try:
        if lumps_data is not None:
            ref_step, segment_sites = choose_segment_from_lumps(
                lumps_data=lumps_data,
                n_steps=n_steps,
                min_size=int(args.min_segment_size),
                snapshot_indices=snap_indices,
            )
            segment_chosen_from_lumps = True
        else:
            raise RuntimeError("lumps.json not found; skipping lump-based selection.")
    except RuntimeError as e:
        print("WARNING:", str(e))
        print(
            "WARNING: Falling back to 1-site segment with maximum occupancy variance "
            "over time."
        )
        ref_step, segment_sites = choose_fallback_segment(occupancy_t)

    segment_sites = sorted(int(s) for s in segment_sites)
    m = len(segment_sites)

    print("Reference segment (S):")
    print(f"  reference step: {ref_step}")
    print(f"  segment size m: {m}")
    print(f"  segment sites:  {segment_sites}")
    if ref_step < len(times):
        print(f"  reference time: {times[ref_step]:.6f}")
    print(f"  chosen_from_lumps: {segment_chosen_from_lumps}")
    print("------------------------------------------------------------")

    if args.verbose:
        print("Verbose: showing reference occupancy pattern on S:")
        occ_ref = occupancy_t[ref_step, segment_sites]
        print(f"  occupancy_t[{ref_step}][S] = {occ_ref.tolist()}")
        print("------------------------------------------------------------")

    # Build a dict step -> psi for quick lookup
    psi_by_step: Dict[int, np.ndarray] = {
        int(step): psi_snap[i] for i, step in enumerate(snap_indices)
    }

    # Analysis over snapshot steps (sorted)
    sorted_steps = sorted(snap_indices)

    print("Step-by-step branch amplitude on S:")
    print(
        "step  time        pattern(S)      |A_S|^2        phase(rad)    dphase(prev)  internal_change"
    )
    print("-" * 90)

    prev_phase = None
    for step in sorted_steps:
        if step < 0 or step >= n_steps:
            continue

        t = float(times[step])

        psi = psi_by_step[step]
        # occupancy pattern on S at this step
        pattern_bits = occupancy_t[step, segment_sites]
        amp = segment_branch_amplitude(psi, n_sites, segment_sites, pattern_bits)
        mag2 = float(np.abs(amp) ** 2)
        phase = float(np.angle(amp))
        if prev_phase is None:
            dphase = 0.0
        else:
            # unwrap small-ish jump around [-pi, pi]
            raw = phase - prev_phase
            # map to (-pi, pi]
            dphase = float((raw + np.pi) % (2.0 * np.pi) - np.pi)

        prev_phase = phase

        internal_flag = (
            bool(internal_change_t[step]) if step < len(internal_change_t) else False
        )

        pattern_str = "".join(str(int(b)) for b in pattern_bits)

        print(
            f"{step:4d}  {t:10.6f}  {pattern_str:>8s}  "
            f"{mag2:12.6e}  {phase:12.6f}  {dphase:12.6f}  {internal_flag}"
        )

    print("-" * 90)
    print("Done.")


if __name__ == "__main__":
    main()
