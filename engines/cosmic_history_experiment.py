#!/usr/bin/env python3
"""
cosmic_history_experiment.py

Single-run "cosmic history" experiment for the Hilbert substrate.

Goal
-----
Run ONE long substrate evolution with fixed parameters and measure how
photon-like packets behave at different epochs of the run, as a toy
opaque -> transparent history.

We DO NOT add any new physics terms. Everything is derived from:

  - Hilbert-space substrate (Config, Substrate)
  - Unitary evolution with defrag
  - pattern_detector.analyze_substrate
  - Existing photon packet tracker from scan_defrag_photon_packets.py

New in this version
-------------------
- n_nodes default increased to 128.
- max_shell_radius default increased to 8.
- Added better "space size" metrics:
    * r_cm(t): amplitude-weighted mean radius of the photon shell.
    * r_90(t): minimal radius where cumulative photon shell amplitude
              reaches 90% of total ("photon horizon radius").

Outputs
--------
- Prints a per-bin summary to stdout.
- Writes a CSV file (default: cosmic_opacity_history.csv) with columns:

    bin_index, t_start, t_end, t_center, n_frames,
    mean_dist_pe, frac_dist_pe_le1, frac_dist_pe_le2,
    mean_r_peak, mean_r_cm, mean_r_90,
    n_up_hops, n_up_tracks, P_up_packet, up_P_reach_target,
    up_mean_track_len, up_mean_max_radius, up_mean_radial_span,
    up_mean_peak_amp,
    n_down_hops, n_down_tracks, P_down_packet, down_P_reach_target,
    down_mean_track_len, down_mean_max_radius, down_mean_radial_span,
    down_mean_peak_amp
"""

import csv
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from substrate import Config, Substrate
import pattern_detector as pd
from scan_defrag_photon_packets import (
    bfs_distances,
    graph_distance,
    classify_hops_multi,
    detect_packet_tracks_for_hops,
    PhotonPacketTrack,
)


@dataclass
class CosmicHistoryConfig:
    # Substrate / graph parameters
    n_nodes: int = 128
    connectivity: float = 0.35
    internal_dim: int = 3
    monogamy_budget: float = 1.0
    defrag_rate: float = 0.1
    dt: float = 0.1
    seed: int = 2025

    # Evolution controls
    burn_in_steps: int = 500
    total_steps: int = 5000
    record_stride: int = 10
    max_graph_radius: int = 12
    max_shell_radius: int = 8

    # "Cosmic" analysis controls
    n_time_bins: int = 4
    target_radius: int = 4  # radius R where we ask "did the packet reach R?"

    # Output
    output_csv: str = "cosmic_opacity_history.csv"


def _compute_bin_indices(times: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given monotonically increasing times (T,), return bin_edges (n_bins+1,)
    and an array bin_index[t] in {0..n_bins-1} for each frame.
    """
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty 1D array")

    t_min = float(times[0])
    t_max = float(times[-1])
    if t_max <= t_min:
        edges = np.linspace(t_min, t_min + 1.0, num=n_bins + 1)
        return edges, np.zeros_like(times, dtype=int)

    edges = np.linspace(t_min, t_max, num=n_bins + 1)
    # np.digitize returns bin in 1..n_bins, we shift to 0..n_bins-1
    bin_idx = np.digitize(times, edges[1:-1], right=False)
    return edges, bin_idx.astype(int)


def _compute_track_stats(
    tracks: List[PhotonPacketTrack],
    target_radius: int,
) -> Tuple[int, float, float, float, float, float]:
    """
    Compute packet statistics for a set of tracks.

    Returns:
      n_tracks_used,
      P_reach_target         (fraction of tracks with max_radius >= target_radius),
      mean_track_len,
      mean_max_radius,
      mean_radial_span,
      mean_peak_amp
    """
    n_tracks = len(tracks)
    if n_tracks == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0

    max_radii = []
    spans = []
    lens = []
    peak_amps = []
    reached_target = 0

    for tr in tracks:
        if tr.radii.size == 0:
            continue
        r_max = int(np.max(tr.radii))
        r_min = int(np.min(tr.radii))
        max_radii.append(r_max)
        spans.append(r_max - r_min)
        lens.append(int(tr.radii.size))
        peak_amps.append(float(np.max(tr.amplitudes)))
        if r_max >= target_radius:
            reached_target += 1

    if len(max_radii) == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0

    n_eff = len(max_radii)
    P_reach = float(reached_target) / float(n_eff)
    mean_len = float(np.mean(lens))
    mean_max_r = float(np.mean(max_radii))
    mean_span = float(np.mean(spans))
    mean_peak = float(np.mean(peak_amps))

    return n_eff, P_reach, mean_len, mean_max_r, mean_span, mean_peak


def _shell_space_metrics(shell_e: np.ndarray) -> Tuple[float, float, float]:
    """
    Given a 1D array shell_e[r] of photon amplitude per shell (r=0..Rmax),
    compute:

      r_peak  - index of maximal shell amplitude
      r_cm    - amplitude-weighted mean radius
      r_90    - minimal radius where cumulative amplitude >= 90% of total

    We treat shell_e as a *positive* photon amplitude distribution by taking
    its absolute value, to avoid sign-cancellation artifacts from the
    z-scored photon_score.
    """
    if shell_e.ndim != 1 or shell_e.size == 0:
        return 0.0, 0.0, 0.0

    # Use positive amplitude for "where is the photon field?"
    shell = np.abs(shell_e)

    total = float(np.sum(shell))
    if total <= 1e-12:
        # effectively no photon amplitude in the shells
        return 0.0, 0.0, 0.0

    r_indices = np.arange(shell.size, dtype=float)

    # r_peak
    r_peak = float(np.argmax(shell))

    # r_cm
    r_cm = float(np.sum(r_indices * shell) / total)

    # r_90
    cum = np.cumsum(shell) / total
    idxs = np.where(cum >= 0.9)[0]
    if idxs.size == 0:
        r_90 = float(shell.size - 1)
    else:
        r_90 = float(idxs[0])

    return r_peak, r_cm, r_90


def run_cosmic_history(cfg: CosmicHistoryConfig) -> None:
    """
    Run a single long substrate evolution and measure photon packet behavior
    in multiple time bins, as a toy opaque -> transparent history.
    """
    # Build base substrate config
    base_cfg = Config(
        n_nodes=cfg.n_nodes,
        internal_dim=cfg.internal_dim,
        monogamy_budget=cfg.monogamy_budget,
        defrag_rate=cfg.defrag_rate,  # we also pass this explicitly per evolve step
        dt=cfg.dt,
        seed=cfg.seed,
        connectivity=cfg.connectivity,
    )

    print("============================================================")
    print("  Cosmic history experiment (single run)")
    print("------------------------------------------------------------")
    print(f"  n_nodes         = {cfg.n_nodes}")
    print(f"  connectivity    = {cfg.connectivity}")
    print(f"  internal_dim    = {cfg.internal_dim}")
    print(f"  monogamy_budget = {cfg.monogamy_budget}")
    print(f"  defrag_rate     = {cfg.defrag_rate}")
    print(f"  dt              = {cfg.dt}")
    print(f"  seed            = {cfg.seed}")
    print(f"  burn_in_steps   = {cfg.burn_in_steps}")
    print(f"  total_steps     = {cfg.total_steps}")
    print(f"  record_stride   = {cfg.record_stride}")
    print(f"  max_shell_radius= {cfg.max_shell_radius}")
    print(f"  max_graph_radius= {cfg.max_graph_radius}")
    print(f"  n_time_bins     = {cfg.n_time_bins}")
    print(f"  target_radius   = {cfg.target_radius}")
    print("============================================================\n")

    substrate = Substrate(base_cfg)
    neighbors = substrate._neighbors  # list[list[int]]
    n_nodes = substrate.n_nodes
    dt = base_cfg.dt

    print("  Burn-in evolution...")
    substrate.evolve(n_steps=cfg.burn_in_steps, defrag_rate=cfg.defrag_rate)

    times: List[float] = []
    dist_pe: List[int] = []
    shell_e_list: List[np.ndarray] = []
    r_peak_list: List[float] = []
    r_cm_list: List[float] = []
    r_90_list: List[float] = []

    t = 0.0

    def record(step_idx: int, t_now: float) -> None:
        """Record one snapshot of proton/electron distance and photon shells."""
        feats, scores, cands = pd.analyze_substrate(substrate)
        p_id = cands.proton_id
        e_id = cands.electron_id

        times.append(t_now)

        d_pe = graph_distance(neighbors, p_id, e_id, max_radius=cfg.max_graph_radius)
        dist_pe.append(d_pe)

        ph_score = scores.photon_score  # (N,)

        dist_from_electron = bfs_distances(
            neighbors, e_id, max_radius=cfg.max_shell_radius
        )

        shell_e = np.zeros(cfg.max_shell_radius + 1, dtype=float)
        for nid in range(n_nodes):
            d_e = dist_from_electron[nid]
            if d_e <= cfg.max_shell_radius:
                shell_e[d_e] += ph_score[nid]

        shell_e_list.append(shell_e)

        r_peak, r_cm, r_90 = _shell_space_metrics(shell_e)
        r_peak_list.append(r_peak)
        r_cm_list.append(r_cm)
        r_90_list.append(r_90)

        print(
            f"  [t={t_now:.3f}] p_id={p_id}, e_id={e_id}, "
            f"dist_p-e={d_pe}, r_peak={r_peak:.2f}, r_cm={r_cm:.2f}, r_90={r_90:.2f}"
        )

    # Main evolution loop
    print("  Main evolution...")
    for step in range(1, cfg.total_steps + 1):
        substrate.evolve(n_steps=1, defrag_rate=cfg.defrag_rate)
        t += dt
        if (step % cfg.record_stride == 0) or (step == cfg.total_steps):
            record(step_idx=step, t_now=t)

    times_arr = np.asarray(times, dtype=float)
    dist_pe_arr = np.asarray(dist_pe, dtype=int)
    shell_e_arr = np.stack(shell_e_list, axis=0)
    r_peak_arr = np.asarray(r_peak_list, dtype=float)
    r_cm_arr = np.asarray(r_cm_list, dtype=float)
    r_90_arr = np.asarray(r_90_list, dtype=float)
    seed_ids_arr = np.zeros_like(dist_pe_arr, dtype=int)  # single-seed tag

    # Basic binding summary
    mean_pe_distance = float(np.mean(dist_pe_arr))
    frac_bind1 = float(np.mean(dist_pe_arr <= 1))
    frac_bind2 = float(np.mean(dist_pe_arr <= 2))

    print("\n  --- overall run summary ---")
    print(f"  frames recorded     : {len(times_arr)}")
    print(f"  mean dist_p-e       : {mean_pe_distance:.3f}")
    print(f"  frac dist<=1        : {frac_bind1:.3f}")
    print(f"  frac dist<=2        : {frac_bind2:.3f}")

    # Classify hops and detect tracks across the whole run
    up_indices, down_indices = classify_hops_multi(dist_pe_arr, seed_ids_arr)
    print(f"\n  Total up-hops (1->2):   {len(up_indices)}")
    print(f"  Total down-hops (2->1): {len(down_indices)}")

    print("  Detecting outward-moving packets around down-hops...")
    down_tracks_all = detect_packet_tracks_for_hops(
        times=times_arr,
        shell_e=shell_e_arr,
        hop_indices=down_indices,
        hop_type="down",
        amp_threshold_factor=0.5,
        min_track_len=3,
        window_before=1,
        window_after=3,
    )

    print("  Detecting inward-moving packets around up-hops...")
    up_tracks_all = detect_packet_tracks_for_hops(
        times=times_arr,
        shell_e=shell_e_arr,
        hop_indices=up_indices,
        hop_type="up",
        amp_threshold_factor=0.5,
        min_track_len=3,
        window_before=1,
        window_after=3,
    )

    # Time-binning for "cosmic history"
    edges, bin_index = _compute_bin_indices(times_arr, cfg.n_time_bins)

    # Pre-bin hop indices -> bin
    up_indices_by_bin: List[List[int]] = [[] for _ in range(cfg.n_time_bins)]
    down_indices_by_bin: List[List[int]] = [[] for _ in range(cfg.n_time_bins)]
    for idx in up_indices:
        b = int(bin_index[idx])
        if 0 <= b < cfg.n_time_bins:
            up_indices_by_bin[b].append(idx)
    for idx in down_indices:
        b = int(bin_index[idx])
        if 0 <= b < cfg.n_time_bins:
            down_indices_by_bin[b].append(idx)

    # Map hop_frame -> track(s)
    up_tracks_by_hop: dict[int, List[PhotonPacketTrack]] = {}
    for tr in up_tracks_all:
        up_tracks_by_hop.setdefault(int(tr.hop_frame), []).append(tr)
    down_tracks_by_hop: dict[int, List[PhotonPacketTrack]] = {}
    for tr in down_tracks_all:
        down_tracks_by_hop.setdefault(int(tr.hop_frame), []).append(tr)

    # Prepare CSV output
    fieldnames = [
        "bin_index",
        "t_start",
        "t_end",
        "t_center",
        "n_frames",
        "mean_dist_pe",
        "frac_dist_pe_le1",
        "frac_dist_pe_le2",
        "mean_r_peak",
        "mean_r_cm",
        "mean_r_90",
        "n_up_hops",
        "n_up_tracks",
        "P_up_packet",
        "up_P_reach_target",
        "up_mean_track_len",
        "up_mean_max_radius",
        "up_mean_radial_span",
        "up_mean_peak_amp",
        "n_down_hops",
        "n_down_tracks",
        "P_down_packet",
        "down_P_reach_target",
        "down_mean_track_len",
        "down_mean_max_radius",
        "down_mean_radial_span",
        "down_mean_peak_amp",
    ]

    print("\n  --- per-bin cosmic opacity / space metrics summary ---")
    with open(cfg.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for b in range(cfg.n_time_bins):
            t_start = float(edges[b])
            t_end = float(edges[b + 1])
            mask = (bin_index == b)
            idx_frames = np.where(mask)[0]
            n_frames = int(idx_frames.size)
            if n_frames == 0:
                print(f"  Bin {b}: empty (no recorded frames).")
                row = {name: 0 for name in fieldnames}
                row["bin_index"] = b
                row["t_start"] = t_start
                row["t_end"] = t_end
                row["t_center"] = 0.5 * (t_start + t_end)
                writer.writerow(row)
                continue

            dist_bin = dist_pe_arr[mask]
            mean_dist = float(np.mean(dist_bin))
            frac1 = float(np.mean(dist_bin <= 1))
            frac2 = float(np.mean(dist_bin <= 2))

            mean_rpk = float(np.mean(r_peak_arr[mask]))
            mean_rcm = float(np.mean(r_cm_arr[mask]))
            mean_r90 = float(np.mean(r_90_arr[mask]))

            t_center = float(np.mean(times_arr[mask]))

            # Hops / tracks restricted to this bin
            up_hops_bin = up_indices_by_bin[b]
            down_hops_bin = down_indices_by_bin[b]

            up_tracks_bin: List[PhotonPacketTrack] = []
            for h in up_hops_bin:
                up_tracks_bin.extend(up_tracks_by_hop.get(int(h), []))

            down_tracks_bin: List[PhotonPacketTrack] = []
            for h in down_hops_bin:
                down_tracks_bin.extend(down_tracks_by_hop.get(int(h), []))

            n_up_hops = len(up_hops_bin)
            n_down_hops = len(down_hops_bin)

            (
                n_up_eff,
                up_P_reach,
                up_mean_len,
                up_mean_max_r,
                up_mean_span,
                up_mean_peak,
            ) = _compute_track_stats(
                up_tracks_bin, cfg.target_radius
            )
            (
                n_down_eff,
                down_P_reach,
                down_mean_len,
                down_mean_max_r,
                down_mean_span,
                down_mean_peak,
            ) = _compute_track_stats(
                down_tracks_bin, cfg.target_radius
            )

            P_up_packet = float(n_up_eff) / float(n_up_hops) if n_up_hops > 0 else 0.0
            P_down_packet = (
                float(n_down_eff) / float(n_down_hops) if n_down_hops > 0 else 0.0
            )

            print(
                f"  Bin {b}: t~{t_center:.3f}, frames={n_frames}, "
                f"<dist_pe>={mean_dist:.2f}, "
                f"<r_cm>={mean_rcm:.2f}, <r_90>={mean_r90:.2f}, "
                f"up: hops={n_up_hops}, tracks={n_up_eff}, P_packet={P_up_packet:.3f}, "
                f"down: hops={n_down_hops}, tracks={n_down_eff}, P_packet={P_down_packet:.3f}"
            )

            row = {
                "bin_index": b,
                "t_start": t_start,
                "t_end": t_end,
                "t_center": t_center,
                "n_frames": n_frames,
                "mean_dist_pe": mean_dist,
                "frac_dist_pe_le1": frac1,
                "frac_dist_pe_le2": frac2,
                "mean_r_peak": mean_rpk,
                "mean_r_cm": mean_rcm,
                "mean_r_90": mean_r90,
                "n_up_hops": n_up_hops,
                "n_up_tracks": n_up_eff,
                "P_up_packet": P_up_packet,
                "up_P_reach_target": up_P_reach,
                "up_mean_track_len": up_mean_len,
                "up_mean_max_radius": up_mean_max_r,
                "up_mean_radial_span": up_mean_span,
                "up_mean_peak_amp": up_mean_peak,
                "n_down_hops": n_down_hops,
                "n_down_tracks": n_down_eff,
                "P_down_packet": P_down_packet,
                "down_P_reach_target": down_P_reach,
                "down_mean_track_len": down_mean_len,
                "down_mean_max_radius": down_mean_max_r,
                "down_mean_radial_span": down_mean_span,
                "down_mean_peak_amp": down_mean_peak,
            }
            writer.writerow(row)

    print(f"\n  Cosmic opacity / space history written to: {cfg.output_csv}\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Single-run cosmic opacity / transparency history experiment."
    )
    parser.add_argument("--n_nodes", type=int, default=128)
    parser.add_argument("--connectivity", type=float, default=0.35)
    parser.add_argument("--internal_dim", type=int, default=3)
    parser.add_argument("--monogamy_budget", type=float, default=1.0)
    parser.add_argument("--defrag_rate", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--burn_in_steps", type=int, default=500)
    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--record_stride", type=int, default=10)
    parser.add_argument("--max_graph_radius", type=int, default=12)
    parser.add_argument("--max_shell_radius", type=int, default=8)
    parser.add_argument("--n_time_bins", type=int, default=4)
    parser.add_argument("--target_radius", type=int, default=4)
    parser.add_argument(
        "--output_csv", type=str, default="cosmic_opacity_history.csv"
    )

    args = parser.parse_args()

    cfg = CosmicHistoryConfig(
        n_nodes=args.n_nodes,
        connectivity=args.connectivity,
        internal_dim=args.internal_dim,
        monogamy_budget=args.monogamy_budget,
        defrag_rate=args.defrag_rate,
        dt=args.dt,
        seed=args.seed,
        burn_in_steps=args.burn_in_steps,
        total_steps=args.total_steps,
        record_stride=args.record_stride,
        max_graph_radius=args.max_graph_radius,
        max_shell_radius=args.max_shell_radius,
        n_time_bins=args.n_time_bins,
        target_radius=args.target_radius,
        output_csv=args.output_csv,
    )

    run_cosmic_history(cfg)


if __name__ == "__main__":
    main()
