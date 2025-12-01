#!/usr/bin/env python3
"""
scan_defrag_photon_packets.py

Scan over different defrag_rate ("temperature") values and measure how often
electron shell hops produce trackable photon packets (inward for up-hops,
outward for down-hops) in the Hilbert substrate engine.

Outputs:
  - Prints a summary table to stdout.
  - Saves a CSV-style text file: defrag_packet_stats.csv
    with columns:
      defrag_rate, n_up_hops, n_up_tracks, P_up_packet,
                   n_down_hops, n_down_tracks, P_down_packet

Updated for new substrate.py API:
  - substrate._neighbors is list[list[int]], not dict
  - Direct indexing instead of .get()
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from typing import List, Literal, Optional, Tuple

import numpy as np

from substrate import Config, Substrate  # type: ignore
import pattern_detector as pd  # type: ignore


# ---------------------------------------------------------------------
# Graph distance helpers
# ---------------------------------------------------------------------


def bfs_distances(neighbors: List[List[int]], src: int, max_radius: int) -> np.ndarray:
    """
    Compute BFS distances from src to all nodes up to max_radius.

    neighbors: list of neighbor lists indexed by node id
    Returns:
      dist: np.ndarray of shape (N,), dist[i] = graph distance(src, i)
            or max_radius + 1 if further.
    """
    n_nodes = len(neighbors)
    dist = np.full(n_nodes, max_radius + 1, dtype=int)
    if n_nodes == 0:
        return dist

    from collections import deque

    dist[src] = 0
    q = deque([src])
    while q:
        nid = q.popleft()
        d = dist[nid]
        if d >= max_radius:
            continue
        for nb in neighbors[nid]:
            if nb < 0 or nb >= n_nodes:
                continue
            if dist[nb] > d + 1:
                dist[nb] = d + 1
                q.append(nb)
    return dist


def graph_distance(neighbors: List[List[int]], a_id: int, b_id: int, max_radius: int) -> int:
    """
    BFS distance between nodes a and b in the substrate graph,
    clipped at max_radius+1 if further.
    """
    if a_id == b_id:
        return 0
    dist = bfs_distances(neighbors, a_id, max_radius=max_radius)
    return int(dist[b_id])


# ---------------------------------------------------------------------
# Photon packet tracking
# ---------------------------------------------------------------------


@dataclass
class PhotonPacketTrack:
    hop_type: Literal["up", "down"]
    hop_frame: int
    times: np.ndarray
    radii: np.ndarray
    amplitudes: np.ndarray


def classify_hops_multi(dist_pe: np.ndarray, seed_ids: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Identify up-hops and down-hops from dist_pe[t], respecting seed boundaries.

    Up-hop:   1 -> 2 (within same seed)
    Down-hop: 2 -> 1 (within same seed)
    """
    T = len(dist_pe)
    up_indices: List[int] = []
    down_indices: List[int] = []
    for t in range(1, T):
        if seed_ids[t] != seed_ids[t - 1]:
            continue
        d_prev = int(dist_pe[t - 1])
        d_now = int(dist_pe[t])
        if d_prev == d_now:
            continue
        if d_prev == 1 and d_now == 2:
            up_indices.append(t)
        elif d_prev == 2 and d_now == 1:
            down_indices.append(t)
    return up_indices, down_indices


def detect_packet_tracks_for_hops(
    times: np.ndarray,
    shell_e: np.ndarray,        # (T, R+1)
    hop_indices: List[int],
    hop_type: Literal["up", "down"],
    amp_threshold_factor: float = 0.5,
    min_track_len: int = 3,
    window_before: int = 1,
    window_after: int = 3,
) -> List[PhotonPacketTrack]:
    """
    Detect coherent photon packet tracks in Δ_shell_e[t, r]
    around electron-centric shells, for given hop set.
    """
    T, _ = shell_e.shape
    baseline_e = np.mean(shell_e, axis=0)
    delta_shell_e = shell_e - baseline_e[np.newaxis, :]

    sigma_delta = float(np.std(delta_shell_e))
    if sigma_delta < 1e-12:
        amp_threshold = 0.0
    else:
        amp_threshold = amp_threshold_factor * sigma_delta

    tracks: List[PhotonPacketTrack] = []

    for t0 in hop_indices:
        t_start = max(0, t0 - window_before)
        t_end = min(T - 1, t0 + window_after)
        window_ts = np.arange(t_start, t_end + 1, dtype=int)
        if window_ts.size == 0:
            continue

        radii: List[int] = []
        amps: List[float] = []
        times_sel: List[float] = []

        for t in window_ts:
            delta_shell_t = delta_shell_e[t]
            r_max = int(np.argmax(delta_shell_t))
            amp = float(delta_shell_t[r_max])
            if amp > amp_threshold:
                radii.append(r_max)
                amps.append(amp)
                times_sel.append(float(times[t]))

        if len(radii) < min_track_len:
            continue

        radii_arr = np.asarray(radii, dtype=int)
        amps_arr = np.asarray(amps, dtype=float)
        times_arr = np.asarray(times_sel, dtype=float)

        diffs = np.diff(radii_arr)
        if hop_type == "down":
            non_decreasing = np.all(diffs >= 0)
            has_strict_outward = np.any(diffs > 0)
            if not (non_decreasing and has_strict_outward):
                continue
        else:
            non_increasing = np.all(diffs <= 0)
            has_strict_inward = np.any(diffs < 0)
            if not (non_increasing and has_strict_inward):
                continue

        tracks.append(
            PhotonPacketTrack(
                hop_type=hop_type,
                hop_frame=int(t0),
                times=times_arr,
                radii=radii_arr,
                amplitudes=amps_arr,
            )
        )

    return tracks


# ---------------------------------------------------------------------
# Single-seed photon event probe for a given defrag_rate
# ---------------------------------------------------------------------


def run_single_seed_probe_for_defrag(
    base_config: Config,
    defrag_rate: float,
    seed: int,
    burn_in_steps: int = 300,
    total_steps: int = 3000,
    record_stride: int = 10,
    max_graph_radius: int = 10,
    max_shell_radius: int = 5,
):
    """
    Run one photon event probe with a given defrag_rate and seed.

    Returns:
      times_s         : (T,)
      dist_pe_s       : (T,)
      shell_p_s       : (T, R+1)
      shell_e_s       : (T, R+1)
      seed_ids_s      : (T,)  (filled with `seed`)
    """
    # Build a Config for this (defrag_rate, seed) combo
    config_dict = asdict(base_config)
    config_dict["defrag_rate"] = defrag_rate
    config_dict["seed"] = seed
    config = Config(**config_dict)

    if not hasattr(config, "dt"):
        setattr(config, "dt", getattr(base_config, "dt", 0.1))
    dt = getattr(config, "dt", 0.1)

    print("------------------------------------------------------------")
    print(f"  Photon event probe (defrag={defrag_rate}, seed={seed})")
    print(
        f"  burn_in_steps={burn_in_steps}, total_steps={total_steps}, "
        f"record_stride={record_stride}"
    )
    print("------------------------------------------------------------")

    substrate = Substrate(config)
    neighbors = substrate._neighbors  # list[list[int]]
    n_nodes = substrate.n_nodes

    print("  Burn-in evolution...")
    # IMPORTANT: actually pass defrag_rate here
    substrate.evolve(n_steps=burn_in_steps, defrag_rate=defrag_rate)

    times: List[float] = []
    dist_pe: List[int] = []
    photon_shell_proton: List[np.ndarray] = []
    photon_shell_electron: List[np.ndarray] = []

    t = 0.0

    def record(step_idx: int, t: float):
        feats, scores, cands = pd.analyze_substrate(substrate)

        p_id = cands.proton_id
        e_id = cands.electron_id

        times.append(t)

        d_pe = graph_distance(neighbors, p_id, e_id, max_radius=max_graph_radius)
        dist_pe.append(d_pe)

        ph_score = scores.photon_score  # (N,)

        dist_from_proton = bfs_distances(neighbors, p_id, max_radius=max_shell_radius)
        dist_from_electron = bfs_distances(neighbors, e_id, max_radius=max_shell_radius)

        shell_p = np.zeros(max_shell_radius + 1, dtype=float)
        shell_e = np.zeros(max_shell_radius + 1, dtype=float)

        for nid in range(n_nodes):
            d_p = dist_from_proton[nid]
            d_e = dist_from_electron[nid]
            if d_p <= max_shell_radius:
                shell_p[d_p] += ph_score[nid]
            if d_e <= max_shell_radius:
                shell_e[d_e] += ph_score[nid]

        photon_shell_proton.append(shell_p)
        photon_shell_electron.append(shell_e)

        print(
            f"  [defrag={defrag_rate}, seed={seed}, t={t:.2f}] "
            f"p_id={p_id}, e_id={e_id}, dist_p-e={d_pe}"
        )

    # initial snapshot
    record(step_idx=0, t=t)

    # main evolution
    for step in range(1, total_steps + 1):
        # IMPORTANT: pass defrag_rate every step
        substrate.evolve(n_steps=1, defrag_rate=defrag_rate)
        t += dt
        if (step % record_stride == 0) or (step == total_steps):
            record(step_idx=step, t=t)

    times_arr = np.asarray(times, dtype=float)
    dist_pe_arr = np.asarray(dist_pe, dtype=int)
    shell_p_arr = np.stack(photon_shell_proton, axis=0)
    shell_e_arr = np.stack(photon_shell_electron, axis=0)
    seed_ids_arr = np.full(times_arr.shape, seed, dtype=int)

    mean_pe_distance = float(np.mean(dist_pe_arr))
    frac_bind1 = float(np.mean(dist_pe_arr <= 1))
    frac_bind2 = float(np.mean(dist_pe_arr <= 2))

    print("  --- per-run summary ---")
    print(f"  frames: {len(times_arr)}")
    print(f"  mean dist_pe: {mean_pe_distance:.3f}")
    print(f"  frac dist<=1: {frac_bind1:.3f}, dist<=2: {frac_bind2:.3f}")

    return times_arr, dist_pe_arr, shell_p_arr, shell_e_arr, seed_ids_arr


# ---------------------------------------------------------------------
# Defrag sweep driver
# ---------------------------------------------------------------------


def run_defrag_sweep(
    n_nodes: int = 64,
    connectivity: float = 0.35,
    max_shell_radius: int = 5,
    defrag_values: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    total_steps: int = 3000,
    record_stride: int = 10,
    window_after: int = 3,
):
    """
    Run defrag sweep with configurable graph size and propagation tracking.
    
    For long-range coherence tests, use:
        n_nodes=256, connectivity=0.08, max_shell_radius=10, window_after=8
    """
    if defrag_values is None:
        defrag_values = [0.02, 0.05, 0.1, 0.2, 0.4]
    if seeds is None:
        seeds = [987, 1234, 2025]
    
    # Base config shared across defrag values and seeds
    base_config = Config(
        n_nodes=n_nodes,
        internal_dim=3,
        monogamy_budget=1.0,
        defrag_rate=0.1,  # overridden per run
        seed=0,
        connectivity=connectivity,
    )
    if not hasattr(base_config, "dt"):
        setattr(base_config, "dt", 0.1)

    burn_in_steps = 300
    max_graph_radius = max_shell_radius + 5

    print("============================================================")
    print("  Defrag-rate sweep: photon packet formation")
    print("============================================================")
    print(
        f"Graph: n_nodes={n_nodes}, connectivity={connectivity}"
    )
    print(
        f"Tracking: max_shell_radius={max_shell_radius}, window_after={window_after}"
    )
    print(
        f"Base config: d={base_config.internal_dim}, "
        f"monogamy={base_config.monogamy_budget}, dt={base_config.dt}"
    )
    print(f"Defrag values: {defrag_values}")
    print(f"Seeds per defrag: {seeds}")
    print("============================================================\n")

    summary_rows = []

    for defrag in defrag_values:
        print("\n############################################################")
        print(f"  Defrag rate = {defrag}")
        print("############################################################")

        all_times: List[np.ndarray] = []
        all_dist: List[np.ndarray] = []
        all_shell_e: List[np.ndarray] = []
        all_seed_ids: List[np.ndarray] = []

        for s in seeds:
            times_s, dist_s, shell_p_s, shell_e_s, seed_ids_s = run_single_seed_probe_for_defrag(
                base_config=base_config,
                defrag_rate=defrag,
                seed=s,
                burn_in_steps=burn_in_steps,
                total_steps=total_steps,
                record_stride=record_stride,
                max_graph_radius=max_graph_radius,
                max_shell_radius=max_shell_radius,
            )

            all_times.append(times_s)
            all_dist.append(dist_s)
            all_shell_e.append(shell_e_s)
            all_seed_ids.append(seed_ids_s)

        times_all = np.concatenate(all_times, axis=0)
        dist_all = np.concatenate(all_dist, axis=0)
        shell_e_all = np.concatenate(all_shell_e, axis=0)
        seed_ids_all = np.concatenate(all_seed_ids, axis=0)

        # Classify hops
        up_indices, down_indices = classify_hops_multi(dist_all, seed_ids_all)
        n_up = len(up_indices)
        n_down = len(down_indices)

        print(f"\n  Total up-hops (1->2):   {n_up}")
        print(f"  Total down-hops (2->1): {n_down}")

        # Detect tracks
        print("  Detecting outward-moving packets around down-hops...")
        down_tracks = detect_packet_tracks_for_hops(
            times=times_all,
            shell_e=shell_e_all,
            hop_indices=down_indices,
            hop_type="down",
            amp_threshold_factor=0.5,
            min_track_len=3,
            window_before=1,
            window_after=window_after,
        )

        print("  Detecting inward-moving packets around up-hops...")
        up_tracks = detect_packet_tracks_for_hops(
            times=times_all,
            shell_e=shell_e_all,
            hop_indices=up_indices,
            hop_type="up",
            amp_threshold_factor=0.5,
            min_track_len=3,
            window_before=1,
            window_after=window_after,
        )

        n_down_tracks = len(down_tracks)
        n_up_tracks = len(up_tracks)

        P_down = float(n_down_tracks) / n_down if n_down > 0 else 0.0
        P_up = float(n_up_tracks) / n_up if n_up > 0 else 0.0

        print("\n  --- defrag-level packet summary ---")
        print(f"  defrag={defrag}")
        print(f"    down: tracks={n_down_tracks} / hops={n_down}  => P_down={P_down:.3f}")
        print(f"    up  : tracks={n_up_tracks}   / hops={n_up}    => P_up={P_up:.3f}")

        # Compute amplitude statistics for detected tracks
        def compute_track_stats(tracks: List[PhotonPacketTrack]):
            if not tracks:
                return {
                    "mean_amp": 0.0,
                    "mean_peak_amp": 0.0,
                    "mean_decay_rate": 0.0,
                    "mean_track_len": 0.0,
                    "mean_max_radius": 0.0,
                    "mean_radial_span": 0.0,
                }
            
            all_amps = []
            peak_amps = []
            decay_rates = []
            track_lens = []
            max_radii = []
            radial_spans = []
            
            for tr in tracks:
                all_amps.extend(tr.amplitudes.tolist())
                peak_amps.append(float(np.max(tr.amplitudes)))
                track_lens.append(len(tr.amplitudes))
                max_radii.append(int(np.max(tr.radii)))
                radial_spans.append(int(np.max(tr.radii) - np.min(tr.radii)))
                
                # Decay rate: linear fit of amplitude vs time index
                if len(tr.amplitudes) >= 2:
                    t_idx = np.arange(len(tr.amplitudes), dtype=float)
                    # Normalized decay: (amp[end] - amp[start]) / amp[start] / n_steps
                    amp_start = tr.amplitudes[0]
                    amp_end = tr.amplitudes[-1]
                    if amp_start > 1e-10:
                        rel_decay = (amp_end - amp_start) / amp_start / (len(tr.amplitudes) - 1)
                        decay_rates.append(float(rel_decay))
            
            return {
                "mean_amp": float(np.mean(all_amps)) if all_amps else 0.0,
                "mean_peak_amp": float(np.mean(peak_amps)) if peak_amps else 0.0,
                "mean_decay_rate": float(np.mean(decay_rates)) if decay_rates else 0.0,
                "mean_track_len": float(np.mean(track_lens)) if track_lens else 0.0,
                "mean_max_radius": float(np.mean(max_radii)) if max_radii else 0.0,
                "mean_radial_span": float(np.mean(radial_spans)) if radial_spans else 0.0,
            }
        
        up_stats = compute_track_stats(up_tracks)
        down_stats = compute_track_stats(down_tracks)
        
        print(f"    up  stats: mean_amp={up_stats['mean_amp']:.2f}, decay={up_stats['mean_decay_rate']:.3f}, max_r={up_stats['mean_max_radius']:.1f}, span={up_stats['mean_radial_span']:.1f}")
        print(f"    down stats: mean_amp={down_stats['mean_amp']:.2f}, decay={down_stats['mean_decay_rate']:.3f}, max_r={down_stats['mean_max_radius']:.1f}, span={down_stats['mean_radial_span']:.1f}")

        summary_rows.append(
            {
                "defrag_rate": defrag,
                "n_up_hops": n_up,
                "n_up_tracks": n_up_tracks,
                "P_up_packet": P_up,
                "n_down_hops": n_down,
                "n_down_tracks": n_down_tracks,
                "P_down_packet": P_down,
                "up_mean_amp": up_stats["mean_amp"],
                "up_peak_amp": up_stats["mean_peak_amp"],
                "up_decay_rate": up_stats["mean_decay_rate"],
                "up_track_len": up_stats["mean_track_len"],
                "up_max_radius": up_stats["mean_max_radius"],
                "up_radial_span": up_stats["mean_radial_span"],
                "down_mean_amp": down_stats["mean_amp"],
                "down_peak_amp": down_stats["mean_peak_amp"],
                "down_decay_rate": down_stats["mean_decay_rate"],
                "down_track_len": down_stats["mean_track_len"],
                "down_max_radius": down_stats["mean_max_radius"],
                "down_radial_span": down_stats["mean_radial_span"],
            }
        )

    # Print final summary table
    print("\n============================================================")
    print("  Defrag sweep summary")
    print("============================================================")
    print(
        "defrag_rate  n_up  n_up_tracks  P_up   n_down  n_down_tracks  P_down"
    )
    for row in summary_rows:
        print(
            f"{row['defrag_rate']:10.3f} "
            f"{row['n_up_hops']:5d} {row['n_up_tracks']:11d} {row['P_up_packet']:.3f}   "
            f"{row['n_down_hops']:6d} {row['n_down_tracks']:13d} {row['P_down_packet']:.3f}"
        )
    
    print("\n--- Amplitude statistics (UP tracks) ---")
    print("defrag_rate  mean_amp  peak_amp  decay_rate  track_len  max_r  span")
    for row in summary_rows:
        print(
            f"{row['defrag_rate']:10.3f} "
            f"{row['up_mean_amp']:9.3f} {row['up_peak_amp']:9.3f} "
            f"{row['up_decay_rate']:11.4f} {row['up_track_len']:10.2f} "
            f"{row['up_max_radius']:6.1f} {row['up_radial_span']:5.1f}"
        )
    
    print("\n--- Amplitude statistics (DOWN tracks) ---")
    print("defrag_rate  mean_amp  peak_amp  decay_rate  track_len  max_r  span")
    for row in summary_rows:
        print(
            f"{row['defrag_rate']:10.3f} "
            f"{row['down_mean_amp']:9.3f} {row['down_peak_amp']:9.3f} "
            f"{row['down_decay_rate']:11.4f} {row['down_track_len']:10.2f} "
            f"{row['down_max_radius']:6.1f} {row['down_radial_span']:5.1f}"
        )

    # Save CSV
    csv_fname = "defrag_packet_stats.csv"
    with open(csv_fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "defrag_rate",
                "n_up_hops",
                "n_up_tracks",
                "P_up_packet",
                "n_down_hops",
                "n_down_tracks",
                "P_down_packet",
                "up_mean_amp",
                "up_peak_amp",
                "up_decay_rate",
                "up_track_len",
                "up_max_radius",
                "up_radial_span",
                "down_mean_amp",
                "down_peak_amp",
                "down_decay_rate",
                "down_track_len",
                "down_max_radius",
                "down_radial_span",
            ]
        )
        for row in summary_rows:
            writer.writerow(
                [
                    row["defrag_rate"],
                    row["n_up_hops"],
                    row["n_up_tracks"],
                    row["P_up_packet"],
                    row["n_down_hops"],
                    row["n_down_tracks"],
                    row["P_down_packet"],
                    row["up_mean_amp"],
                    row["up_peak_amp"],
                    row["up_decay_rate"],
                    row["up_track_len"],
                    row["up_max_radius"],
                    row["up_radial_span"],
                    row["down_mean_amp"],
                    row["down_peak_amp"],
                    row["down_decay_rate"],
                    row["down_track_len"],
                    row["down_max_radius"],
                    row["down_radial_span"],
                ]
            )
    print(f"\nSaved summary to {csv_fname}")


def main():
    import sys
    
    # Check for command line mode
    if len(sys.argv) > 1 and sys.argv[1] == "long":
        print("Running LONG-RANGE coherence test...")
        print("(256 nodes, sparse connectivity, 10-shell tracking)\n")
        run_defrag_sweep(
            n_nodes=256,
            connectivity=0.08,
            max_shell_radius=10,
            defrag_values=[0.02, 0.1, 0.4],
            seeds=[987, 1234],
            total_steps=5000,
            record_stride=5,
            window_after=8,
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "medium":
        print("Running MEDIUM-RANGE coherence test...")
        print("(128 nodes, moderate connectivity, 7-shell tracking)\n")
        run_defrag_sweep(
            n_nodes=128,
            connectivity=0.12,
            max_shell_radius=7,
            defrag_values=[0.02, 0.05, 0.1, 0.2, 0.4],
            seeds=[987, 1234, 2025],
            total_steps=4000,
            record_stride=8,
            window_after=6,
        )
    else:
        print("Running STANDARD test (use 'long' or 'medium' arg for extended tests)...")
        print("(64 nodes, standard connectivity, 5-shell tracking)\n")
        run_defrag_sweep(
            n_nodes=64,
            connectivity=0.35,
            max_shell_radius=5,
            defrag_values=[0.02, 0.05, 0.1, 0.2, 0.4],
            seeds=[987, 1234, 2025],
            total_steps=3000,
            record_stride=10,
            window_after=3,
        )


if __name__ == "__main__":
    main()