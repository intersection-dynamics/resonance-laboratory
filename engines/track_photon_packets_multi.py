#!/usr/bin/env python3
"""
track_photon_packets_multi.py

Detect coherent inward/outward photon packet tracks (around the electron)
from the stacked multi-seed dataset photon_event_probe_multi.npz.

Very similar to track_photon_packets.py, but:

  - Uses photon_shell_electron from the multi-seed file
  - Uses seed_ids to ensure we don't join hops across seeds
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np


@dataclass
class PhotonPacketTrack:
    hop_type: Literal["up", "down"]
    hop_frame: int
    times: np.ndarray
    radii: np.ndarray
    amplitudes: np.ndarray


# ---------------------------------------------------------------------
# Data loading & hop classification
# ---------------------------------------------------------------------


def load_event_data(fname: str = "photon_event_probe_multi.npz"):
    data = np.load(fname, allow_pickle=True)
    times = data["times"]  # (T_total,)
    dist_pe = data["dist_pe"]  # (T_total,)
    shell_e = data["photon_shell_electron"]  # (T_total, R+1)
    seed_ids = data["seed_ids"]  # (T_total,)
    meta_json = data["meta_json"].item()
    meta = json.loads(meta_json)
    return times, dist_pe, shell_e, seed_ids, meta


def classify_hops_multi(dist_pe: np.ndarray, seed_ids: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Same as in analyze_photon_events_multi: classify up/down hops
    *within* each seed.
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


# ---------------------------------------------------------------------
# Packet detection (same logic as single-run version)
# ---------------------------------------------------------------------


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
    T, Rplus1 = shell_e.shape
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


def save_tracks(
    fname: str,
    up_tracks: List[PhotonPacketTrack],
    down_tracks: List[PhotonPacketTrack],
):
    up_times = [tr.times for tr in up_tracks]
    up_radii = [tr.radii for tr in up_tracks]
    up_amps = [tr.amplitudes for tr in up_tracks]
    up_hops = [tr.hop_frame for tr in up_tracks]

    down_times = [tr.times for tr in down_tracks]
    down_radii = [tr.radii for tr in down_tracks]
    down_amps = [tr.amplitudes for tr in down_tracks]
    down_hops = [tr.hop_frame for tr in down_tracks]

    np.savez(
        fname,
        up_times=np.array(up_times, dtype=object),
        up_radii=np.array(up_radii, dtype=object),
        up_amps=np.array(up_amps, dtype=object),
        up_hop_frames=np.array(up_hops, dtype=int),
        down_times=np.array(down_times, dtype=object),
        down_radii=np.array(down_radii, dtype=object),
        down_amps=np.array(down_amps, dtype=object),
        down_hop_frames=np.array(down_hops, dtype=int),
    )


def print_track_summary(tracks: List[PhotonPacketTrack], label: str, max_examples: int = 5):
    print(f"\n=== {label} tracks ===")
    print(f"Total tracks: {len(tracks)}")
    if not tracks:
        return

    lengths = np.array([len(tr.times) for tr in tracks], dtype=int)
    print(f"Track length: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.2f}")

    amps = np.concatenate([tr.amplitudes for tr in tracks])
    print(f"Amplitude stats: min={amps.min():.4f}, max={amps.max():.4f}, mean={amps.mean():.4f}")

    print(f"\nExample tracks (up to {max_examples}):")
    for i, tr in enumerate(tracks[:max_examples]):
        print(f"  Track {i}: hop_frame={tr.hop_frame}, n={len(tr.times)}")
        print(f"    times  : {np.array2string(tr.times, precision=2)}")
        print(f"    radii  : {tr.radii}")
        print(f"    amps   : {np.array2string(tr.amplitudes, precision=3)}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    print("Loading photon_event_probe_multi.npz...")
    times, dist_pe, shell_e, seed_ids, meta = load_event_data()
    T, Rplus1 = shell_e.shape
    print(f"Loaded T_total = {T} frames, electron shell radius R = {Rplus1-1}.")
    print("Meta:", json.dumps(meta, indent=2))

    up_indices, down_indices = classify_hops_multi(dist_pe, seed_ids)
    print()
    print(f"Total up-hops (1 -> 2):   {len(up_indices)}")
    print(f"Total down-hops (2 -> 1): {len(down_indices)}")

    print("\nDetecting outward-moving packets around DOWN-hops (emission-like)...")
    down_tracks = detect_packet_tracks_for_hops(
        times=times,
        shell_e=shell_e,
        hop_indices=down_indices,
        hop_type="down",
        amp_threshold_factor=0.5,
        min_track_len=3,
        window_before=1,
        window_after=3,
    )

    print("\nDetecting inward-moving packets around UP-hops (absorption-like)...")
    up_tracks = detect_packet_tracks_for_hops(
        times=times,
        shell_e=shell_e,
        hop_indices=up_indices,
        hop_type="up",
        amp_threshold_factor=0.5,
        min_track_len=3,
        window_before=1,
        window_after=3,
    )

    n_up = len(up_indices)
    n_down = len(down_indices)
    print("\n=== Summary of packet detection (multi-seed) ===")
    print(f"Down-hops with outward-moving packet tracks: {len(down_tracks)} / {n_down}")
    print(f"Up-hops with inward-moving packet tracks:    {len(up_tracks)} / {n_up}")

    print_track_summary(down_tracks, label="Outward (down-hop / emission-like)")
    print_track_summary(up_tracks, label="Inward (up-hop / absorption-like)")

    out_fname = "photon_packet_tracks_multi.npz"
    save_tracks(out_fname, up_tracks=up_tracks, down_tracks=down_tracks)
    print(f"\nSaved tracks to {out_fname}")


if __name__ == "__main__":
    main()
