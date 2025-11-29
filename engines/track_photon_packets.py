#!/usr/bin/env python3
"""
track_photon_packets.py

Goal:
  From photon_event_probe.npz, try to identify *coherent wavefronts* in
  Δphoton_score (electron-centered shells) that move outward / inward over
  several frames, and treat those as "photon packets".

We:

  - Load:
      times                  : (T,)
      dist_pe                : (T,)
      photon_shell_electron  : (T, R+1)
      meta_json

  - Build a baseline photon shell profile around the electron:
      baseline_e[r] = mean_t photon_shell_electron[t, r]

  - Define:
      Δ_shell_e[t, r] = photon_shell_electron[t, r] - baseline_e[r]

  - Identify hop frames:
      up-hops   : dist_pe 1 -> 2
      down-hops : dist_pe 2 -> 1

  - Around each hop at frame t0, look in a time window:
      window indices: t = t0-1 .. t0+3   (clipped to [0, T-1])

    For each t in window, find:
      r_max[t] = argmax_r Δ_shell_e[t, r]
      amp[t]   = Δ_shell_e[t, r_max[t]]

    For down-hops (candidate emission):
      We look for:
        * amp[t] > amp_threshold (positive bump),
        * radii r_max[t] monotone non-decreasing with at least one strict
          outward step (r increases),
        * at least min_len frames in the track.

      If satisfied, we record an outward-moving photon packet track.

    For up-hops (candidate absorption):
      Similarly, but with radii monotone non-increasing with at least one
      strict inward step (r decreases).

Outputs:

  - Prints summary:
      number of hops,
      number with outward/inward packets,
      examples of tracks.

  - Saves photon_packet_tracks.npz with arrays:
      times
      radii
      amplitudes
      hop_index
      hop_type (0 = up, 1 = down)

Use:

  python track_photon_packets.py

You should run photon_event_probe.py first so photon_event_probe.npz exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np


@dataclass
class PhotonPacketTrack:
    hop_type: Literal["up", "down"]  # "up" or "down"
    hop_frame: int                   # index t0 of the hop
    times: np.ndarray                # shape (L,)
    radii: np.ndarray                # shape (L,)
    amplitudes: np.ndarray           # shape (L,)


# ---------------------------------------------------------------------
# Data loading & hop classification
# ---------------------------------------------------------------------


def load_event_data(fname: str = "photon_event_probe.npz"):
    data = np.load(fname, allow_pickle=True)
    times = data["times"]  # (T,)
    dist_pe = data["dist_pe"]  # (T,)
    shell_e = data["photon_shell_electron"]  # (T, R+1)
    meta_json = data["meta_json"].item()
    meta = json.loads(meta_json)
    return times, dist_pe, shell_e, meta


def classify_hops(dist_pe: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Identify up-hops and down-hops from dist_pe[t].

    Framework:
      dist_pe[t] is typically in {1,2}.

      Up-hop:   1 -> 2 (electron moves outward)
      Down-hop: 2 -> 1 (electron moves inward)

    We look for t >= 1 with dist_pe[t-1] != dist_pe[t]:
      - 1 -> 2 => up-hop at t
      - 2 -> 1 => down-hop at t
    """
    T = len(dist_pe)
    up_indices: List[int] = []
    down_indices: List[int] = []
    for t in range(1, T):
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
# Packet detection
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
    For each hop at t0 in hop_indices, attempt to detect a coherent
    photon packet track in Δ_shell_e[t, r] around the electron.

    Steps:

      - Compute baseline_e[r] = mean_t shell_e[t, r]
      - Δ_shell_e[t, r] = shell_e[t, r] - baseline_e[r]

      - For each hop t0:
          * window t = t_start .. t_end where:
              t_start = max(0, t0 - window_before)
              t_end   = min(T-1, t0 + window_after)

          * For each t in window:
              - r_max[t] = argmax_r Δ_shell_e[t, r]
              - amp[t]   = Δ_shell_e[t, r_max[t]]

          * Keep only frames where amp[t] > amp_threshold, where:
              amp_threshold = amp_threshold_factor * std(Δ_shell_e)

          * Check monotonicity of r_max over surviving frames:
              - For "down" hops (emission-like):
                    radii non-decreasing & at least one strict increase
              - For "up" hops (absorption-like):
                    radii non-increasing & at least one strict decrease

          * If track length >= min_track_len and monotonicity condition holds,
            record PhotonPacketTrack.

    Returns:
      List[PhotonPacketTrack]
    """
    T, Rplus1 = shell_e.shape
    baseline_e = np.mean(shell_e, axis=0)                # (R+1,)
    delta_shell_e = shell_e - baseline_e[np.newaxis, :]  # (T, R+1)

    # Global amplitude threshold to avoid tracking noise
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
            delta_shell_t = delta_shell_e[t]  # (R+1,)
            r_max = int(np.argmax(delta_shell_t))
            amp = float(delta_shell_t[r_max])
            # filter by amplitude
            if amp > amp_threshold:
                radii.append(r_max)
                amps.append(amp)
                times_sel.append(float(times[t]))

        if len(radii) < min_track_len:
            continue

        radii_arr = np.asarray(radii, dtype=int)
        amps_arr = np.asarray(amps, dtype=float)
        times_arr = np.asarray(times_sel, dtype=float)

        # Monotonicity checks
        diffs = np.diff(radii_arr)
        if hop_type == "down":
            # Emission-like: radii move outward
            non_decreasing = np.all(diffs >= 0)
            has_strict_outward = np.any(diffs > 0)
            if not (non_decreasing and has_strict_outward):
                continue
        else:
            # hop_type == "up": absorption-like: radii move inward
            non_increasing = np.all(diffs <= 0)
            has_strict_inward = np.any(diffs < 0)
            if not (non_increasing and has_strict_inward):
                continue

        # If we get here, we consider it a coherent packet track
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
    """
    Save tracks into a single NPZ file for later plotting / analysis.
    """
    # We store them as ragged arrays via object dtype lists.
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
    """
    Print a summary of detected tracks and show a few example tracks.
    """
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
    print("Loading photon_event_probe.npz...")
    times, dist_pe, shell_e, meta = load_event_data()
    T, Rplus1 = shell_e.shape
    print(f"Loaded T = {T} frames, electron shell radius R = {Rplus1-1}.")
    print("Meta:", json.dumps(meta, indent=2))

    up_indices, down_indices = classify_hops(dist_pe)
    print()
    print(f"Total up-hops (1 -> 2):   {len(up_indices)}")
    print(f"Total down-hops (2 -> 1): {len(down_indices)}")

    # Detect photon packet tracks
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

    # Summaries
    n_up = len(up_indices)
    n_down = len(down_indices)
    print("\n=== Summary of packet detection ===")
    print(f"Down-hops with outward-moving packet tracks: {len(down_tracks)} / {n_down}")
    print(f"Up-hops with inward-moving packet tracks:    {len(up_tracks)} / {n_up}")

    print_track_summary(down_tracks, label="Outward (down-hop / emission-like)")
    print_track_summary(up_tracks, label="Inward (up-hop / absorption-like)")

    # Save tracks for plotting
    out_fname = "photon_packet_tracks.npz"
    save_tracks(out_fname, up_tracks=up_tracks, down_tracks=down_tracks)
    print(f"\nSaved tracks to {out_fname}")


if __name__ == "__main__":
    main()
