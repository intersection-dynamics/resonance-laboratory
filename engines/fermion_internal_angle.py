"""
fermion_internal_angle.py

Extract the internal-state chain ordering and assign a U(1)-like
angle coordinate θ(k) to each internal state.

This script reads:
    run_root/analysis_transition_graph/adjacency.npy
    run_root/analysis_transition_graph/state_index.json

And writes:
    run_root/analysis_transition_graph/internal_angle.json
    run_root/analysis_transition_graph/internal_angle.npz

Formula for angles (for a path of length N):
    θ_k = 2π * k / (N-1)

Usage:

    python fermion_internal_angle.py --run-root \
        outputs\\precipitating_event\\YYYYMMDD_HHMMSS_your_run_tag

"""

from __future__ import annotations
import argparse
import json
import os
import numpy as np
import math


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_chain_order(adj: np.ndarray) -> list[int]:
    """
    Given a weighted adjacency matrix for a known simple path/cycle
    of internal states, reconstruct the ordering.

    We look for nodes with in_degree == 0 (start), then follow the
    unique outgoing edge at each step.
    """
    n = adj.shape[0]
    # Convert weights to binary edges
    A = (adj > 0).astype(int)

    # Compute indegree & outdegree
    indeg = A.sum(axis=0)
    outdeg = A.sum(axis=1)

    # Identify start node for an open chain
    # (node with indegree == 0)
    starts = np.where(indeg == 0)[0]
    if len(starts) == 1:
        start = int(starts[0])
        # Build chain
        chain = [start]
        current = start
        visited = set(chain)

        while True:
            # find successor j with A[current,j] == 1
            succs = np.where(A[current] == 1)[0]
            if len(succs) == 0:
                break
            if len(succs) > 1:
                raise RuntimeError("More than one successor detected; not a simple chain.")
            nxt = int(succs[0])
            if nxt in visited:
                break
            chain.append(nxt)
            visited.add(nxt)
            current = nxt

        return chain

    # If no unique start, maybe it's a cycle
    if len(starts) == 0:
        # pick any node and follow until it loops
        start = 0
        chain = [start]
        visited = {start}
        current = start
        while True:
            succs = np.where(A[current] == 1)[0]
            if len(succs) == 0:
                break
            if len(succs) > 1:
                raise RuntimeError("More than one successor detected in cycle.")
            nxt = int(succs[0])
            if nxt in visited:
                # Cycle detected — stop before repetition
                break
            chain.append(nxt)
            visited.add(nxt)
            current = nxt
        return chain

    raise RuntimeError("Unexpected graph structure — multiple starts but no clear chain.")


def main():
    parser = argparse.ArgumentParser(
        description="Assign internal-state angle θ(k) from transition graph."
    )
    parser.add_argument(
        "--run-root",
        required=True,
        help="Run root for a precipitating_event simulation.",
    )
    args = parser.parse_args()

    run_root = os.path.abspath(args.run_root)
    tdir = os.path.join(run_root, "analysis_transition_graph")

    adj_path = os.path.join(tdir, "adjacency.npy")
    idx_path = os.path.join(tdir, "state_index.json")

    if not os.path.exists(adj_path):
        raise FileNotFoundError(adj_path)
    if not os.path.exists(idx_path):
        raise FileNotFoundError(idx_path)

    print("===============================================")
    print("  Fermion Internal Angle Embedding")
    print("===============================================")
    print(f"run_root: {run_root}")
    print("-----------------------------------------------")
    print(f"Loading adjacency: {adj_path}")
    A = np.load(adj_path)
    n_states = A.shape[0]
    print(f"Loaded adjacency for {n_states} internal states.")

    print(f"Loading state index: {idx_path}")
    state_info = load_json(idx_path)  # dict { "0": {...}, "1": {...}, ... }

    # ----------------------------------------------------------
    # 1. Determine chain order
    # ----------------------------------------------------------
    chain = find_chain_order(A)
    L = len(chain)
    print(f"Chain length detected: {L}")

    if L != n_states:
        print(f"WARNING: Chain covers {L}/{n_states} states. Proceeding anyway.")

    # ----------------------------------------------------------
    # 2. Assign θ(k)
    # ----------------------------------------------------------
    if L == 1:
        theta_vals = {chain[0]: 0.0}
    else:
        theta_vals = {
            chain[k]: 2.0 * math.pi * (k / (L - 1))
            for k in range(L)
        }

    # ----------------------------------------------------------
    # 3. Prepare output structures
    # ----------------------------------------------------------
    out_json_list = []
    for i in range(n_states):
        st = state_info[str(i)]
        out_json_list.append(
            {
                "idx": i,
                "state_id": st["state_id"],
                "pattern_bits": st["pattern_bits"],
                "center_site": st["center_site"],
                "theta": theta_vals.get(i, None),
            }
        )

    out_json_path = os.path.join(tdir, "internal_angle.json")
    print(f"Writing: {out_json_path}")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_json_list, f, indent=2)

    out_npz_path = os.path.join(tdir, "internal_angle.npz")
    print(f"Writing: {out_npz_path}")
    np.savez_compressed(
        out_npz_path,
        idx=np.array([x["idx"] for x in out_json_list], dtype=int),
        state_id=np.array([x["state_id"] for x in out_json_list], dtype=int),
        center_site=np.array([x["center_site"] for x in out_json_list], dtype=int),
        theta=np.array([x["theta"] for x in out_json_list], dtype=float),
        pattern_bits=np.array([x["pattern_bits"] for x in out_json_list], dtype=object),
    )

    print("Done.")
    print("You now have θ(k) for every internal state.")


if __name__ == "__main__":
    main()
