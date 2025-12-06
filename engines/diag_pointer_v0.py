#!/usr/bin/env python3
"""
diag_pointer_v0.py — Substrate v0.1
Threshold-free classical emergence probe.

Inputs:
    quench_output/timeseries.npz

Outputs:
    pointer_analysis.json
"""

import numpy as np, json, os

def main():
    data = np.load("quench_output/timeseries.npz")
    times = data["times"]
    n_t   = data["n_t"]   # (T, N)

    T,N = n_t.shape

    # vacuum baseline = average before quench at t=4
    mask = times <= 4.0
    n_vac = n_t[mask].mean(axis=0)

    # excess above vacuum
    excess = n_t - n_vac[None,:]
    excess[excess < 0] = 0

    total_exc = excess.sum(axis=1)
    fraction  = np.zeros_like(excess)
    for k in range(T):
        if total_exc[k] > 0:
            fraction[k] = excess[k] / total_exc[k]

    out = {
        "n_vac": n_vac.tolist(),
        "total_excess": total_exc.tolist(),
        "fraction_per_site": fraction.tolist(),
    }

    with open("quench_output/pointer_analysis.json","w") as f:
        json.dump(out, f, indent=2)

    print("Wrote quench_output/pointer_analysis.json")

if __name__ == "__main__":
    main()
