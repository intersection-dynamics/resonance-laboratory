#!/usr/bin/env python3
"""
Substrate Stress Test (Aggressive)
==================================
Push the unified substrate framework until it bends
and map out where the current constructions stop
behaving like the intended "emergent SM toy model".

This script is intentionally *mean*:
- higher noise,
- stronger parameter sweeps,
- basic monotonicity checks,
- and more explicit "out of regime" flags.
"""

import numpy as np
from substrate import Substrate, GaugeField, QCD, WeakForce, MassField, SubstrateFramework


# ---------------------------------------------------------------------
# 1. Baseline: does the nominal configuration behave as advertised?
# ---------------------------------------------------------------------

def run_baseline():
    print("\n[BASELINE]")
    print("-" * 40)
    framework = SubstrateFramework(n_sites=6)
    results = framework.test_all(verbose=False)

    gauge_ok = results["gauge"]["no_flux"]["symmetric"] * results["gauge"]["with_flux"]["symmetric"] < 0
    qcd_ok   = results["qcd"]["baryon_RGB"]["casimir"] < 0.1
    weak_ok  = results["weak"]["left_coupling"] > 0 and results["weak"]["right_coupling"] == 0

    print(f"Gauge transmutation: {'✓' if gauge_ok else '✗'}")
    print(f"QCD confinement: {'✓' if qcd_ok else '✗'}")
    print(f"Parity violation: {'✓' if weak_ok else '✗'}")

    return results


# ---------------------------------------------------------------------
# 2. Parameter sweeps that *should* change nontrivially
# ---------------------------------------------------------------------

def test_parameter_sweep():
    """Vary key parameters and look for sensible trends, not just invariants."""
    print("\n[PARAMETER SWEEP]")
    print("-" * 40)

    # --- QCD: look at energy gaps vs confinement ---
    print("\nQCD confinement / energy gap ΔE = 3E_single - E_baryon:")
    confinements = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0]
    gaps = []

    for k in confinements:
        qcd = QCD(3)
        H = qcd.hamiltonian(mass=1.0, confinement=k)
        baryon = qcd.baryon_state()
        single = qcd.basis_state([1, 0, 0])  # single 'R' at one site

        E_b = float(np.real(np.vdot(baryon, H @ baryon)))
        E_s = float(np.real(np.vdot(single, H @ single)))
        dE  = 3.0 * E_s - E_b
        gaps.append(dE)

        C = float(np.real(np.vdot(baryon, qcd.casimir @ baryon)))
        print(f"  κ={k:5.1f}: E_b={E_b:7.3f}, E_s={E_s:7.3f}, ΔE={dE:7.3f}, C={C:.4f}")

    # simple monotonicity check: does ΔE tend to grow with κ?
    monotone = all(gaps[i+1] >= gaps[i] - 1e-6 for i in range(len(gaps)-1))
    print(f"  ΔE(κ) non-decreasing? {'✓' if monotone else '✗'}")

    # --- NEW: finer low-κ scan + critical κ* estimate ---
    print("\nQCD low-κ baryon favorability (ΔE = 3E_single - E_baryon):")
    low_kappa = [0.05, 0.10, 0.20, 0.30, 0.50]
    low_gaps = []

    print("   κ       ΔE     baryon favored?")
    for k in low_kappa:
        qcd = QCD(3)
        H = qcd.hamiltonian(mass=1.0, confinement=k)
        baryon = qcd.baryon_state()
        single = qcd.basis_state([1, 0, 0])

        E_b = float(np.real(np.vdot(baryon, H @ baryon)))
        E_s = float(np.real(np.vdot(single, H @ single)))
        dE  = 3.0 * E_s - E_b
        low_gaps.append(dE)

        favored = dE > 0.0  # ΔE>0 means 3 singles more expensive → baryon is favored
        flag = "✓" if favored else "✗"
        print(f"  {k:4.2f}   {dE:7.3f}        {flag}")

    # estimate κ* where ΔE crosses zero via linear interpolation
    k_star = None
    prev_k, prev_d = None, None
    for k, d in zip(low_kappa, low_gaps):
        if prev_k is None:
            prev_k, prev_d = k, d
            continue
        if prev_d < 0.0 and d > 0.0:
            # linear interpolation between (prev_k, prev_d) and (k, d)
            k_star = prev_k + (0.0 - prev_d) * (k - prev_k) / (d - prev_d)
            break
        prev_k, prev_d = k, d

    if k_star is not None:
        print(f"  estimated κ* ≈ {k_star:.3f} where ΔE crosses 0 (baryon turns energetically favored)")
    else:
        print("  no sign change in ΔE over sampled low-κ range (cannot estimate κ*)")

    # --- Weak: check linearity of left-handed matrix element vs g ---
    print("\nWeak coupling: left/right matrix elements vs g")
    couplings = [0.1, 0.25, 0.5, 1.0, 2.0]
    L_values = []
    R_values = []

    for g in couplings:
        weak = WeakForce(2)
        H = weak.hamiltonian(coupling=g)
        L = float(np.real(np.vdot(
            weak.basis_state(['dL', 'uL']),
            H @ weak.basis_state(['uL', 'dL'])
        )))
        R = float(np.real(np.vdot(
            weak.basis_state(['dR', 'uR']),
            H @ weak.basis_state(['uR', 'dR'])
        )))
        L_values.append(L)
        R_values.append(abs(R))
        print(f"  g={g:4.2f}: L={L:7.4f}, R={R:7.4f}")

    # Fit L ≈ a * g, check residual
    L_values = np.array(L_values)
    couplings_arr = np.array(couplings)
    a_fit, _, _, _ = np.linalg.lstsq(couplings_arr.reshape(-1, 1), L_values, rcond=None)
    L_fit = a_fit[0] * couplings_arr
    residual = float(np.max(np.abs(L_values - L_fit)))
    R_max = float(np.max(R_values))

    print(f"  L ~ a*g linear fit residual: {residual:.3e}")
    print(f"  max |R| over sweep: {R_max:.3e}")
    linear_ok = residual < 1e-3
    chiral_ok = R_max < 1e-6
    print(f"  linear-in-g left channel? {'✓' if linear_ok else '✗'}")
    print(f"  right-handed still ~0? {'✓' if chiral_ok else '✗'}")


# ---------------------------------------------------------------------
# 3. Aggressive noise injection
# ---------------------------------------------------------------------

def test_noise():
    """Add *strong* noise, see where qualitative patterns die."""
    print("\n[NOISE INJECTION]")
    print("-" * 40)

    # push well beyond the previous 0.2 upper bound
    noise_levels = [0.00, 0.01, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00]
    n_trials = 20

    for noise in noise_levels:
        gauge_ok = 0
        qcd_ok   = 0
        weak_ok  = 0

        for seed in range(n_trials):
            np.random.seed(seed)

            # --- Gauge test: sign flip persists with nontrivial magnitude ---
            try:
                gauge = GaugeField(4)
                P = gauge.exchange_operator(0, 1)

                psi_02 = gauge.basis_state([1, 0, 1, 0], [0, 0, 0])
                psi_12 = gauge.basis_state([0, 1, 1, 0], [0, 0, 0])
                psi_sym = (psi_02 + psi_12) / np.sqrt(2)
                psi_sym += noise * np.random.randn(gauge.dim)
                psi_sym /= np.linalg.norm(psi_sym)

                psi_02f = gauge.basis_state([1, 0, 1, 0], [1, 0, 0])
                psi_12f = gauge.basis_state([0, 1, 1, 0], [1, 0, 0])
                psi_sym_f = (psi_02f + psi_12f) / np.sqrt(2)
                psi_sym_f += noise * np.random.randn(gauge.dim)
                psi_sym_f /= np.linalg.norm(psi_sym_f)

                ex1 = float(np.real(np.vdot(psi_sym, P @ psi_sym)))
                ex2 = float(np.real(np.vdot(psi_sym_f, (P @ gauge.link_phase[0]) @ psi_sym_f)))

                if (ex1 * ex2 < 0) and (abs(ex1) > 0.05) and (abs(ex2) > 0.05):
                    gauge_ok += 1
            except Exception:
                pass

            # --- QCD test: baryon still energetically favored & color-neutral-ish ---
            try:
                qcd = QCD(3)
                H = qcd.hamiltonian(mass=1.0, confinement=5.0)
                dim = qcd.dim
                noise_H = noise * np.random.randn(dim, dim)
                noise_H = (noise_H + noise_H.T) / 2.0
                H_noisy = H + noise_H

                baryon = qcd.baryon_state()
                single = qcd.basis_state([1, 0, 0])

                E_b = float(np.real(np.vdot(baryon, H_noisy @ baryon)))
                E_s = float(np.real(np.vdot(single, H_noisy @ single)))
                C_b = float(np.real(np.vdot(baryon, qcd.casimir @ baryon)))

                if (E_b < 3.0 * E_s) and (abs(C_b) < 0.2):
                    qcd_ok += 1
            except Exception:
                pass

            # --- Weak test: still mostly left-handed ---
            try:
                weak = WeakForce(2)
                H = weak.hamiltonian(coupling=1.0)
                dim_w = weak.dim
                noise_H = noise * np.random.randn(dim_w, dim_w)
                noise_H = (noise_H + noise_H.T) / 2.0
                H_noisy = H + noise_H

                psi_L_in = weak.basis_state(['uL', 'dL'])
                psi_L_out = weak.basis_state(['dL', 'uL'])

                psi_R_in = weak.basis_state(['uR', 'dR'])
                psi_R_out = weak.basis_state(['dR', 'uR'])

                L = abs(float(np.real(np.vdot(psi_L_out, H_noisy @ psi_L_in))))
                R = abs(float(np.real(np.vdot(psi_R_out, H_noisy @ psi_R_in))))

                if L > 2.0 * R and L > 0.05:
                    weak_ok += 1
            except Exception:
                pass

        print(f"Noise={noise:4.2f}: gauge={gauge_ok}/{n_trials}, qcd={qcd_ok}/{n_trials}, weak={weak_ok}/{n_trials}")


# ---------------------------------------------------------------------
# 4. Scale testing: where representations change
# ---------------------------------------------------------------------

def test_scale():
    """Increase size until failure or representation change."""
    print("\n[SCALE TESTING]")
    print("-" * 40)

    print("\nSubstrate (full vs single-excitation):")
    for n in [4, 6, 8, 10, 12, 14]:
        try:
            use_full = (n <= 10)
            sub = Substrate(n, use_full_space=use_full)
            H = sub.hopping_hamiltonian()
            psi = sub.excitation(0)
            psi_t = sub.evolve(psi, H, t=5.0)
            approx_regime = "full" if use_full else "single-excitation"
            print(f"  n={n:2d}: dim={sub.dim:6d}, regime={approx_regime:>16s}, ipr={sub.ipr(psi_t):.4f} ✓")
        except MemoryError:
            print(f"  n={n:2d}: MemoryError ✗ (beyond feasible Hilbert space)")
            break
        except Exception as e:
            print(f"  n={n:2d}: {e} ✗")

    print("\nQCD scaling:")
    for n in [2, 3, 4, 5]:
        try:
            qcd = QCD(n)
            H = qcd.hamiltonian()
            evals = np.linalg.eigvalsh(H)
            print(f"  n={n}: dim={qcd.dim:5d}, E0={evals[0]:.3f} ✓")
        except MemoryError:
            print(f"  n={n}: MemoryError ✗")
            break
        except Exception as e:
            print(f"  n={n}: {e} ✗")


# ---------------------------------------------------------------------
# 5. Edge cases and explicit "out of applicability"
# ---------------------------------------------------------------------

def test_edge_cases():
    """Pathological inputs and 'out of regime' cases."""
    print("\n[EDGE CASES]")
    print("-" * 40)

    cases = [
        ("zero confinement",    lambda: QCD(3).hamiltonian(confinement=0.0)),
        ("huge confinement",    lambda: QCD(3).hamiltonian(confinement=1000.0)),
        ("zero weak coupling",  lambda: WeakForce(2).hamiltonian(coupling=0.0)),
        ("negative hopping",    lambda: Substrate(4).hopping_hamiltonian(t=-1.0)),
        ("single site",         lambda: Substrate(1)),
        ("two sites",           lambda: Substrate(2).hopping_hamiltonian()),
    ]

    for name, func in cases:
        try:
            _ = func()
            print(f"  {name}: ✓")
        except Exception as e:
            print(f"  {name}: ✗ ({e})")


# ---------------------------------------------------------------------
# 6. QCD stability basin as a little phase diagram
# ---------------------------------------------------------------------

def test_stability_basin():
    """Map (mass, confinement) region where baryon looks QCD-like."""
    print("\n[STABILITY BASIN]")
    print("-" * 40)
    print("\nQCD: mass × confinement")
    masses = [0.1, 0.5, 1.0, 2.0]
    confinements = [1.0, 5.0, 10.0, 20.0]

    header = "       κ→" + "".join([f"{k:6.0f}" for k in confinements])
    print(header)
    print("mass↓")

    for m in masses:
        row_flags = []
        for k in confinements:
            qcd = QCD(3)
            H = qcd.hamiltonian(mass=m, confinement=k)
            baryon = qcd.baryon_state()
            single = qcd.basis_state([1, 0, 0])

            E_b = float(np.real(np.vdot(baryon, H @ baryon)))
            E_s = float(np.real(np.vdot(single, H @ single)))
            C_b = float(np.real(np.vdot(baryon, qcd.casimir @ baryon)))

            ok = (E_b < 3.0 * E_s) and (abs(C_b) < 0.1)
            row_flags.append("✓" if ok else "✗")
        print(f" {m:4.1f}  ", "    ".join(row_flags))


# ---------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------

def main():
    print("=" * 50)
    print("SUBSTRATE STRESS TESTS (AGGRESSIVE)")
    print("=" * 50)

    run_baseline()
    test_parameter_sweep()
    test_noise()
    test_scale()
    test_edge_cases()
    test_stability_basin()

    print("\n" + "=" * 50)
    print("STRESS TESTS COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
