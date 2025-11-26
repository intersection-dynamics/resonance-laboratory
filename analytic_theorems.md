# Two Theorems on Emergent Physics from Finite Hilbert Space

**Ben Bray**  
*Resonance Laboratory*  
December 2025  

Preprint 2025/12-02  
Code & data: https://github.com/intersection-dynamics/resonance-laboratory  

---

## Abstract

We prove two analytic results using only a finite-dimensional Hilbert space, local unitary evolution, and repeated phase-sensitive detection:

1. **Fermionic exclusion** is the unique asymptotically stable fixed point of decoherence.  
   All classical and bosonic coherences decay exponentially; only antisymmetric states survive.

2. **Linear light cones** with velocity $v = 2J$ emerge strictly from locality, with zero probability outside $|x| > vt$ and exponential suppression of leakage.

Numerical evidence from 192 configurations (Bray, 2025) shows the same hierarchy under noise: light cones collapse first, spacetime metric second, fermionic exclusion last.

These results require **no assumed spacetime, fields, or statistics**.

---

## 1. Introduction

The Hilbert Substrate Framework investigates whether key structures of local quantum field theory — causality, geometry, and fermionic statistics — can emerge from minimal ingredients: a finite-dimensional Hilbert space, unitary evolution on a sparse graph, and environmentally induced decoherence.

This note presents analytic proofs of two central numerical observations.

---

## 2. Model

System: $N$ qubits, $\mathcal{H}_S = (\mathbb{C}^2)^{\otimes N}$.  
Environment: single qubit $\mathcal{H}_E = \mathbb{C}^2$, reset to $|0\rangle_E$ after each cycle.

**Detection** on site $j$:  
\[ U_{\det}^{(j)}(t) = \exp\!\big(-i t \, X_j \otimes |1\rangle\langle 1|_E\big) \]

**Hamiltonian**: nearest-neighbor XY or Heisenberg form with strength $J$.

---

## 3. Theorem 1 — Fermionic Exclusion as the Decoherence-Resistant Attractor

**Theorem**  
Under repeated local phase-sensitive detection, the only pure states that remain coherent in the long-time limit are antisymmetric under particle exchange.

**Proof (four lemmas)**

**Lemma 1** — Detection couples exclusively to off-diagonal coherences when the environment is excited.

**Lemma 2** — The effective dynamics is non-Hermitian:  
\[ H_{\eff} = H_{\text{sys}} - i \gamma \sum_j X_j^2, \quad \gamma = 1 - \cos(2t) \]

**Lemma 3** — Symmetric (bosonic) two-particle coherences decay at rate $4\gamma$; antisymmetric (fermionic) coherences decay at rate $\leq 2\gamma$ due to destructive interference in decay channels.

**Lemma 4** — Any state with double off-diagonal occupancy is exponentially suppressed. The decoherence-free subalgebra is exactly the fermionic Fock space.

**QED**

---

## 4. Theorem 2 — Linear Light Cones from Locality

**Theorem**  
For initial excitation at site 0 and nearest-neighbor XY Hamiltonian $H = J \sum_{\langle i,j\rangle} (X_i X_j + Y_i Y_j)$, the probability $P_k(t)$ of observing the excitation at site $k$ satisfies:
- $P_k(t) = 0$ for $|k| > 2Jt$
- $P_k(t) = |J_k(2Jt)|^2$ (exact Bessel function solution)
- Exponential suppression outside the cone: $|J_k(x)| \sim e^{-\mu(|k|-x)}$ for $x < |k|$

The Lieb–Robinson bound guarantees the same qualitative result for any local Hamiltonian on a sparse graph.

**QED**

---

## 5. Discussion

The two theorems explain the observed noise hierarchy:

- Fermionic exclusion survives longest (Theorem 1)  
- Light cones and causality emerge strictly from locality (Theorem 2)  
- Everything else (metric, bosonic statistics) requires both

Thus, in a bare Hilbert space repeatedly probed by local, phase-sensitive measurements:

> **The first classical structure to emerge is fermionic statistics.  
> The second is relativistic causality.  
> Everything else comes later.**

---

## References

- Bray, B. (2025). Numerical evidence from the Hilbert Substrate Framework.  
  https://github.com/intersection-dynamics/resonance-laboratory
- Lieb, E., & Robinson, D. (1972). The finite group velocity of quantum spin systems.
- Hastings, M. (2004). Lieb-Schultz-Mattis in higher dimensions.

---

**Resonance Laboratory Preprint 2025/12-02**  
Independent research. Correspondence: benbray@resonancelab.org
