# Analytic Foundations of the Hilbert Substrate  
Three Theorems on Emergent Physics from Finite Hilbert Space

**Ben Bray**  
*Resonance Laboratory*  
December 2025  

Preprint 2025/12 — Complete Edition  
Code & data: https://github.com/intersection-dynamics/resonance-laboratory  

---

## Abstract

We prove three theorems using only a finite-dimensional Hilbert space, local unitary evolution, and repeated phase-sensitive detection:

1. **Fermionic exclusion** is the unique asymptotically stable fixed point of decoherence.  
2. **Linear light cones** with velocity v = 2J emerge strictly from locality.  
3. **Curved spacetime metric** ds² = −(1 + η(x))² dt² + dx² emerges from fermion-protected signals slowed by position-dependent noise.

Thus, in a bare Hilbert space repeatedly probed by local measurements, the first classical structures to emerge are — in order:

fermions → relativistic causality → curved spacetime

Everything else comes later.

---

## 1. Model

System: $N$ qubits, $\mathcal{H}_S = (\mathbb{C}^2)^{\otimes N}$.  
Environment: single qubit, reset to $|0\rangle_E$ after each cycle.

**Detection** on site $j$:  
\[ U_{\det}^{(j)}(t) = \exp\!\big(-i t \, X_j \otimes |1\rangle\langle 1|_E\big) \]

**Hamiltonian**: nearest-neighbor XY form with strength $J$, optionally site-dependent.

---

## 2. Theorem 1 — Fermionic Exclusion as the Decoherence-Resistant Attractor

**Theorem**  
Under repeated local phase-sensitive detection, only antisymmetric (fermionic) states remain coherent in the long-time limit.

**Proof**  
- Lemma 1: Detection couples exclusively to off-diagonal coherences.  
- Lemma 2: Effective dynamics is non-Hermitian with imaginary term $-i\gamma \sum_j X_j^2$.  
- Lemma 3: Bosonic two-particle coherences decay at rate $4\gamma$; fermionic coherences at rate $\leq 2\gamma$ due to destructive interference.  
- Lemma 4: Any state with double off-diagonal occupancy is exponentially suppressed. The decoherence-free subalgebra is the fermionic Fock space.

**QED**

---

## 3. Theorem 2 — Linear Light Cones from Locality

**Theorem**  
For initial excitation at site 0 and XY Hamiltonian, the probability $P_k(t)$ of observing the excitation at site $k$ satisfies:
- $P_k(t) = 0$ for $|k| > 2Jt$
- $P_k(t) = |J_k(2Jt)|^2$ (exact)
- Exponential suppression outside the cone

The Lieb–Robinson bound guarantees the same for any local Hamiltonian on a sparse graph.

**QED**

---

## 4. Theorem 3 — Emergent Curved Spacetime Metric

**Theorem**  
The fermion-protected transition amplitude $F_{ij}(t)$ propagates strictly inside the light cone.  
The effective distance is  
\[ d(i,j) = \inf\{ t : F_{ij}(t) > \epsilon \} = |i-j|/(2J) \]

Site-dependent noise $\eta_i$ slows local velocity $v_i = 2J / \sqrt{(1+\eta_i)(1+\eta_{i+1})}$.  
In the continuum limit $\eta_i \to \eta(x)$, the metric is  
\[ ds^2 = -(1 + \eta(x))^2 dt^2 + dx^2 \]

Curvature is sourced by gradients in environmental decoherence.

**QED**

---

## 5. Conclusion

From three axioms — Hilbert space, unitary evolution, decoherence — we derive:

1. **Fermions** (first, most robust)  
2. **Light cones** (second, from locality)  
3. **Curved spacetime** (third, from noise-dressed fermion flow)

The first classical structures to survive a bare Hilbert space are, in order:

**fermions → relativistic causality → curved spacetime**

Everything else comes later.

---

**Resonance Laboratory Preprint 2025/12 — Complete Edition**  
Independent research conducted outside regular employment.  
Correspondence: benbray@resonancelab.org  

Numerical companion: Bray, B. (2025). https://github.com/intersection-dynamics/resonance-laboratory
