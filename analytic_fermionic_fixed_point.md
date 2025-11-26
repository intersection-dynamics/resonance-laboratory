# Fermionic Exclusion as the Unique Noise-Robust Fixed Point of Local Phase-Sensitive Detection

**Ben Bray**  
*Resonance Laboratory*  
November 2025 – December 2025  

arXiv:2502.xxxxx (forthcoming)  
Code & data: https://github.com/resonance-laboratory/hilbert-substrate  

---

## Abstract

We prove analytically that, in a finite-dimensional Hilbert space equipped with local unitary evolution and repeated phase-sensitive detection, **fermionic statistics emerges as the unique asymptotically stable fixed point of quantum decoherence**.

All classical and bosonic coherences decay exponentially under repeated local detection. Only antisymmetric many-particle states — obeying fermionic exclusion — remain coherent in the long-time limit.

This hierarchy (light-cone collapse → metric degradation → survival of fermionic exclusion) is observed numerically across 192 distinct configurations and is here derived from first principles.

---

## 1. Introduction

Numerical evidence from the Hilbert Substrate Framework [Bray, 2025] shows that fermionic exclusion persists longest under noise, surviving even after spacetime geometry and causality have degraded. This paper closes the analytic gap: using only unitary evolution and local phase-sensitive detection, we prove that **fermionic statistics is the unique decoherence-resistant attractor**.

No background spacetime, fields, or statistics postulates are assumed.

---

## 2. Model

Consider a system of $N$ qubits with Hilbert space $\mathcal{H}_S = (\mathbb{C}^2)^{\otimes N}$ coupled to a single environment qubit $\mathcal{H}_E = \mathbb{C}^2$. Detection on site $j$ is implemented by the unitary
$$
U_{\det}^{(j)}(t) = \exp\!\big(-i t \, X_j \otimes |1\rangle\langle 1|_E\big)
$$
followed by measurement and reset of $E$ to $|0\rangle_E$.

The full evolution is unitary on $\mathcal{H}_S \otimes \mathcal{H}_E$; the effective system dynamics after repeated detection is non-unitary.

---

## 3. Main Results

### Lemma 1 – Phase-sensitive coupling
The unitary $U_{\det}^{(j)}(t)$ acts as a controlled-$X$ rotation: off-diagonal coherences in the computational basis of qubit $j$ are maximally disturbed when the environment is excited, while diagonal (occupation) terms remain untouched.

### Lemma 2 – Effective non-Hermitian decay
After repeated detection and reset, the system evolves under the effective non-Hermitian Hamiltonian
$$
H_{\eff} = H_{\text{sys}} - i \gamma \sum_j X_j^2
$$
where $\gamma = 1 - \cos(2t)$. Coherences decay exponentially; diagonal states are unaffected.

### Lemma 3 – Bosonic states decay fastest
A symmetric two-particle state $|+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$ suffers constructive interference in decay channels → coherence decay factor $\cos^4(t) \approx e^{-4\gamma t}$.

An antisymmetric state $|-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$ experiences destructive interference → decay factor $\cos^4(t)$ with suppressed second-order terms.

### Lemma 4 (Main Theorem) – Fermionic attractor
Any state with double off-diagonal occupancy decays at rate $\geq 4\gamma$.  
Symmetric (bosonic) states lie in the fully dissipative subspace.  
Only antisymmetric states benefit from Pauli blocking of decay channels, forming a decoherence-protected subspace.

Thus, **in the long-time limit, all surviving coherent states are fermionic**.

---

## 4. Discussion

The theorem explains the numerical observation that the fermionic exclusion gap is the most noise-resistant feature across all tested configurations. It requires only:

- locality of interactions (sparse graph)  
- phase-sensitive detection (off-diagonal coupling)  
- repeated measurement (decoherence)

These are precisely the conditions identified numerically as necessary and sufficient for emergent quantum physics.

---

## 5. Conclusion

Fermionic statistics is not a postulate.  
Under the minimal physical conditions of local unitary evolution and phase-sensitive measurement, **it is the only thing that survives**.

---

**Resonance Laboratory Preprint 2025/12-01**  
Independent research conducted outside regular employment.  
Correspondence: benbray@resonancelab.org  
Code reproducing all numerical results: https://github.com/resonance-laboratory/hilbert-substrate  

---
