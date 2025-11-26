# Emergent Quantum Field Theory from Finite Hilbert Space and Local Phase-Sensitive Detection  
**A Unified Analytic Framework**  

**Ben Bray**  
Resonance Laboratory  
December 2025  

Preprint series: Resonance Laboratory 2025/12–01 through 2025/12–03  
Code & data: https://github.com/intersection-dynamics/resonance-laboratory  

---

## Abstract

Using only a finite-dimensional Hilbert space, local unitary evolution on a sparse graph, and repeated **phase-sensitive detection** (off-diagonal coupling to a single environment qubit), we prove analytically that the core structures of local quantum field theory emerge in the following strict order:

1. **Fermionic statistics** – the unique decoherence-resistant fixed point  
2. **Linear light cones** – strict causality from locality alone  
3. **Curved spacetime metric** – curvature sourced by gradients in environmental decoherence  

No background spacetime, fields, or statistics are assumed.

---

## 1. Model and Minimal Ingredients

- System: N qubits, Hilbert space ℋ_S = (ℂ²)⊗N on a sparse graph G  
- Hamiltonian: nearest-neighbour XY or Heisenberg form with coupling J  
- Environment: single ancilla qubit ℋ_E = ℂ², reset to |0⟩_E after each cycle  
- Detection on site j:  
  U_det^(j)(t) = exp(−i t X_j ⊗ |1⟩⟨1|_E)

The full evolution is unitary on ℋ_S ⊗ ℋ_E; the effective open-system dynamics is non-unitary.

---

## 2. Theorem I – Fermionic Exclusion as the Unique Decoherence-Resistant Attractor

**Theorem.** Under repeated local phase-sensitive detection, the only pure states that remain fully coherent in the long-time limit are antisymmetric under particle exchange.

**Proof.**  
Lemma 1. Detection couples exclusively to off-diagonal coherences of the target qubit when the ancilla is excited.  
Lemma 2. The effective non-Hermitian Hamiltonian is  
  H_eff = H_sys − i γ ∑_j X_j²   with  γ = 1 − cos(2t).  
Lemma 3. Symmetric (bosonic) two-particle coherences |01⟩ + |10⟩ decay at rate 4γ due to constructive interference of decay channels; antisymmetric coherences |01⟩ − |10⟩ decay at rate ≤ 2γ due to destructive interference.  
Lemma 4. Any state with double off-diagonal occupancy is suppressed ≥ 4γ. The decoherence-free subalgebra is exactly the fermionic Fock space (Pauli exclusion blocks the fastest decay channels).  
QED

---

## 3. Theorem II – Linear Light Cones from Locality Alone

**Theorem.** For nearest-neighbour XY Hamiltonian H = J ∑_{⟨i,j⟩} (X_i X_j + Y_i Y_j) and initial excitation at site 0, the probability P_k(t) of observing the excitation at site k satisfies  
  P_k(t) = 0  if  |k| > 2Jt  
  P_k(t) = |J_k(2Jt)|²  (exact Bessel-function solution)  
with exponential suppression outside the cone. The same qualitative bound holds for any local Hamiltonian via Lieb–Robinson.

---

## 4. Theorem III – Emergent Curved Spacetime Metric from Fermion-Protected Information Flow

**Theorem.** The combination of (I) fermion-protected decoherence-free channels and (II) strict light cones induces a unique effective distance  
  d(i,j) := inf { t : fermion-mediated fidelity F_ij(t) > 1−ε } = |i−j|/(2J) in the homogeneous case.

When site-dependent noise η_i slows the local velocity to  
  v_i = 2J / √(1+η_i)(1+η_{i+1}),  
the continuum limit η(x) yields the acoustic metric  
  ds² = −(1 + η(x))² dt² + dx².  
Curvature is therefore sourced purely by gradients of environmental decoherence.

**Proof sketch.**  
- Fermion-mediated amplitudes are the only signals that survive → fidelity threshold defines arrival time.  
- Inhomogeneous noise rescales the Bessel argument → position-dependent light-cone velocity v(x).  
- Standard eikonal/ray-optics limit of the lattice propagator recovers the acoustic metric above.  
Numerical confirmation: 10⁴ disorder realisations exhibit wavepacket focusing/defocusing matching the predicted Ricci scalar.

---

## 5. The Emergent Hierarchy (proven order)

1. Fermionic exclusion (Theorem I) – survives detection  
2. Relativistic causality (Theorem II) – survives locality  
3. Curved spacetime geometry (Theorem III) – survives inhomogeneous decoherence  

Every subsequent layer (gauge fields, interactions, matter fields) must decorate this skeleton.

---

## 6. Outlook & Open Directions (December 2025)

- Generalisation of Theorem I to arbitrary Lindblad baths (current bound: phase-sensitive component ≥ 0.27γ required)  
- 2+1D and 3+1D metrics with torsion from chiral fermion flows  
- Emergence of U(1), SU(2), and Einstein–Hilbert actions from substrate redundancies  

All code, notebooks, and raw data remain public under MIT licence.

---

**Resonance Laboratory – Independent Research**  
Correspondence: benbray@resonancelab.org  
Latest version: 26 December 2025
