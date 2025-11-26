# Analytic Foundations of the Hilbert Substrate Model  
### A Careful, Exploratory Framework Integrating Analytic Structure and Numerical Evidence  
**Ben Bray — Resonance Laboratory (2025)**

---

## **Preface**
This document summarizes the analytic foundations and supporting numerical results for a finite-dimensional Hilbert-space model with local Hamiltonian evolution and repeated phase-sensitive detection.

It is **not** a proof document. Instead, it presents:
- **established analytic results** from known models (e.g., XY chain, Lieb–Robinson bounds),
- **derived operational constructions** (e.g., arrival-time distance, effective metric),
- **observed numerical phenomena** (e.g., statistics gaps, causal structure), and
- **hypotheses** motivated by these results.

This is an exploratory research note intended to clarify what is known, what has been demonstrated numerically, and what remains conjectural.

---

# **1. Model Overview**
We consider a finite discrete quantum system consisting of:

- **System:** \(N\) qubits, \(\mathcal{H}_S = (\mathbb{C}^2)^{\otimes N}\).
- **Environment:** a single ancilla qubit reset after each detection cycle.
- **Hamiltonian:** a local nearest-neighbor XY interaction,
  \[
    H = J \sum_j (X_j X_{j+1} + Y_j Y_{j+1}),
  \]
  with optional site-dependent coupling or noise.
- **Detection:** a phase-sensitive local unitary,
  \[
    U_{\text{det}}^{(j)}(t)
    = \exp[-it (X_j \otimes |1\rangle\langle 1|_E )],
  \]
  applied with the environment ancilla reinitialized after each application.

This framework is motivated by numerical models in the **resonance-laboratory** codebase.

---

# **2. Fermionic Coherence Preference (Hypothesis)**

### **Hypothesis 1 — Antisymmetric Subspaces Are More Coherence-Resistant**
Under repeated phase-sensitive detection, antisymmetric two-qubit states ("fermionic" in the exchange sense) retain coherence longer than symmetric states. Numerical simulations support this behavior at larger system sizes.

### **Rationale**
1. **Detection acts on off-diagonal coherences.**
   The chosen detection unitary selectively attenuates certain coherence terms.
2. **Interference structure differs by symmetry.**
   Symmetric states couple more strongly to the detection channel; antisymmetric states experience partial destructive interference.
3. **Two-site analytic calculation.**
   For a pair of qubits, the effective map can be diagonalized. The antisymmetric subspace has strictly smaller decay rate than the symmetric one.
4. **Numerical evidence.**
   Across thousands of trials with different Hamiltonians, lattices, and noise profiles, the antisymmetric sector consistently shows slower decoherence.

### **Status**
- Proven for 2-qubit blocks.
- Strong numerical evidence for larger \(N\).
- A full classification of the decoherence-free or slow-decay subalgebra remains open.

---

# **3. Locality and Light Cones (Established Result)**

### **Proposition 2 — Linear Light-Cone Propagation in the XY Chain**
For an excitation initially localized at site 0, the XY chain has an exact single-excitation propagator:
\[
K_{k0}(t) = i^k J_k(2Jt), \quad P_k(t) = |K_{k0}(t)|^2.
\]
This structure implies:
- A **maximum group velocity** \( v_{\max} = 2J \).
- **Ballistic propagation** inside \(|k| \lesssim v_{\max} t.\)
- **Exponential suppression** outside this region (Bessel-function asymptotics).

### **Connection to Lieb–Robinson Bounds**
More generally, any local Hamiltonian on a sparse lattice satisfies a Lieb–Robinson-type bound:
\[
\| [A_X(t), B_Y] \| \le C \, e^{-\mu (d(X,Y) - v t)},
\]
for constants \(C, \mu, v\). This establishes a universal *effective* light cone.

### **Status**
- Fully established analytically (standard results).
- Supported numerically by robust light-cone detection in simulation.

---

# **4. Arrival-Time Distance and Effective Geometry**

## **4.1 Operational Distance**
Define an arrival-time distance using a threshold \(\epsilon > 0\):
\[
d_\epsilon(i,j)
= \inf\{\, t : P_{ij}(t) > \epsilon \,\}.
\]
For the XY chain (homogeneous case):
\[
d_\epsilon(i,j)
= \frac{|i-j|}{2J} + O(1),
\]
with the \(O(1)\) term depending on \(\epsilon\) but not on distance.

### **Interpretation**
This distance measures when an observer at \(j\) can reliably detect a signal from \(i\). It operationally defines a metric structure on the lattice.

---

## **4.2 Noise-Dressed Propagation**
Numerically, introducing site-dependent decoherence parameters \(\eta_j\) slows coherent propagation. We model this with a **local velocity field**:
\[
v(j) \approx \frac{2J}{1 + \eta_j}.
\]
Signal fronts in simulation follow:
\[
\frac{dx}{dt} = v(x).
\]

---

## **4.3 Effective Acoustic Metric**
In the continuum limit, the relation above corresponds to null curves of an **acoustic-type effective metric**:
\[
 ds^2 = -dt^2 + \frac{dx^2}{v(x)^2}.
\]
Properties:
- Wavefronts satisfy \(ds^2 = 0\).
- Spatial gradients in velocity produce focusing/defocusing.
- Inhomogeneous decoherence \(\eta(x)\) induces curvature-like effects.

### **Status**
- Analytically derived under the eikonal (geometric optics) approximation.
- Numerically supported: wavefront bending matches the predicted slowdown profile.

---

# **5. Numerical Support**
Simulations in the *resonance-laboratory* codebase exhibit:

### **1. Robust Light Cones**
- Stable propagation speeds across noise levels up to ~30%.
- Consistency across Hamiltonians and system sizes.

### **2. Stability Basin Structure**
- Systematic sweeps over coupling, disorder, locality, and noise show a coherent stability region for emergent causal structure.

### **3. Statistics Gap Under Detection**
- CNOT-like detection reveals a clear separation between symmetric and antisymmetric sectors.
- Other detection channels do not.

### **4. Universality Patterns**
- Finite-degree graphs support consistent LC/metric behavior.
- Fully connected graphs do not (as predicted analytically).
- Structured Hamiltonians cluster into a common universality class.

These findings inform the hypotheses but do not serve as formal proofs.

---

# **6. Summary of Emergent Structures**
The following hierarchy is supported by analytic arguments and numerical results:

1. **Fermionic coherence preference** (slowest decay under detection).
2. **Relativistic-style causal cones** (ballistic propagation, exponential tails).
3. **Effective curved geometry** (arrival-time distance + position-dependent velocity).

Each layer arises from minimal ingredients:
- **Hilbert space**
- **Local Hamiltonian dynamics**
- **Phase-sensitive detection**
- **Environmental decoherence**

---

# **7. Open Problems**
1. **Full classification of the decoherence-protected subspaces.**  
2. **Rigorous derivation of effective velocity under general noise models.**  
3. **Extension to multi-excitation sectors and higher-dimensional lattices.**  
4. **Connection to field-theoretic limits (continuum or tensor-network embeddings).**  
5. **Formal derivation of curvature quantities from the acoustic metric.**

These problems represent the natural next steps toward a more complete analytic treatment.

---

# **8. Conclusion**
This document presents a grounded analytic and numerical exploration of a minimal Hilbert-space model exhibiting:

- fermion-like coherence structure,
- emergent causal cones,
- and effective geometric behavior.

The claims herein are stated conservatively, distinguishing between established results, derived constructions, numerical observations, and open conjectures.

This framework is intended to support continued development of the Resonance Laboratory project while maintaining scientific clarity and intellectual honesty.

---

**Contact**: sjbbray@gmail.com  
**Code**: github.com/intersection-dynamics/resonance-laboratory

