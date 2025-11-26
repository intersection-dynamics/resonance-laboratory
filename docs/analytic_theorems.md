# Analytic Foundations of the Hilbert Substrate Model

### **A Hybrid-Style Research Document — Accessible, Careful, and Academically Disciplined**

**Ben Bray — Resonance Laboratory (2025)**

This document provides the analytic framework behind the **Hilbert Substrate Model**. It blends readability with strict separation between:

* **Established analytic results**
* **Derived constructions**
* **Numerical observations**
* **Conjectures**
* **Open problems**

It is written as a reference document for the GitHub repository and for anyone studying the emergent causal, statistical, and geometric structure found in these simulations.

---

# **0. Preface and Scope**

This is **not** a formal proof paper. It is a disciplined research note that:

* states results precisely,
* cites known theorems responsibly,
* distinguishes analytic reasoning from numerical evidence,
* and organizes the project’s intellectual structure.

It is intended for readers familiar with quantum information, condensed matter, and emergent-structure models.

---

# **1. The Hilbert Substrate Model**

We consider a finite-dimensional quantum system defined by:

### **System**

* (N) qubits
* Hilbert space (\mathcal{H}_S = (\mathbb{C}^2)^{\otimes N})

### **Environment**

* A single ancilla qubit, reset to (|0\rangle) after each detection cycle.

### **Hamiltonians**

The model supports:

* XY chain
* Heisenberg chain
* Hopping/XX models
* Random local Hamiltonians

All Hamiltonians are **finite-degree, local**, and of the general form:
[
H = \sum_{(i,j)\in E} h_{ij},
]
with (h_{ij}) acting on nearest neighbors.

### **Detection Operator**

A phase-sensitive detection applied to site (j):
[
U_{\text{det}}^{(j)}(t) = \exp\big[-it (X_j \otimes |1\rangle\langle1|)\big].
]
The ancilla is traced out and reset after each cycle.

### **Noise Model**

Independent Lindblad noise is applied on each qubit:
[
L_j = \sqrt{\gamma_m},\sigma^-_j + \sqrt{\gamma_p},\sigma^+*j + \sqrt{\gamma*\phi},X_j.
]
This combines:

* amplitude decay
* excitation
* dephasing in the X-basis

All numerical experiments incorporate this noise.

---

# **2. Fermionic Coherence Preference**

This section states what is **proven**, what is **derived**, and what is **conjectured**.

---

## **2.1 Proposition — Two-Qubit Fermionic Coherence Preference (Analytic)**

**Proposition.**
For a two-qubit system evolving under any combination of local XY/Heisenberg interactions and the Lindbladian operators:
[
L_j = \sqrt{\gamma_m},\sigma^-_j + \sqrt{\gamma_p},\sigma^+*j + \sqrt{\gamma*\phi},X_j,
]
the induced channel after one detection cycle has:

* a **unique slowest-decaying coherent eigenvector**,
* supported in the **antisymmetric Bell subspace**.

### Sketch of Derivation

1. Write the two-qubit Kraus map for one cycle.
2. Diagonalize the superoperator in the Pauli basis.
3. Symmetric and antisymmetric subspaces split under X-dephasing.
4. The antisymmetric coherence eigenvalue is strictly closer to the unit circle.

### Consequence

The antisymmetric sector has **strictly lower decoherence rate** than any symmetric sector state.

*(Figure 1: Two-qubit channel eigenvalue spectrum)*

---

## **2.2 Conjecture — Fermionic Uniqueness Threshold for N Qubits**

### **Definition (Dephasing Ratio)**

Let:
[
\Gamma = \max(\gamma_m,\gamma_p), \qquad \lambda = \frac{\gamma_\phi}{\Gamma}.
]

### **Conjecture.**

There exists a universal threshold (\lambda^* \approx 0.27) such that:

1. **If (\lambda > \lambda^*):**
   The slowest-decaying coherent subspace for all (N) is (up to tensor factors) the **antisymmetric wedge**:
   [
   \wedge^1 \mathbb{C}^2 \oplus \wedge^2 \mathbb{C}^2 \oplus \cdots.
   ]

2. **If (\lambda < \lambda^*):**
   No decoherence-protected subspace persists beyond (O(1)) cycles.

### **Numerical Evidence**

* Universality scans for (N\le8) give a stable threshold (0.27\pm0.03).
* Threshold insensitive to topology (chain/ring/square/honeycomb/random).
* Antisymmetric gap: 2–4× coherence lifetime improvement above threshold.

*(Figure 2: Threshold extraction across 192 configurations)*

---

## **2.3 Detection-Channel Landscape (Numerical Classification)**

Generalized detection unitaries parameterized by two real parameters,
[
U(\alpha,\beta) = \exp[-i (\alpha X + \beta ZZ) \otimes |1\rangle\langle1|],
]
selectively preserve different symmetry sectors. A full sweep over the (\alpha,\beta) triangle reveals:

* **CNOT-like detectors (\alpha \to 1, \beta \to 0)** preserve *fermionic* coherence (antisymmetric combinations).
* **ZZ-like detectors (\beta \to 1, \alpha \to 0)** preserve *bosonic/local* coherence (symmetric combinations).
* The diagonal band (\beta \approx 1 - \alpha) shows negligible statistics discrimination.

### Evidence from Detection Landscapes

![Detection landscape over (α, β) parameter space, showing statistics gap, local (bosonic) fidelity, and superposition (fermionic) fidelity.](images/detection_landscape.png)

* CNOT-like detectors preserve antisymmetric superpositions.
* ZZ-like detectors destroy them.

### Spin–Statistics Alignment

![Spin–statistics relationship: final fidelities for singlet/triplet and related states.](images/spin_statistics.png)

* Triplet states (s=1) remain near full fidelity under ZZ-like detection.
* Singlet states (s=0) retain fidelity preferentially under CNOT-like detection.

This provides a numerical classification of detector-induced symmetry selection, complementing the analytic two-qubit proposition and supporting the N-body conjecture.

# **3. Causal Structure — Light Cones**

Local Hamiltonians generate finite-speed propagation.

---

## **3.1 Proposition — XY Chain Propagator (Exact Result)**

For an excitation initially at site 0 in the XY chain:
[
K_{k0}(t) = i^k J_k(2Jt), \qquad P_k(t)=|K_{k0}(t)|^2.
]

### Consequences

* **Ballistic propagation** with velocity (v_{\max} = 2J).
* **Exponential suppression** outside (|k| \lesssim v_{\max} t\).
* **No strict cutoff**, consistent with Bessel-function tails.

*(Figure 3: Bessel-based propagator light cone)*

---

## **3.2 Proposition — Lieb–Robinson Bound (General Locality)**

For any finite-degree local Hamiltonian:
[
|[A_X(t),B_Y]| \le C e^{-\mu(d(X,Y)-v t)}.
]
This establishes a **universal light cone** independent of microscopic structure.

### Numerical Confirmation

* LC metric stable for all local lattices tested.
* LC breaks down only on fully connected graphs.

*(Figure 4: empirical arrival-time light cone)*

---

# **4. Emergent Geometry — Arrival-Time Metric**

Local propagation speed produces an effective geometric structure.

---

## **4.1 Definition — Arrival-Time Distance**

Let (\epsilon>0). Define
[
d_\epsilon(i,j)=\inf{t : P_{ij}(t) > \epsilon}.
]

### Analytic Result (XY chain)

[
d_\epsilon(i,j)= \frac{|i-j|}{2J} + O(1).
]
This defines an **operational distance**.

*(Figure 5: Linear scaling of (d_\epsilon) with separation)*

---

## **4.2 Noise-Dressed Velocity**

Numerical experiments show propagation slows under site-dependent noise (\eta_j).
A phenomenological model:
[
v(j) \approx \frac{2J}{1+\eta_j}.
]

*(Figure 6: velocity profile vs. decoherence)*

---

## **4.3 Proposition — Acoustic Effective Metric**

In the continuum limit, the signal front satisfies:
[
\frac{dx}{dt} = v(x).
]
This is equivalent to null propagation in an **acoustic metric**:
[
ds^2 = -dt^2 + \frac{dx^2}{v(x)^2}.
]

### Interpretation

* Spatial gradients in (v(x)) mimic curvature.
* Wavefronts exhibit focusing/defocusing.

*(Figure 7: comparison of wavefront bending to metric prediction)*

---

# **5. Numerical Evidence Summary**

This section catalogs the major empirical findings.

### **1. Statistics Gap**

Only **CNOT-like** detection generates antisymmetric preference.

### **2. Light-Cone Stability**

* Robust to noise up to ~30%.
* Universality across local graphs.

### **3. Noise-Type Dependence**

* Sparse noise destabilizes LC.
* Smooth noise preserves LC.

### **4. Universality Patterns**

* Local graphs: ~20–25% universal.
* Fully connected: 0%.
* CNOT detection: ~70% universal internally.

*(Figures 8–11: universality and LC metrics)*

---

# **6. Open Problems**

1. Full spectral characterization of (N)-qubit Lindbladian under detection.
2. Formal derivation of (\lambda^*) threshold.
3. Extension to multi-excitation sectors.
4. Tensor-network embedding for continuum limits.
5. Closed-form curvature expressions for discrete velocity fields.

These represent the forward direction of this research.

---

# **7. Conclusion**

The Hilbert Substrate Model unifies three emergent structures:

1. **Fermionic coherence preference** (analytic for 2 qubits, numerical evidence for many).
2. **Relativistic-style causal cones** (analytic and numerical).
3. **Effective geometric metrics** (derived + empirical).

The project is exploratory but disciplined, and this document reflects its present analytic maturity.

---

# **8. References**

* XY chain solution: standard in quantum spin chains (Lieb, Schultz, Mattis, 1961).
* Lieb–Robinson bounds: Lieb & Robinson (1972); Nachtergaele et al. (2006).
* Decoherence-free subspaces: Zanardi & Rasetti (1997); Lidar et al.
* Acoustic metric: Unruh (1981); Visser (1993).

---

# **Appendix A — Reproducing the Experiments**

Run:

```
python stability_analysis.py
python noise_analysis.py
python universality_test.py
```

Outputs appear in:

```
stability_results/
noise_analysis/
universality_results/
```

Each script generates plots, JSON summaries, and LC/metric measurements used throughout this document.
