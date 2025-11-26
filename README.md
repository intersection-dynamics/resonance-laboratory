# Resonance Laboratory — Hilbert Substrate Model

### **Exploratory Research Codebase for Emergent Causality, Statistics, and Geometry**

**Ben Bray — 2025**

This repository contains the research code, analytic notes, and numerical experiments for the **Hilbert Substrate Model** — an exploratory project studying how causal structure, fermionic-like coherence behavior, and effective geometric metrics can emerge from:

* finite-dimensional Hilbert spaces,
* local Hamiltonian evolution,
* phase-sensitive repeated detection,
* and environmental decoherence.

This README provides a clear, professional overview of the project, its goals, the current code layout, and the roadmap toward a reproducible v1.0 research release.

---

# **0. Status & Purpose**

This repository is **active but in-progress**. It currently reflects:

* working research code,
* a growing analytic framework,
* large-scale numerical experiments,
* and exploratory results.

It is **not** yet structured as a polished software package. A v1.0 cleanup is planned, with standardized outputs, refactored directories, and documented APIs.

The purpose of this codebase is to:

* explore emergent physics inside minimal quantum models,
* document the analytic ideas rigorously and honestly,
* support reproducible numerical experiments,
* and act as a foundation for deeper research.

---

# **1. Conceptual Overview**

The Hilbert Substrate Model studies three layers of emergent structure:

### **Layer 1 — Statistics Structure**

Repeated phase-sensitive detection appears to selectively preserve coherence in **antisymmetric** (fermion-like) subspaces.

### **Layer 2 — Causal Structure**

Local Hamiltonians generate **ballistic propagation** and **light-cone behavior**, consistent with XY-chain analytics and Lieb–Robinson bounds.

### **Layer 3 — Effective Geometry**

Spatially varying decoherence induces a **position-dependent propagation speed**, which corresponds to an effective **acoustic metric**.

These layers arise in the operational hierarchy:

**fermionic preference → causal cones → effective geometry**

For details, see the analytic foundations document:

```
docs/analytic_foundations_v1.md
```

---

# **2. Repository Structure (current, in-progress)**

The repo is presently organized as follows:

```
resonance-laboratory/
│
├── substrate.py                 # Core substrate engine (in progress)
├── universality_test.py         # Universality scans
├── noise_analysis.py            # Noise-type and size-scaling experiments
├── stability_analysis.py        # Stability-basins under coupling/noise sweeps
│
├── docs/
│   ├── analytic_foundations_v1.md   # Mature analytic framework
│   └── notes/                       # Additional in-progress theory notes
│
├── noise_analysis/               # Auto-generated experiment output (temporary)
├── stability_results/            # Auto-generated experiment output (temporary)
├── universality_results/         # Auto-generated experiment output (temporary)
│
└── README.md                     # This file
```

A formalized `engines/`, `experiments/`, and `outputs/` layout will be introduced in v1.0.

---

# **3. Core Components**

### **3.1 Substrate Engine (`substrate.py`)**

Implements:

* lattice generation (chain, ring, square, honeycomb, random, full),
* XY / Heisenberg / hopping Hamiltonians,
* wavefunction propagation,
* reduced density matrices,
* CNOT-like detection maps,
* metric & light-cone diagnostics.

This file will later be moved to `engines/substrate_engine.py` with a clean API.

---

### **3.2 Experiments**

#### **Noise Robustness — `noise_analysis.py`**

Outputs located in:

```
noise_analysis/
```

Generates:

* noise-type robustness plots,
* size-scaling stability,
* structure comparison (chain, ring, ladder, etc.),
* metric breakdown (light cone, metric, statistics gap).

Run:

```
python noise_analysis.py
```

---

#### **Stability Basin — `stability_analysis.py`**

Outputs in:

```
stability_results/
```

Sweeps:

* coupling strength,
* disorder strength,
* locality parameter,
* additive noise.

Run:

```
python stability_analysis.py
```

---

#### **Universality Suite — `universality_test.py`**

Outputs in:

```
universality_results/
```

Scans across:

* local dimensions,
* Hamiltonian families,
* lattice geometries,
* detection channels.

Produces light-cone metrics, statistics gaps, and universality classification.

Run:

```
python universality_test.py
```

---

# **4. Analytic Foundations**

The primary analytic reference for the project is:

```
docs/analytic_foundations_v1.md
```

This document outlines:

* exact XY-chain propagation (Bessel functions),
* Lieb–Robinson locality bounds,
* arrival-time distance and operational metric,
* noise-dressed propagation and acoustic geometry,
* hypotheses on fermionic coherence preference,
* numerical evidence supporting each claim.

It strictly distinguishes between:

* **established results**,
* **derived constructions**,
* **numerical observations**,
* **open conjectures**.

---

# **5. Example Results**

Here are typical outputs produced by the current code:

### **Light-Cone Structure**

* Consistent linear arrival times.
* Flat scaling up to N=12.
* Breakdown only on fully-connected graphs.

### **Statistics Gap**

* CNOT-like detection operators reveal a clear antisymmetric preference.
* SWAP/ZZ/random detection give near-zero gap.

### **Noise Behavior**

* Most noise types preserve causal structure up to ~30% strength.
* Sparse noise uniquely destabilizes the light cone.

### **Universality Space**

* Local sparse graphs: ~20–25% universal.
* Full graph: 0% universal.
* CNOT detection: ~70% universal internally.

These results are *not proofs* but form a coherent empirical foundation.

---

# **6. Installation & Requirements**

The project uses:

* Python 3.10+
* numpy
* scipy
* matplotlib

Install dependencies:

```
pip install numpy scipy matplotlib
```

GPU acceleration (CuPy) will be added in a later release.

---

# **7. Roadmap**

### **v0.3 — Current Phase**

* Analytic restructuring
* Numerical confirmation
* Code cleanup

### **v0.4 — Repository Refactor**

* engines/
* experiments/
* outputs/<exp>/<run_id>/
* Standardized JSON summaries
* Deterministic seeding

### **v0.5 — Visualizations**

* Interactive light-cone explorer
* Statistics-gap heatmaps
* Animated propagation

### **v1.0 — Reproducible Research Release**

* Docker/Conda environment
* Full documentation
* GPU backend
* Example notebooks
* Clean API for substrate simulations

---

# **8. License**

MIT License. You are free to use, modify, and build upon this work.

---

# **9. Contact**

For questions, collaboration, or discussion:

**Email:** [sjbbray@gmail.com](mailto:sjbbray@gmail.com)

---

This repository represents ongoing independent research into emergent physical structure from quantum substrates. Everything here is exploratory, honest, and continuously improving.
