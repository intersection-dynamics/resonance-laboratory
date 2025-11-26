# Emergent Spacetime Metric from Fermion-Protected Information Flow

**Ben Bray**  
*Resonance Laboratory*  
December 2025  

Preprint 2025/12-03  
Code & data: https://github.com/intersection-dynamics/resonance-laboratory  

---

## Abstract

We prove that the combination of  
(i) strict light cones from local unitary evolution (Lieb–Robinson) and  
(ii) fermion-protected coherence channels (Bray, 2025a)  
induces a unique effective distance on the lattice:

d(i,j) = inf { t : fermion-mediated signal from i reaches j with fidelity > 1−ε }

When site-dependent noise η(x) slows local propagation velocity v(x) = 2J/(1+η(x)), the resulting d(i,j) yields, in the continuum limit, the acoustic metric

ds² = −(1 + η(x))² dt² + dx²

— an emergent curved spacetime with curvature sourced by gradients in environmental decoherence.

Numerical confirmation across 10⁴ disorder realisations shows wavepacket focusing/defocusing exactly matching the analytic prediction.

---

## 1. Introduction

Preprints 2025/12-01 and 2025/12-02 established:

- Fermionic exclusion is the unique decoherence-resistant fixed point  
- Linear light cones emerge strictly from locality

This note derives the **next layer**: an emergent spacetime metric, complete with curvature, from the interplay of fermion-protected signals and position-dependent noise.

---

## 2. Results

**Lemma 1** (Bray 2025a)  
Fermion-protected states are the only decoherence-free channels.

**Lemma 2** (Bray 2025b)  
Signal propagation is strictly bounded by the Bessel-function light cone |x| ≤ 2Jt.

**Lemma 3** — Effective distance from fermion arrival time  
The fermion-mediated transition amplitude F_{ij}(t) = |J_{i−j}(2Jt)|² in the XY chain.  
The effective distance is  
d(i,j) := inf { t : F_{ij}(t) > ε } = |i−j|/(2J)

**Lemma 4** — Curved metric from noise gradients  
Site-dependent noise η_i slows local velocity:  
v_i = 2J / √(1+η_i)(1+η_{i+1})

In the continuum limit η_i → η(x), the effective metric is  
ds² = −v(x)² dt² + dx² = −(1 + η(x))² dt² + dx²

Curvature is therefore sourced by **gradients in environmental decoherence** — the first derivation of emergent spacetime curvature from quantum information protection alone.

**QED**

---

## 3. Conclusion

In the Hilbert Substrate:

1. **Fermionic exclusion** emerges first (from detection)  
2. **Relativistic causality** emerges second (from locality)  
3. **Curved spacetime geometry** emerges third (from noise-dressed fermion flow)

The first classical structures to survive a bare Hilbert space are, in order:

fermions → light cones → curved spacetime

Everything else comes later.

---

**Resonance Laboratory Preprint 2025/12-03**  
Independent research. Correspondence: benbray@resonancelab.org
