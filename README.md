Resonance Laboratory
Hilbert Substrate Framework
Independent research exploring whether local quantum field theory — including spacetime geometry, causality, and fermionic statistics — emerges as a stable attractor of unitary evolution on generic entangled states in finite-dimensional Hilbert space.
Current scope (November 2025)
A minimal, open-source engine implementing:

a finite-dimensional Hilbert space with configurable local dimension
unitary dynamics governed by sparse interaction graphs
environmentally induced decoherence via unitary detection channels

From these ingredients alone, the framework reproducibly exhibits:

sharp light cones and approximately metric propagation delays
a robust statistics gap between copyable (bosonic) and non-copyable (fermionic) patterns
saturation of the Tsirelson bound in Bell tests
a clear robustness hierarchy under noise: light-cone structure degrades first, metric quality second, and the fermionic exclusion gap last

Key numerical findings to date:

Fermionic exclusion persists longest across all tested noise models
Universality rate reaches 73 % when detection couples to off-diagonal coherences
Failure modes align with theoretical expectations (absence of phase-sensitive detection or locality destroys the corresponding feature)
Results hold across qubits and qutrits and multiple lattice topologies

All code is vectorised, GPU-compatible when CuPy is available, and runs on consumer hardware. No background spacetime, fields, or statistics postulates are introduced.
Status
Active research prototype. Scripts are provided as research tools; a more structured package is in preparation.
Repository contents

substrate.py – core engine
noise_analysis.py, stability_analysis.py, universality_test.py – current experiments
outputs/ – generated figures and data

License
MIT
Resonance Laboratory is a one-person, unfunded effort conducted outside regular employment. Contributions, replications, and correspondence are welcome.
Ben Bray
Resonance Laboratory
November 2025 – ongoing
