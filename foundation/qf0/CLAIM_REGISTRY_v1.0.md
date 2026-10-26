# Tech-Orb QF Claim Registry v1.0 — Human View

**Frozen:** 2026-10-26  
**Machine authority:** `CLAIM_REGISTRY_v1.0.json`  
**Result status:** registration only; no claim has received a scientific PASS

This view summarizes the machine-readable records. The JSON record retains the full mathematical definition, supporting and conflicting evidence, dependency list, validators, fail condition, verdict, and review date for every claim.

| Claim ID | Statement (bounded) | Label | Dependencies | Current verdict |
|---|---|---|---|---|
| QF-FND-001 | Global constrained relational quantum structure as candidate substrate | HYPOTHESIS | — | Registered; untested |
| QF-TIME-001 | Internal-clock conditioning may yield effective relational time | HYPOTHESIS | QF-FND-001 | Protocol frozen; reproduction not executed |
| QF-DIST-001 | Coordinate-free relational data may yield distance/pseudodistance | HYPOTHESIS | QF-FND-001 | Benchmark skeleton only |
| QF-DIM-001 | Effective dimension may be inferred without presupposing 3 | HYPOTHESIS | QF-DIST-001 | Benchmark skeleton only |
| QF-CAUS-001 | QF-0D may combine time/distance/dimension into Lorentzian causal structure | HYPOTHESIS | QF-TIME-001, QF-DIST-001, QF-DIM-001 | QF-0D; deferred until A/B/C pass |
| QF-GR-001 | A viable implementation must recover the relevant GR/relativistic limit | ESTABLISHED | QF-CAUS-001 | Requirement established; recovery absent |
| QF-CLASS-001 | Classical geometry may require robust/redundant accessible records | HYPOTHESIS | QF-CAUS-001 | Deferred to QF-2 |
| QF-TAIJI-001 | Dual-sector relational phase topology | ANSATZ | QF-GR-001, QF-CLASS-001 | Deferred; QF-3 unauthorized |
| QF-BHWH-001 | BH–junction–WH transition | HYPOTHESIS | QF-TAIJI-001 | Deferred high risk; unauthorized |
| QF-REJ-001 | Information literally replaces time as fourth coordinate | REJECTED-LITERAL-VERSION | — | Rejected literal model |
| QF-REJ-002 | Higher-dimensional object literally casts spacetime as a shadow | REJECTED-LITERAL-VERSION | — | Rejected literal model |

## Human review cards

### QF-FND-001 — substrate candidate

- **Definition:** `Q=(H,A,rho,K,T)` without fundamental classical space or external/meta-time.
- **Evidence boundary:** adjacent theories make this investigable; they do not support its truth in nature.
- **Fail:** invalid quantum structure, reference inconsistency, hidden time, answer leakage, or no non-gauge/predictive emergence chain.
- **Validators:** Quantum, Reference, Predictivity, Validator Leakage, Novelty.

### QF-TIME-001 — relational time

- **Definition:** a constrained global state conditioned on an internal clock, with a Page–Wootters-type analytic target.
- **Conflict/risk:** finite clocks, interactions, and clock choices can spoil normalized unitary evolution.
- **Fail:** hidden external time; unstable normalization; uncontrolled clock dependence; post-result threshold changes.
- **Validators:** Quantum, Reference, Causality.

### QF-DIST-001 — emergent distance

- **Definition:** preregistered `d_ij=F(R_ij;theta)` from relational inputs only.
- **Conflict/risk:** relational matrices need not determine a unique metric geometry.
- **Fail:** coordinate/dimension leakage, failed declared axioms/distortion thresholds, or failure against null controls.
- **Validators:** Geometry, Validator Leakage, Novelty.

### QF-DIM-001 — effective dimension

- **Definition:** at least two independent, calibrated, scale-aware estimators.
- **Conflict/risk:** finite size, sampling, topology, anisotropy, and noise bias estimates.
- **Fail:** blind 1D/2D/3D targets missed under frozen calibration, assumed answer 3, or uncontrolled estimator disagreement.
- **Validators:** Dimension, Validator Leakage, Novelty.

### QF-CAUS-001 — QF-0D relational spacetime / causality

- **Definition:** effective causal order, `(-,+,+,+)`-equivalent signature in scope, null propagation, and no-signalling behavior.
- **Fail:** inconsistent order, unexplained preferred frame, prohibited cycles, signalling, or unstable null structure.
- **Validators:** Causality, Reference, Quantum.
- **Stage boundary:** QF-0D combines separately passed QF-0A/B/C ingredients. QF-1, not QF-0D, is responsible for local Lorentz/GR low-energy recovery.

### QF-GR-001 — GR recovery requirement

- **Definition:** derived recovery of Einstein dynamics or a declared constrained modification, plus verified relativistic limits.
- **Evidence boundary:** the requirement is established; Tech-Orb has not met it.
- **Fail:** contradiction with verified behavior or assuming the structure claimed to be derived.
- **Validators:** GR, Causality, Predictivity.

### QF-CLASS-001 — classicalization

- **Definition:** preregistered robustness, redundancy, independence, and operational accessibility for geometry-like records.
- **Evidence boundary:** decoherence and redundant-record frameworks supply tools, not proof of geometry or unique outcomes.
- **Fail:** unavailable/non-independent records, failed metrics, or interpretation beyond measurements.
- **Validators:** Classicality, Quantum, Information.

### QF-TAIJI-001 — deferred topology ansatz

- **Definition:** sectors `S_plus`, `S_J`, and `S_minus`, without literal shape or fundamental meta-time.
- **Evidence boundary:** no supporting evidence; the diagram is hypothesis-generating only.
- **Fail:** no coherent topology/map compatible with local Lorentzian physics and quantum consistency, or no discriminating content.
- **Validators:** Causality, Information, Predictivity, Analogy.

### QF-BHWH-001 — deferred high-risk hypothesis

- **Definition:** placeholder `BH <-> J <-> WH` requiring explicit state spaces, channel/transition law, observables, and singularity treatment.
- **Evidence boundary:** black-hole information research is adjacent context, not evidence for this transition.
- **Fail:** causal/unitary inconsistency, undefined cross-sector observables, or no discriminating prediction.
- **Validators:** Information, Causality, GR, Predictivity, Novelty.

### QF-REJ-001 — information as fourth coordinate

- **Rejected literal form:** `ds^2=dI^2+dx^2+dy^2+dz^2` as a fundamental replacement for Lorentzian spacetime.
- **Reason:** it does not recover signature, light cones, or relativistic causal classification.
- **Reopening threshold:** a non-relabelling construction that recovers all relevant physics and operationally establishes coordinate status.

### QF-REJ-002 — geometric-shadow projection

- **Rejected literal form:** an ordinary higher-to-lower-dimensional shadow used as the fundamental quantum emergence mechanism.
- **Reason:** it supplies no quantum composition, conditioning, causality, classicalization, or unique observable bridge.
- **Reopening threshold:** an explicit, non-analogical quantum map with discriminating consequences.

## Promotion rule

No registry status may be promoted by terminology similarity, a paper citation, AI agreement, a toy-model reproduction, a validator fit, or a PASS at another layer. Promotion requires a separately frozen test, the declared evidence, a bounded verdict, and a Decision Log entry.
