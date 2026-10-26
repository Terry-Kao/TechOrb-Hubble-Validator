# Tech-Orb Research Program

Tech-Orb is a falsification-first research program investigating whether effective spacetime can arise from a global constrained relational quantum structure without assuming fundamental classical space or external/meta-time. This repository contains the developing Foundation, effective-model interfaces, and independent validators used to attack concrete implementations.

## Current Status — 2026-10-26

- **Tech-Orb:** mother hypothesis; not an established physical theory.
- **Active Foundation proposal:** Relational Quantum Foundation Canon v1.0 with working ontology `Q=(H,A,rho,K,T)`. This is a hypothesis/governance baseline, not evidence.
- **QF-0A:** known-result relational-time reproduction protocol frozen; implementation and scientific execution not started.
- **QF-0B/C:** emergent-distance/effective-dimension benchmark skeleton only; thresholds and execution not started.
- **QF-1/QF-2/QF-3:** deferred. Taiji and BH-J-WH/cyclic topology work are not authorized.
- **HIA / HEM / Hubble tension / CLASS / HyRec:** validators, not the foundation itself.
- **HEM v26.4:** a historical/reproducible CLASS-level sensitivity and degeneracy experiment only; no posterior or physical validation.
- **HEM v27:** R0 and R1 pipeline PASS with model verdict NOT APPLICABLE. R2 is NOT STARTED; this package does not start or modify HEM gates.
- **PFM:** external SAFE HOLD — COMPLETE / RESUMABLE; not awakened or modified here.
- **Historical HIA/RMP results:** retained as historical evidence. They constrain tested implementations or subclasses only; they are not universal no-go theorems.

The authoritative project state is [control/PROJECT_STATE.md](control/PROJECT_STATE.md). New Foundation claims are registered in the [QF Claim Registry](foundation/qf0/CLAIM_REGISTRY_v1.0.md); historical claims remain in the [Epistemic Ledger](control/EPISTEMIC_LEDGER.md), and artifacts are traceable through the [Evidence Index](control/EVIDENCE_INDEX.md).

## Mother Hypothesis vs. Validators

| Layer | Role | What failure means |
|---|---|---|
| Tech-Orb Foundation | Mother research hypothesis | A failed foundational requirement may exclude the stated hypothesis or implementation at its declared scope. |
| HIA / HEM | Concrete model families and experimental implementations | Excludes only the tested claim, implementation, or subclass. |
| Hubble, CLASS, HyRec and likelihood pipelines | Validators | Diagnose reproducibility, observational fit, or physical consistency; they do not directly decide the entire mother hypothesis. |

Validator success is also scoped: a correct, reproducible pipeline is not evidence that its tested model survives. **Gate PASS and model survival are separate verdicts.**

## Active Foundation Architecture

The governing research question, five claim labels, ontology, one-way Foundation → effective model → validators rule, roadmap, rejected literal versions, scope boundaries, and twelve global falsification gates are frozen in [Foundation Canon v1.0](foundation/qf0/FOUNDATION_CANON_v1.0.md).

The active roadmap is QF-0 relational toy-model reproduction/benchmarks → QF-1 Lorentz/GR recovery → QF-2 classicalization → QF-3 separately authorized topology research. PASS at any stage never establishes the next stage. A future QF-0A PASS would be known-result reproduction only and not Tech-Orb evidence.

The older F0/F1/F2 architecture below is preserved as historical/provisional Foundation v1 evidence; it is not deleted or retroactively rewritten.

## Preserved Foundation v1 Architecture

### F0 — Four core hypotheses

1. **Physics State Manifold** — investigate whether physical states admit non-trivial, coordinate-independent state-space structure.
2. **Projection / Effective Representation** — distinguish physical projection from coordinate relabeling and identify invariant observables.
3. **Emergent Time** — require recovery of proper time, Lorentz/gravitational time dilation, causal order, and null structure.
4. **Invariant / Constraint / Symmetry** — search for transformation laws and representation-independent structure before proposing conserved formulas.

### F1 — Taiji Candidate Topology

The two-sector / branch-exchange picture is an **ANSATZ**, not known cosmic topology and not a Foundation postulate. See [topology/TAIJI_DUAL_TOPOLOGY.md](topology/TAIJI_DUAL_TOPOLOGY.md).

### F2 — Formula / Quantity Geometry

F2 studies whether relations among physical quantities have useful geometric structure. It must respect dimensions, covariance, identifiability, and known limits; it is not permission to add unlike quantities or declare a universal transformation matrix.

The complete scope and advancement criteria are in [foundation/FOUNDATION_V1_SPEC.md](foundation/FOUNDATION_V1_SPEC.md) and [foundation/MATHEMATICAL_REQUIREMENTS.md](foundation/MATHEMATICAL_REQUIREMENTS.md). Proposed links to observables belong in the [Bridge Register](foundation/BRIDGE_REGISTER.md), not in Foundation canon.

## HEM v27 Validator

HEM v27 uses one canonical sequence:

| Gate | Name | Purpose |
|---|---|---|
| R0 | Reproducibility Freeze | Freeze executable source, environment, parameters, inputs, patches, outputs, and hashes. |
| R1 | Baseline Definition Audit | Establish a verified ΛCDM/CLASS/HyRec baseline and parameter conventions. |
| R2 | Published Model Reproduction | Reproduce relevant published benchmarks before claiming a new implementation. |
| R3 | Full Multi-Probe Likelihood | Evaluate the model with an auditable likelihood, covariance, nuisance parameters, and multiple probes. |
| R4 | Physicality Benchmark | Test whether the effective deformation follows from a viable physical model and respects independent constraints. |

An R3 run can be a **pipeline PASS even if it returns `H0 ≈ 69` or excludes the model**. `H0 > 71` is prohibited as a gate-success criterion. BBN is used only as an early-universe consistency constraint for magnetogenesis/PMF scenarios, not as a direct retrospective test that closes a recombination result. Any magnetogenesis action remains a Candidate Bridge / **ANSATZ** unless separately derived and promoted under governance.

See [validators/hem_v27/GATE_PROTOCOL.md](validators/hem_v27/GATE_PROTOCOL.md).

## What v26.4 Does and Does Not Establish

**It may establish, after R0 reproduction:** that a specified effective recombination deformation in a specified CLASS setup has a geometric sensitivity or degeneracy direction, including a possible acoustic-angle lock.

**It does not establish:**

- that data prefer `H0 ≈ 73`;
- a cosmological posterior or Planck acceptance;
- a full TT/TE/EE plus BAO/SN likelihood result;
- a viable PMF or magnetogenesis mechanism;
- a universal mapping between a magnetic-field amplitude and an effective clumping parameter;
- any Tech-Orb Foundation ontology.

Theory-to-theory spectrum RMS is an engineering diagnostic, not an observational likelihood. `r_s(z_*)` and the BAO drag ruler `r_d = r_s(z_d)` must not be conflated.

## Research Governance

- Every material claim carries one epistemic class: **FACT, DERIVED, HYPOTHESIS, ANSATZ, EXCLUDED,** or **OPEN QUESTION**.
- Every entry also carries an evidence status: **VERIFIED, SUPPORTED, PROVISIONAL, REPRO-BLOCKED, SUPERSEDED,** or **HISTORICAL**.
- `EXCLUDED` must specify **CLAIM, IMPLEMENTATION,** or **SUBCLASS** and an explicit boundary. Universal exclusions require universal evidence.
- Numerical success cannot upgrade an ansatz to a fact. AI review consensus is critique metadata, not scientific evidence.
- Promotions, exclusions, gate decisions, and supersessions require an entry in [control/DECISION_LOG.md](control/DECISION_LOG.md).

## Repository Structure

| Path | Purpose |
|---|---|
| `control/` | Current state, governance, ledger, decisions, and evidence map |
| `foundation/` | Foundation v1 specification, requirements, bridge register, falsification matrix |
| `topology/` | F1 Taiji candidate topology |
| `validators/` | Current validator protocols and bounded exclusion maps |
| `reviews/` | Review provenance; never a substitute for evidence |
| `docs/`, `papers/`, `core_engine/` | Preserved HIA/Hubble research history and code |
| `archive_RMP_v4/` | Preserved RMP/Origin historical material |
| `validation/` | Legacy validation area; not canonical for HEM v27 |

## Current Next Actions

1. Review and merge (or revise) the bounded QF-0 Foundation initialization package without interpreting merge as scientific validation.
2. If accepted, create a separate QF-0A implementation authorization bound to the frozen v1.0 protocol before any result inspection.
3. Independently audit QF-0A for external-time leakage, clock-choice handling, negative controls, and no-new-physics wording.
4. Resolve every `TBF` item in the QF-0B/C skeleton in a separately reviewed protocol before generating adjudicative benchmarks.
5. Populate the literature matrix from primary sources without using compatibility or citation similarity as evidence.

No merge into `main` should be interpreted as scientific validation; it records the governed state of the research program.

