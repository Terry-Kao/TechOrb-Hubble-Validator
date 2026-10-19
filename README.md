# Tech-Orb Research Program

Tech-Orb is a falsification-first research program investigating whether observed physics can be represented as local effective structure on a deeper physics state manifold. This repository contains both the developing foundation specification and independent validators used to attack concrete implementations.

## Current Status — 2026-10-20

- **Tech-Orb:** mother hypothesis; not an established physical theory.
- **Foundation v1:** specification stage. F0 contains four hypotheses, F1 is a Taiji topology ansatz, and F2 is a formula/quantity-geometry research module.
- **HIA / HEM / Hubble tension / CLASS / HyRec:** validators, not the foundation itself.
- **HEM v26.4:** a CLASS-level sensitivity and degeneracy experiment only. It does not establish a Hubble-tension solution, a data-preferred posterior, or a physical magnetogenesis model. Until canonical R0 is completed, its evidence status is **REPRO-BLOCKED**.
- **HEM v27:** next validator line, governed by R0 through R4. No gate is currently recorded as passed in this repository.
- **Historical HIA/RMP results:** retained as historical evidence. They constrain tested implementations or subclasses only; they are not universal no-go theorems.

The authoritative project state is [control/PROJECT_STATE.md](control/PROJECT_STATE.md). Claims must be classified in the [Epistemic Ledger](control/EPISTEMIC_LEDGER.md) and traceable through the [Evidence Index](control/EVIDENCE_INDEX.md).

## Mother Hypothesis vs. Validators

| Layer | Role | What failure means |
|---|---|---|
| Tech-Orb Foundation | Mother research hypothesis | A failed foundational requirement may exclude the stated hypothesis or implementation at its declared scope. |
| HIA / HEM | Concrete model families and experimental implementations | Excludes only the tested claim, implementation, or subclass. |
| Hubble, CLASS, HyRec and likelihood pipelines | Validators | Diagnose reproducibility, observational fit, or physical consistency; they do not directly decide the entire mother hypothesis. |

Validator success is also scoped: a correct, reproducible pipeline is not evidence that its tested model survives. **Gate PASS and model survival are separate verdicts.**

## Foundation Architecture

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

1. Complete HEM v27 R0 with a clean-room reproduction manifest and exact hashes.
2. Audit the baseline parameterization, neutrino assumptions, recombination backend, and `r_s`/`r_d` definitions under R1.
3. Select published benchmark model(s), datasets, and tolerances for R2 before implementing new physics.
4. Define the R3 likelihood stack and nuisance treatment without any target-value success rule.
5. Turn the three top Foundation attacks in [foundation/FALSIFICATION_MATRIX.md](foundation/FALSIFICATION_MATRIX.md) into executable test specifications.

No merge into `main` should be interpreted as scientific validation; it records the governed state of the research program.

