# Tech-Orb Research Program

## Current host-satisfiable re-freeze acceptance — 2026-10-30

This section supersedes all earlier QF-0A current-status checkpoints below; historical records remain preserved.
PR #23 Host-Satisfiable Environment Repair: ACCEPTED / MERGED. Approved head f38bf8561891f820517c873e70960dca200069a8; merge M4 cb1c44c0b888595ef07bf3bc68db6eb3fe43aa48. Repairs F-14 CPU source, F-15 positive satisfiability and F-16 validation-stack observation; technical bytes frozen by PR #24.
PR #24 Host-Satisfiable Main-Lineage Re-Freeze: ACCEPTED / MERGED. Approved head d6c499b5060c7e54a2c56a832b83dbde63542c58; merge M5 60ff4cd1d41d8e77def72e741c3a25969b95b4b7.
Source baseline remains M4 cb1c44c0b888595ef07bf3bc68db6eb3fe43aa48. Main-lineage source re-freeze accepted.
Manifest blob f74f1bf4e9fd6e81b2c11c1bf2008198953d1673; SHA-256 fb072ab0abeb443baf52f15807a827e94b43524081b89ffb3873da7df9995241; 41 entries; scientific_execution NOT_PERFORMED.
Runner blob d4113bfcacc2c7abc91c97f5b8e41ed0359f5f1a; environment 10fb01ea128dcfec3d7b795f06903ac228c05f6b; prereg ac74a7753b35d0f7a0c9ce2e4ebb16ccc05081d1; runtime closure test 60fbd82ef2c13c508b52db273018cebb573de0bc.

Claude's historical final challenge against 6aaf4a24882fcc527ff82abfd1001413064c42b9 returned REVISE for F-14 HIGH. That verdict remains historical evidence. F-14 was subsequently host-confirmed; PR #23 repaired it and PR #24 re-froze M4.
F-01 CLOSED; F-02 CLOSED; F-03 CLOSED; F-04 CLOSED; F-05 CLOSED; F-06 ACCEPTABLE; F-07 CLOSED; F-08 ACCEPTABLE.
F-04 was previously reopened through F-14 after Claude's final challenge; F-14/F-15/F-16 repair and main-lineage re-freeze now restore F-04 to CLOSED, pending the next independent Claude narrow final review.
F-14 CLOSED: registry ProcessorNameString, narrow whitespace-only normalization, no fallback, missing/error/empty/wrong value fail closed; repaired host satisfies freeze.
F-15 CLOSED: positive coverage and real-host satisfiability observed; rejection coverage retained.
F-16 CLOSED: actual jsonschema 4.23.0 observed, valid date-time accepted, invalid rejected, missing/nonfunctional checker fails closed.
Remaining follow-ups: F-09 LOW / ACCEPTABLE; F-10 LOW / OPEN; F-11 LOW / OPEN (scientific-strength limitation); F-12 LOW / OPEN; F-13 LOW / substantially mitigated; F-17 MEDIUM / execution condition; F-18 LOW; F-19 LOW; F-20 LOW; F-21 LOW (build metadata, not loaded-DLL attestation). These are not closed by this sync.

HOST-LOCAL OBSERVATION: full frozen-host satisfiability PASS after merge; CPU AMD Ryzen 7 9700X 8-Core Processor from Windows registry, strip-and-collapse-whitespace; jsonschema 4.23.0 and behavioral date-time capability PASS. This is neither CI evidence nor external/GitHub-independent replication. Independent Claude review remains outstanding.
**QF0A-v1.1-QUAL-001 — IMMUTABLE HISTORICAL ATTEMPT**
EXECUTED / NUMERICAL GATES PASS / EVIDENCE CONTRACT BLOCKED.
Machine PASS is historical runner output only; Foundation NOT_APPLICABLE; not accepted qualification; no retry; historical bytes immutable.
QUAL-002 authorization: NOT CREATED / UNAUTHORIZED; run: NOT STARTED; evidence: ABSENT.
Host/environment repair and main-lineage re-freeze accepted, pending independent Claude final narrow challenge and Terry governance decision.
Classification: Software Qualification + Known-result Reproduction + Constructibility Demonstration. No Foundation/ontology, emergent-time, absence-of-meta-time, Page–Wootters empirical, GR/Lorentz, QF-0D or empirical physics support is established.
No technical frozen artifact modified by this governance sync. No QF-0A or scientific execution.
Next gate: **Claude Independent Narrow Final Pre-QUAL-002 Review — Post-F14/F15/F16 Repair / Final-Main Edition**.
Only after an acceptable verdict: Terry Governance Decision — Authorize or Hold QF0A-v1.1-QUAL-002.


## Current QF-0A Phase B acceptance — 2026-10-30

This section supersedes every earlier QF-0A checkpoint below.
Runtime Provenance & Environment-Observation Closure + Main-Lineage Re-Freeze: independently reviewed, ACCEPTED / MERGED.
Phase A PR #19 merge: bab7393b370c36b72b9f08f4e7eb4285bf49420c; ACCEPTED / MERGED.
Transition PR #20 merge: ec6497164b89b1c050e0457ca8cb3770305990bf; ACCEPTED / MERGED. Replaces Phase-A-only ancestry-failure assertion with lifecycle-aware provenance readiness.
Phase B PR #21 approved head: 4c317490386cf5b5c04b477c5be0327f796dea95; merge M3: dad1dedb166262046dce2b6d7af74eada794fc06; ACCEPTED / MERGED.
Source baseline M2: ec6497164b89b1c050e0457ca8cb3770305990bf.
Manifest blob: 8a240e7effcdd6b2fe7adf053aea1b5203d2a38a; SHA-256: 8b05e8ef9a0adff3cec373bde6ac92625577898b5764c4ea86ef1d7623ea8667; 41 entries; scientific_execution NOT_PERFORMED.
F-01 CLOSED (governance policy); F-02 CLOSED (MAIN-LINEAGE SOURCE BASELINE); F-03 CLOSED (loaded module origin/bytes); F-04 CLOSED (environment and BLAS/LAPACK observation); F-05 CLOSED (canonical absent evidence root); F-07 CLOSED (executed static leakage validation).
F-06/F-08: no new scientific finding or independently inferred finding classification is asserted by this sync.
Protocol blob 3af303a471d99397429d33f77ca512578f40f972; SHA-256 1cf9f7f3a8cfa6a5b0e12a3290d7e545f44586f9359aaa5367754415cdcc5f3a.
Prereg blob 1afb1653c32432092cf85214eaedb7bc56213b48; SHA-256 e1816e2d54755fd3f5c1a8adac411c3efbcdb15c597c009f5f9daea6744bd386.
Environment blob 4bb6f73d5cfd96422829d3f0680ed52c9614e9e0; SHA-256 13bd8a95059363a5b2208f36e514e0e7d08b4467f4a0db35e2ce2bf37c41299a.
Dependency lock SHA-256 9ec9da7aaa0f7a7b85d9cb33909d03b0b934f0948c0fbb1c1182274e2edea0b2.
Technical frozen artifacts are unchanged by this synchronization.

**QF0A-v1.1-QUAL-001 — IMMUTABLE HISTORICAL ATTEMPT**
EXECUTED / NUMERICAL GATES PASS / EVIDENCE CONTRACT BLOCKED.
Raw machine verdict PASS is historical runner output only; Foundation verdict NOT_APPLICABLE.
Not accepted qualification; no retry; historical bytes immutable.
QUAL-002 authorization: NOT CREATED / UNAUTHORIZED; run: NOT STARTED; evidence: ABSENT.
Implementation/provenance repair and re-freeze gates accepted, pending independent final pre-QUAL-002 challenge review.

Classification: Software Qualification + Known-result Reproduction + Constructibility Demonstration.
A future QUAL-002 PASS establishes no Foundation validation, Tech-Orb ontology, emergent time, absence of external/meta-time, Page–Wootters physical reality, GR/Lorentz emergence, QF-0D support, or empirical physics result.
No QF-0A execution or qualification authorization occurred in this synchronization.

Next gate: **Claude Independent Final Pre-QUAL-002 Challenge Review — Post-Repair / Final-Main Edition**.
Only after an acceptable independent verdict: Terry Governance Decision — Authorize or Hold QF0A-v1.1-QUAL-002.

## Current QF-0A evidence-contract acceptance — 2026-10-29

This section supersedes earlier QF-0A status checkpoints below.
PR #17 approved head `bb4a07170414204f83b6bb7540497162433f6194` was squash merged at `cc6bf75e4ae43dcac5327adb78500a7b575c57fb`. Evidence Contract Serialization + Exact Inventory Repair: **ACCEPTED / MERGED / NOT EXECUTED**.
Protocol v1.1: **FROZEN / ACCEPTED**; blob `3af303a471d99397429d33f77ca512578f40f972`; SHA-256 `1cf9f7f3a8cfa6a5b0e12a3290d7e545f44586f9359aaa5367754415cdcc5f3a`.
Preregistration v1.1: **FROZEN / ACCEPTED**; blob `ce92de9e5d2b0dadf8e40f194ddfaf9b94a32844`; SHA-256 `60d90f23f3369456b24d8af83236f232a745aeb2bb125099725408859237821c`.
Environment Freeze v1.1: **FROZEN / ACCEPTED / READY / NOT EXECUTED**; blob `4bb6f73d5cfd96422829d3f0680ed52c9614e9e0`; SHA-256 `13bd8a95059363a5b2208f36e514e0e7d08b4467f4a0db35e2ce2bf37c41299a`; dependency lock SHA-256 `9ec9da7aaa0f7a7b85d9cb33909d03b0b934f0948c0fbb1c1182274e2edea0b2`. Host readiness was separately verified host-locally; GitHub has not independently re-measured host hashes.
Scientific + Adjudication Source Freeze v1.1: **FROZEN / ACCEPTED / NOT EXECUTED**; source baseline `80403cb383807334066eec482012ed42d4ded0ca`; manifest `foundation/qf0/qf0a_impl/manifests/QF0A_SOURCE_MANIFEST_v1.1.json`: 40 entries; SHA-256 `a86d73cd80094711a4790c2e6944a400e59fe7e8d8833e62b20f754a7e48a5ce`.
Implementation blobs: evidence.py `b7e717d2a4750c94a25bb3a4f3488710b4727388`; runner.py `1076e3b70a3bed710e306475d5da1a22236bbc79`; test_evidence_serialization.py `f26b100fe883fa8395a8049b1fa6359e8572795c`.

Evidence contract closure: **ACCEPTED**. Inventory validation is set equality: exactly 20 pre-verdict regular files, 22 final regular files, and 21 non-checksum paths in SHA256SUMS; only raw/ and logs/ directories are allowed. Extra files/directories, correctly hashed extra checksum entries, missing entries, duplicate paths, and hash mismatches fail closed. A machine-verdict file alone does not constitute accepted qualification; successful official process exit and successful G3 final sealing are mandatory.

QUAL-001: **FIRST QUALIFICATION ATTEMPT — EXECUTED / NUMERICAL GATES PASS / EVIDENCE CONTRACT BLOCKED**; **IMMUTABLE HISTORICAL ATTEMPT / NUMERICAL GATES PASS / EVIDENCE CONTRACT BLOCKED**. Raw machine verdict PASS and Foundation verdict NOT_APPLICABLE remain historical output; governance adjudication BLOCKED because raw_output_index, negative_controls, and machine_verdict failed schemas. No retry; no accepted qualification. Historical bytes remain immutable.
QUAL-002 authorization: **NOT CREATED / UNAUTHORIZED**; run: **NOT STARTED**. This synchronization generated no science, authorization, evidence root, or machine verdict.

Classification remains **SOFTWARE QUALIFICATION / KNOWN-RESULT REPRODUCTION / CONSTRUCTIBILITY DEMONSTRATION**. A future PASS does not validate Tech-Orb Foundation, emergent time in nature, absence of external time, clock ambiguity resolution, arbitrary temporal-frame equivalence, interacting relational dynamics, or QF-0D.

**Next gate: Claude Independent Final Pre-QUAL-002 Challenge Review**.

## Earlier checkpoint record (QF-0A statuses superseded above)


Tech-Orb is a falsification-first research program investigating whether effective spacetime can arise from a global constrained relational quantum structure without assuming fundamental classical space or external/meta-time. This repository contains the developing Foundation, effective-model interfaces, and independent validators used to attack concrete implementations.

## Current Status — 2026-10-29

- **Tech-Orb:** mother hypothesis; not an established physical theory.
- **Active Foundation proposal:** Relational Quantum Foundation Canon v1.0 with working ontology `Q=(H,A,rho,K,T)`. This is a hypothesis/governance baseline, not evidence.
- **QF-0A:** v1.0 is historical, superseded for future execution, and never executed. The independently approved v1.1 protocol, preregistration, environment, scientific/adjudication source freeze, runtime frozen-source identity closure, and static provenance audit are now FROZEN / ACCEPTED / NOT EXECUTED. Scientific execution remains UNAUTHORIZED and the first qualification run has NOT STARTED. A future PASS would qualify software/pipeline behavior only, not support the Foundation or QF-0D.
- **QF-0B/C:** emergent-distance/effective-dimension benchmark skeleton only; thresholds and execution not started.
- **QF-0D:** relational spacetime/causality stage restored; it may combine A/B/C only after their separate PASS verdicts. Execution is not authorized.
- **QF-1/QF-2/QF-3:** deferred. QF-1 is Lorentz/GR low-energy recovery; Taiji and BH-J-WH/cyclic topology work are not authorized.
- **HIA / HEM / Hubble tension / CLASS / HyRec:** validators, not the foundation itself.
- **HEM v26.4:** a historical/reproducible CLASS-level sensitivity and degeneracy experiment only; no posterior or physical validation.
- **HEM v27 dual provenance:** authoritative `main` records R0/R1 pipeline PASS, model verdict NOT APPLICABLE, and R2 = NOT STARTED. Unmerged Draft PR #8 at head `451b2a3875d16fea026fe74748bb3117f2c457ac` contains non-authoritative candidate evidence that R2 numerical execution completed with candidate verdict BLOCKED (560/560 adjudicable PASS; 334/334 diagnostic-only within tolerance; numerical FAIL 0), blocked by missing frozen legal extraction routes for three curves and an unresolved neutrino-baseline discrepancy. PR #9 records provenance only; it neither adjudicates/merges PR #8 nor authorizes R2 amendment/R3/R4.
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

The active roadmap is QF-0A emergent time → QF-0B emergent distance → QF-0C effective dimension → QF-0D relational spacetime/causality → QF-1 Lorentz/GR low-energy recovery → QF-2 classicalization → QF-3 Taiji relational phase topology. QF-0D tests causal/Lorentzian structure; QF-1 separately tests local Lorentz/covariance, locality, equivalence behavior, GR dynamics/conservation, and matter/QFT embedding. PASS at any stage never establishes the next stage. A future QF-0A PASS would be known-result reproduction only and not Tech-Orb evidence.

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

1. Prepare a separate QF-0A v1.1 identity-bound Scientific Execution Authorization against the final authoritative `main`; no authorization exists in this acceptance synchronization.
2. Only after that separate authorization is reviewed and issued may the first qualification run begin.
3. Preserve the v1.0 review chain as historical process evidence while using v1.1 exclusively for any future execution decision.
4. Resolve every `TBF` item in the QF-0B/C skeleton in a separately reviewed protocol before generating adjudicative benchmarks.
5. Populate the literature matrix from primary sources without using compatibility or citation similarity as evidence.

No merge into `main` should be interpreted as scientific validation; it records the governed state of the research program.


## QF-0A v1.1 post-governance execution-readiness status (2026-10-29)

The repository repair and re-freeze from PR #15 is **ACCEPTED / MERGED / NOT EXECUTED**. Protocol v1.1 and Preregistration v1.1 remain **FROZEN / ACCEPTED / NOT EXECUTED**. Environment Freeze v1.1 remains **FROZEN / ACCEPTED / READY / NOT EXECUTED**; frozen Windows environment provisioning was verified separately on the execution host, not independently re-measured by GitHub.

The Scientific + Adjudication Source Freeze v1.1 is **FROZEN / ACCEPTED / NOT EXECUTED** at source baseline `3c144f0f523800973bcff68f40d59a1e34ed6932`, with 39 manifest entries and source-manifest SHA-256 `ed504e4ddec500de69ea5d2834b532ad17d8f45a5a1d8f8a4489d6a12b132e09`. Runtime post-governance readiness closure is **ACCEPTED**.

Scientific Execution Authorization is **NOT CREATED / UNAUTHORIZED**. The first qualification run is **NOT STARTED**. QF-0A remains only `SOFTWARE QUALIFICATION / KNOWN-RESULT REPRODUCTION / CONSTRUCTIBILITY DEMONSTRATION`; a future PASS would not validate Tech-Orb Foundation, emergent time in nature, absence of external time, clock ambiguity resolution, arbitrary temporal-frame equivalence, interacting relational dynamics, or QF-0D.

**Next gate:** QF-0A v1.1 — Identity-Bound Scientific Execution Authorization + First Qualification Run.
