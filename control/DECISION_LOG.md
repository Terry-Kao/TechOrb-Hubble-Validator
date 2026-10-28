# Decision Log

## D-2026-10-20-01 — Governance Reset

**Status:** Approved by merged PR #2 (4134c4b0e95c28975e4a64e97157ffc9c7b2f843)  
**Decision:** Separate Tech-Orb mother hypothesis from all validators; establish Foundation F0/F1/F2, an epistemic/evidence ledger, bridge register, falsification matrix, and HEM v27 R0–R4 protocol.

**Reason:** The prior repository conflated specific HIA/Hubble implementations, numerical target attainment, and broad scientific conclusions. The reset makes claim scope and evidence state auditable.

**Preserved:** All pre-reset HIA/RMP code, paper sources, plots, and audit records remain available as historical evidence.

## D-2026-10-20-02 — Bound the late-time negative result

**Status:** Approved by merged PR #2  
**Decision:** Reclassify the old “comprehensive No-Go theorem” as a historical exclusion study of tested late-time subclasses. Universal wording is superseded.

**Reason:** The repository contains numerical investigations of particular kinematic, screened-gravity, and disformal implementations, not a proof covering all smooth late-time theories.

## D-2026-10-20-03 — Classify HEM v26.4

**Status:** Approved by merged PR #2  
**Decision:** Record v26.4 only as a CLASS-level sensitivity/degeneracy experiment with evidence status `REPRO-BLOCKED` before R0.

**Not established:** a data-preferred posterior, Planck acceptance, full multi-probe viability, physical PMF/magnetogenesis derivation, or Foundation validation.

## D-2026-10-20-04 — Freeze HEM v27 gates

**Status:** Approved by merged PR #2  
**Decision:** Canonical order is R0 Reproducibility Freeze → R1 Baseline Definition Audit → R2 Published Model Reproduction → R3 Full Multi-Probe Likelihood → R4 Physicality Benchmark.

**Rules:** Pipeline PASS is separate from model survival; `H0 > 71` is forbidden as a gate criterion; BBN has only the scoped early-universe consistency role recorded in the gate protocol.

## D-2026-10-20-05 — Freeze candidate R0 recovery; retain BLOCKED

**Status:** Proposed in R0 recovery draft PR; no gate promotion  
**Decision owner:** Repository maintainer through PR review  
**Previous state:** R0 NOT PASSED; v26.4 REPRO-BLOCKED.  
**Proposed state:** R0 BLOCKED / NOT PASSED; model NOT APPLICABLE. Evidence status remains REPRO-BLOCKED.

**Evidence:** [recovery package](../validators/hem_v27/r0/recovery-2026-10-20/README.md), [manifest](../validators/hem_v27/r0/recovery-2026-10-20/MANIFEST.json), [issue #3](https://github.com/Terry-Kao/TechOrb-Hubble-Validator/issues/3).

**Scope:** Freeze inspected candidate runner/notebook, patched RECFAST source, complete tracked diff, small historical summary and source/output hashes. Record that N_ncdm=0 is inherited from CLASS defaults and distinguish the 0.06 eV neutrino comparison. Larger outputs remain local and indexed by checksum.

**Unresolved:** ENV-02 (available Docker engine stores container data on unauthorized C:; sole existing WSL distro lacks toolchain), PROV-01 (historical build/runtime identity), RAW-01 (deleted INI/raw outputs), and BASE-01 (baseline conventions). Source-to-output attribution and independent clean-room reproduction remain unproved. ENV-01 is closed after the reboot; TOL-01 is closed by the pre-run contract.

**Promotion condition:** A separate auditable clean-room gate report must resolve these blockers and satisfy GATE_PROTOCOL.md before any R0 PASS decision. Any recovered mismatch remains visible and prevents promotion.

## D-2026-10-20-06 — Freeze first-rerun tolerances; retain BLOCKED after reboot

**Status:** Proposed in R0 recovery draft PR; no gate promotion
**Decision:** Freeze `TOLERANCE_CONTRACT.md` before any numerical rerun. Close ENV-01 because Docker Desktop and its Linux engine became reachable after reboot. Open ENV-02 because Docker's data VHDX is on C: and the only existing WSL distro lacks the required toolchain.

**Execution consequence:** Do not create a container, image, distro, build, or numerical output until a pre-existing clean Linux runtime can keep all task artifacts, metadata, caches, temporary files, logs, and outputs within the authorized H: tree.

**Gate consequence:** R0 remains BLOCKED / NOT PASSED; model verdict remains NOT APPLICABLE; R1-R4 remain NOT STARTED.

## D-2026-10-20-07 — Authorize Docker system writes and adjudicate R0 FAIL

**Status:** Executed on the R0 recovery branch; pending PR review.
**Decision:** Close ENV-02 after the user explicitly authorized existing Docker Desktop and WSL to use their default C: system components, VHDX, metadata, layers and caches. Keep controllable Tech-Orb source, inputs, logs, raw outputs, derived artifacts, manifests and reports on H:.

**Execution:** Exact CLASS commit `aa92943e4ab86b56970953589b4897adf2bd0f99` was built as clean upstream and scientific-only patched source in Docker. The unmodified control passed exactly. Historical and explicit-0.06-eV baselines were kept separate. All four declared cases completed without changing `b`, `n_s`, `A_s`, `H0`, the model, or the frozen tolerances.

**Adjudication:** 50/52 field comparisons passed. `max_residual_final_pct` failed for `Mode1_b1.40` (delta 0.0030040430 > 0.0005) and `Mode2_b1.35` (delta 0.0009771732 > 0.0005). R0 pipeline state changes from BLOCKED to **FAIL**. Model verdict remains NOT APPLICABLE. R1-R4 remain NOT STARTED.

**Provenance boundary:** PROV-01 and RAW-01 remain historical limitations because original historical build/run identity and raw files are missing. The new rerun itself has complete raw evidence and provenance, so those historical limitations do not override the completed candidate-source FAIL.

**Evidence:** `validators/hem_v27/r0/recovery-2026-10-20/R0_CLEANROOM_REPORT.md` and `cleanroom/`.

## D-2026-10-21-08 — Reconcile historical environment and close R0 PASS

**Status:** Executed on the R0 recovery branch; pending PR review.
**Decision:** R0 pipeline verdict is **PASS** for historical v26.4 reproducibility. Model verdict remains **NOT APPLICABLE**.

**Evidence:** Alpine 3.20.10 historical environment reconstructed with 137/137 package-version match; historical binary reproduced bit-for-bit; 52/52 frozen fields and the three primary artifacts reproduce exactly. See `validators/hem_v27/r0/historical-reconciliation-2026-10-21/`.

**Preserved contrary evidence:** Debian 12 remains 50/52 and is classified as cross-environment numerical portability sensitivity evidence. Its raw evidence and failure verdict are not deleted.

**Bounded scope:** v26.4 is `[DERIVED][HISTORICAL][REPRODUCIBLE]` CLASS-level sensitivity/degeneracy evidence only. This does not establish a posterior, solve the Hubble tension, or validate HEM/Tech-Orb physics.

**Consequence:** R1 may start only after its baseline specification, tolerance contract, audit checklist, and manifest skeleton are frozen before numerical inspection.

## D-2026-10-21-09 — Freeze R1 baseline audit before execution

**Status:** Executed after pre-registration; PASS pending PR final review.

**Decision:** Separate the historical-v26.4 RECFAST / implicit-`N_ncdm=0` control from the canonical-modern HyRec / explicit-one-species `sum m_nu = 0.06 eV` adjudicative baseline. Freeze inputs, audited quantities, tolerances and evidence requirements before viewing R1 results.

**Rules:** No target `H0` gate; no parameter adjustment after inspection; HyRec-versus-RECFAST is diagnostic only; R1 PASS is not physical validation.

**Evidence:** `validators/hem_v27/r1/` and Issue #5.

**Adjudication:** Historical control artifacts are exact. Two pristine-source canonical HyRec runs are byte-identical across all declared raw output families and all frozen identity/consistency checks pass. The first dirty-source attempt is preserved and excluded. Pipeline verdict: **PASS**. Model verdict: **NOT APPLICABLE**.

## Decision Entry Template

- ID / date
- status and decision owner
- changed claim or gate
- previous state
- new state
- evidence links and exact versions
- scope and unresolved objections
- rollback/supersession condition

## D-2026-10-26-10 — Initialize the Relational Quantum Foundation and freeze QF-0A protocol

**Status:** MERGED / GOVERNANCE BASELINE on `main` at `4c2920871898add6254902fb3963784e2761d375` through PR #9. This status records governance acceptance, not scientific validation.

**Decision:** Make Global Constrained Relational Quantum Structure / Relational Quantum Foundation the active Tech-Orb research line. Register the working ontology `Q=(H,A,rho,K,T)`, one-way Foundation → effective model → validators governance, the five-label Foundation vocabulary, twelve global falsification gates, and non-skipping rule.

**Frozen scope:** Foundation Canon v1.0, Claim Registry v1.0, QF-0A known-result reproduction protocol, QF-0B/C benchmark skeleton, and Literature & Collision Matrix skeleton.

**Explicit non-actions:** No QF-0A numerical implementation/execution; no new-physics claim; no HEM R2 repeat/amendment or R3/R4; no PFM change/resume; no Taiji, BH-WH, or cyclic-topology work; no rewrite of historical evidence.

**Superseded active framing:** `foundation/FOUNDATION_V1_SPEC.md` remains preserved as historical/provisional Foundation v1 evidence but no longer defines the active research direction if this decision is merged.

**Evidence:** `foundation/qf0/` and `QF0_FOUNDATION_INITIALIZATION_REPORT_2026-10-26.md`.

**Next gate:** independent review of this initialization package. After acceptance, authorize a separate implementation branch for QF-0A v1.0 that is byte-bound to the frozen protocol before any result inspection.

**Independent review repair:** PR #9 received `REVISE`. The same Draft PR now records HEM R2 authoritative-main versus unmerged-PR8 candidate provenance, restores QF-0D as relational spacetime/causality below QF-1, replaces the ambiguous Section 9 eigenvector selection with a fixed basis-independent near-kernel projector continuation, renames the finite-clock suite as a finite-resolution ideal-clock diagnostic, and freezes the clock-scramble permutation. No numerical/scientific execution occurred; re-review is required.

**Closure:** The repaired head `44b76793d8dce3a2a29eecf132bf649e1de19a24` received independent APPROVE and was squash-merged as `4c2920871898add6254902fb3963784e2761d375`. GitHub reported no configured/status checks; this was recorded as `none configured/reported`, not CI PASS.

## D-2026-10-26-11 — Propose QF-0A implementation preregistration v1.0

**Status:** MERGED / CONTRACT BASELINE / NOT EXECUTED through PR #10 at `2d125671c0d882df6d1de4eb2a0066d5bad989de`.

**Decision:** Freeze the future QF-0A implementation source-tree boundary, CPython/runtime and minimal dependency policy, `float64`/`complex128` numeric policy, backend/environment identity, determinism, evidence directory, hashing, leakage controls, raw-output index, negative controls, and machine-verdict separation before scientific source or adjudicative output exists.

**Protocol binding:** `foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.0.md`; Git blob `c11599a058caa8dfb0912c4e00a0656bef4688bd`; SHA-256 `fb7969c11e9e2d4b1eebdd2495692dc422eca418bbbaa85bbabcda2815811d8b`; authoritative main `4c2920871898add6254902fb3963784e2761d375`.

**Explicit non-actions:** No scientific implementation modules, conditional states, fidelity, trace distance, interaction results, clock-choice results, raw arrays, evidence run, or scientific/machine verdict instance was produced. QF-0B/C/D, QF-1/2/3, HEM, PFM, and topology remain unchanged.

**Evidence:** `foundation/qf0/QF0A_IMPLEMENTATION_PREREGISTRATION_v1.0.md` and `foundation/qf0/qf0a_impl/`.

**Next gate:** independent read-only review of this Draft PR. A separately authorized implementation gate is required before source completion or any scientific/numerical execution.

## D-2026-10-27-12 — Propose QF-0A Environment + Source Freeze v1.0

**Status:** MERGED / ACCEPTED / NOT EXECUTED through PR #11. Approved head `fda04be2e00494599689e86065674dd7d185ca40`; merge/main `ffa2a74592596e7c7e64e1a188061c422c71b260`.

**Decision:** Bind the approved preregistration to CPython 3.12.10, NumPy 2.1.3, SciPy 1.14.1, hashed Windows AMD64 wheels, OpenBLAS identities, single-thread settings, and a reviewed scientific-source inventory. The analytic and conditional paths are structurally separate; the exact six controls are defined but not run.

**Explicit non-actions:** No frozen conditional-state orbit, fidelity, trace distance, interaction suite, clock suite, scientific negative control, evidence run, or real machine verdict was generated or inspected.

**Accepted identities:** protocol blob `c11599a058caa8dfb0912c4e00a0656bef4688bd`, protocol SHA-256 `fb7969c11e9e2d4b1eebdd2495692dc422eca418bbbaa85bbabcda2815811d8b`, environment manifest SHA-256 `6c968cf7a036989efca495fa882fccc8b829d9ddd9ccf7960d849c4d24727791`, dependency lock SHA-256 `9ec9da7aaa0f7a7b85d9cb33909d03b0b934f0948c0fbb1c1182274e2edea0b2`, scientific source baseline `946965d47fa66d0822cd8023534ee282fddc8f98`, and source manifest SHA-256 `c8d06cb95095a022870152115c591774c64f4df73e54c5a576054e7ad4d35f27`.

**Next gate:** QF-0A External Logic & Scientific Challenge Review — PENDING. Scientific execution remains unauthorized.

## D-2026-10-27-13 — Require external logic and scientific challenge before execution authorization

**Status:** PENDING / GOVERNANCE GATE / EXECUTION UNAUTHORIZED.

**Decision:** Obtain an independent cross-model challenge review by Claude before the first adjudicative run. The review must challenge QF-0A mathematics, epistemic interpretation, whether the protocol tests its stated target, implementation/protocol alignment, hidden circularity or answer leakage, falsifiability, negative-control design, and whether execution should proceed unchanged.

**Authority boundary:** Claude review is advisory independent evidence, not GitHub SSOT by itself. Scientific execution remains unauthorized until Terry/ChatGPT adjudicates the review and issues a separate Scientific Execution Authorization.

**Explicit non-actions:** No runner activation, frozen reproduction, resolution/interaction/clock suite, real negative control, evidence run, raw array, machine verdict, fidelity, or trace-distance calculation is authorized by this synchronization.

## D-2026-10-28-14 — Accept external REVISE and propose QF-0A v1.1 pre-execution re-freeze

**Status:** PROPOSED / PRE-EXECUTION REPAIR / NOT EXECUTED; execution UNAUTHORIZED.

**Decision:** Accept the independent challenge review as a repair input and block execution pending a versioned v1.1 protocol, preregistration, environment/source/adjudication freeze, and independent read-only re-review. v1.0 remains historical and never executed.

**Pre-result disclosures:** (1) the ideal primary Page–Wootters conditional relation is an exact DFT-based algebraic identity, so expected numerical deviation is machine-precision scale and output is software qualification rather than an unknown prediction; (2) the frozen history-state representation is mathematically equivalent to its registered analytic history representation, so final output cannot distinguish direct constrained construction from equivalent external-time packaging—the rule is a source/provenance attestation; (3) for each registered nonzero `g`, the exact kernel is one-dimensional with stationary `|0,0>` zero mode, and the projected continuation is a near-kernel geometry diagnostic rather than an exact constrained physical state; (4) Clock B tests only registered relabeling, rephasing invariance of POVM elements, and index bookkeeping.

**Evidence boundary:** A future QF-0A PASS is pipeline/software qualification and constructibility evidence only. It is not supporting evidence for time in nature, the Foundation ontology, or QF-0D.

**Execution boundary:** No scientific authorization artifact exists. No registered conditional states, metrics, diagnostic suite, controls, evidence run, or machine verdict may be generated before the v1.1 package passes independent review and a separate authorization is issued.

**Next gate:** Independent read-only review of QF-0A v1.1 Pre-Execution Repair / Re-Freeze.

