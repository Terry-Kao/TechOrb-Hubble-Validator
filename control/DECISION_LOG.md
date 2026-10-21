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

## D-2026-10-21-10 — Freeze R2 reference data and tolerances before execution

**Status:** Proposed in Draft PR #8; numerical execution not performed.
**Decision:** Admit independently digitized Fig. 2/3 references from the author-submitted vector PDFs in official arXiv `2108.02747v3`; freeze row masks and pointwise tolerances before any R2 model output exists.

**Recovery result:** No author machine-readable Fig. 2/3 table or plotting script was located. Method A extracts vector paths; Method B independently traces a 1200 dpi raster. Only reference-vs-reference dispersion determines masks and uncertainty.

**Tolerance:** Fig. 2 `x_e` absolute `0.012`; Fig. 2 `g` absolute `0.00025 Mpc^-1`; Fig. 3 `log10(x_e)` absolute `0.045 dex`. Every admitted point must pass; no aggregate escape clause.

**License boundary:** The pinned author CLASS fork has no repository-level license. Do not commit its source bytes; retain local scientific inspection/execution only and record identities/hashes.

**Consequence:** R2 changes from reference-data BLOCKED to **REFERENCE ADMISSION COMPLETE — NUMERICAL EXECUTION AUTHORIZED BUT NOT YET RUN**. A separate work is required to execute and adjudicate R2.

**Evidence:** `validators/hem_v27/r2/REFERENCE_RECOVERY_REPORT.md`, `REFERENCE_DATA_MANIFEST.md`, `FIG2_REFERENCE.csv`, `FIG3_REFERENCE.csv`, digitization method/uncertainty records, and `TOLERANCE_CONTRACT.md`.

## Decision Entry Template

- ID / date
- status and decision owner
- changed claim or gate
- previous state
- new state
- evidence links and exact versions
- scope and unresolved objections
- rollback/supersession condition

