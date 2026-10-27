# Evidence Index

This index distinguishes current control documents from preserved historical evidence. Inclusion is not endorsement.

## Current Control and Specifications

| Evidence ID | Path | Role | Status |
|---|---|---|---|
| CTRL-STATE | `control/PROJECT_STATE.md` | Integrated current state | Governance approved in PR #2; scientific claims retain their ledger status |
| CTRL-GOV | `control/SCIENTIFIC_GOVERNANCE.md` | Claim and gate rules | Governance approved in PR #2; scientific claims retain their ledger status |
| CTRL-LEDGER | `control/EPISTEMIC_LEDGER.md` | Claim registry | Governance approved in PR #2; scientific claims retain their ledger status |
| FND-SPEC | `foundation/FOUNDATION_V1_SPEC.md` | Foundation scope | PROVISIONAL |
| FND-REQ | `foundation/MATHEMATICAL_REQUIREMENTS.md` | Minimum mathematical/physical requirements | PROVISIONAL |
| VAL-HEM27 | `validators/hem_v27/GATE_PROTOCOL.md` | Canonical validator protocol | PROVISIONAL |
| QF0-CANON-20261026 | `foundation/qf0/FOUNDATION_CANON_v1.0.md` | Active Relational Quantum Foundation governance proposal | FROZEN; no scientific result |
| QF0-CLAIMS-JSON-20261026 | `foundation/qf0/CLAIM_REGISTRY_v1.0.json` | Machine-readable QF claim registry | FROZEN; structurally validated |
| QF0-CLAIMS-MD-20261026 | `foundation/qf0/CLAIM_REGISTRY_v1.0.md` | Human-readable QF claim registry | FROZEN; no claim PASS |
| QF0A-PROTOCOL-20261026 | `foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.0.md` | Relational-time known-result reproduction protocol | FROZEN; NOT EXECUTED |
| QF0BC-SKELETON-20261026 | `foundation/qf0/QF0BC_BENCHMARK_SPEC_SKELETON_v1.0.md` | Distance/dimension blind benchmark architecture | SKELETON; NOT EXECUTED |
| QF0-LIT-JSON-20261026 | `foundation/qf0/LITERATURE_COLLISION_MATRIX_v1.0.json` | Machine-readable literature/collision skeleton | EMPTY SKELETON |
| QF0-LIT-MD-20261026 | `foundation/qf0/LITERATURE_COLLISION_MATRIX_v1.0.md` | Human template and domain taxonomy | EMPTY SKELETON |
| QF0-REPORT-20261026 | `QF0_FOUNDATION_INITIALIZATION_REPORT_2026-10-26.md` | Scope, validation, limitations, and next gate | GOVERNANCE REPORT; no scientific result |
| QF0A-IMPL-PREREG-20261026 | `foundation/qf0/QF0A_IMPLEMENTATION_PREREGISTRATION_v1.0.md` | Runtime, numeric, source, evidence, leakage, and verdict contracts | PROPOSED / DRAFT / NOT EXECUTED |
| QF0A-IMPL-SCHEMAS-20261026 | `foundation/qf0/qf0a_impl/schemas/` | Run manifest, machine verdict, raw-output index, and negative-control JSON Schemas | DRAFT CONTRACTS; no result instances |
| QF0A-LEAKAGE-20261026 | `foundation/qf0/qf0a_impl/LEAKAGE_AUDIT_CHECKLIST.md` | External-time, analytic-target, post-hoc, exclusion, scramble, window, and cross-program leakage controls | DRAFT CHECKLIST; NOT EXECUTED |
| QF0A-STATIC-VALIDATOR-20261026 | `foundation/qf0/qf0a_impl/tests/validate_preregistration.py` | Protocol/hash/schema/status and forbidden-output structural checks | STATIC VALIDATION ONLY; no science |

## Historical HIA / Late-Time Evidence

| Evidence ID | Path | What it contains | Current use |
|---|---|---|---|
| HIA-OPT-245 | `core_engine/hia_joint_optimizer.py` | Simplified physical-density-locked optimizer | HISTORICAL implementation; not a likelihood pipeline |
| HIA-WP-245 | `docs/HIA_v24.5_Whitepaper.md` | v24.5 claims and motivating ansatz | HISTORICAL; overclaims superseded |
| HIA-PAPER-25 | `papers/HIA_NoGo_Theorem/manuscript.tex` | Late-time exclusion manuscript | HISTORICAL exclusion study |
| HIA-CODE-25 | `papers/HIA_NoGo_Theorem/Evidence_Code_v25.py` | Disformal ray-tracing code | HISTORICAL implementation evidence |
| HIA-AUDIT-25 | `papers/HIA_NoGo_Theorem/Numerical_Audit_Log.md` | Iteration/failure narrative | HISTORICAL; scoped by exclusion map |
| HIA-PLOT-25 | `papers/HIA_NoGo_Theorem/hia_disformal_raytrace.png` | Stored plot | HISTORICAL output; provenance not yet frozen under R0 rules |
| LT-MAP | `validators/late_time/EXCLUSION_MAP_V1_V25.md` | Current bounded interpretation | PROVISIONAL synthesis |

## Historical RMP / Origin Material

| Evidence ID | Path | Current classification |
|---|---|---|
| RMP-ORIGIN | `archive_RMP_v4/Origin_Protocol/` | HISTORICAL and speculative; not current Foundation canon |
| RMP-WHITEPAPER | `archive_RMP_v4/Origin_Protocol/ORIGIN_WHITE_PAPER.md` | HISTORICAL ANSATZ |
| RMP-FRAMEWORK | `archive_RMP_v4/Origin_Protocol/THEORY_FRAMEWORK.md` | HISTORICAL; validation/solution claims superseded |

## Review Provenance

| Evidence ID | Path | Use restriction |
|---|---|---|
| REVIEW-4AI-2026-10 | `reviews/2026-10-four-ai-review/README.md` | Critique provenance only; AI agreement is not evidence |

## HEM v27 R0 Candidate Recovery — 2026-10-20

| Evidence ID | Path | Status / scope |
|---|---|---|
| R0-RECOVERY-20261020 | `validators/hem_v27/r0/recovery-2026-10-20/README.md` | Inspected candidate files; historical producer identity unproved |
| R0-MANIFEST-20261020 | `validators/hem_v27/r0/recovery-2026-10-20/MANIFEST.json` | SHA256 and storage locations; larger files local only |
| R0-SOURCE-20261020 | `validators/hem_v27/r0/recovery-2026-10-20/sources/` | Byte-preserved runner/notebook/patched RECFAST source |
| R0-DIFF-20261020 | `validators/hem_v27/r0/recovery-2026-10-20/CLASS_FULL_DIFF.patch` | Complete tracked diff against aa92943e4ab86b56970953589b4897adf2bd0f99 |
| R0-RESULTS-20261020 | `validators/hem_v27/r0/recovery-2026-10-20/historical_outputs/` | Historical summary CSV; REPRO-BLOCKED |
| R0-BASELINE-20261020 | `validators/hem_v27/r0/recovery-2026-10-20/BASELINE_COMPARISON.md` | Reconstructed inputs and explicit neutrino convention comparison; no numerical rerun |
| R0-ENV-20261020 | `validators/hem_v27/r0/recovery-2026-10-20/ENVIRONMENT_CHECK.json` | Docker server unavailable; clean room not run |
| R0-ENV-POST-REBOOT-20261020 | `validators/hem_v27/r0/recovery-2026-10-20/ENVIRONMENT_CHECK_POST_REBOOT.json` | Engine reachable; C:-hosted Docker data violates H:-only boundary; no run |
| R0-TOLERANCE-20261020 | `validators/hem_v27/r0/recovery-2026-10-20/TOLERANCE_CONTRACT.md` | Binding criteria frozen before first clean-room rerun |
| R0-CHECKS-20261020 | `validators/hem_v27/r0/recovery-2026-10-20/RECOVERY_CHECKS.json` | Notebook equality, ZIP contents and source-hunk match; no scientific acceptance |
| R0-REPORT-20261020 | `validators/hem_v27/r0/recovery-2026-10-20/R0_CLEANROOM_REPORT.md` | Completed clean-room report; R0 FAIL, model verdict NOT APPLICABLE |
| R0-CLEANROOM-20261020 | `validators/hem_v27/r0/recovery-2026-10-20/cleanroom/` | Dockerfile, scientific-only patch, harness, environment/control/verdict records, commands/logs/INIs, derived tables/figure and complete raw-file SHA-256 manifest |

## R0-H Historical Environment Reconciliation — 2026-10-21

| Evidence ID | Path | Status / scope |
|---|---|---|
| R0H-REPORT-20261021 | `validators/hem_v27/r0/historical-reconciliation-2026-10-21/R0H_FINAL_REPORT.md` | R0 PASS for historical reproducibility; model NOT APPLICABLE |
| R0H-FIELDS-20261021 | `validators/hem_v27/r0/historical-reconciliation-2026-10-21/FIELD_COMPARISON_CANDIDATE_B.csv` | 52/52 exact; preserves Debian 50/52 comparison |
| R0H-ENV-20261021 | `validators/hem_v27/r0/historical-reconciliation-2026-10-21/ENV_LOCK_CANDIDATE_B.json` | 137/137 historical package-version match |
| R0H-BINARY-20261021 | `validators/hem_v27/r0/historical-reconciliation-2026-10-21/HISTORICAL_BINARY_FORENSICS.md` | Bit-for-bit binary reproduction at historical path |

## Historical provenance and portability record retained after R0 PASS

- provenance tying candidate source and binary to the historical numerical run;
- original compiler/dependency/environment identities and build/run logs;
- original per-run INIs and raw CLASS outputs, deleted by the runner;
- full downstream baseline convention audit beyond the separate historical and explicit Σmν=0.06 eV tracks.

The R0-H package resolves the historical environment and producer identity for the adjudicated artifacts. The Debian 12 mismatch remains visible as portability-sensitivity evidence. R0 PASS is limited to reproducibility; R1–R4 remain separate gates.

## HEM v27 R1 Pre-registration — 2026-10-21

| Evidence ID | Path | Status / scope |
|---|---|---|
| R1-SPEC-20261021 | `validators/hem_v27/r1/BASELINE_SPEC.md` | Frozen two-track baseline definition; no numerical result |
| R1-TOL-20261021 | `validators/hem_v27/r1/TOLERANCE_CONTRACT.md` | Binding pre-execution acceptance contract |
| R1-CHECKLIST-20261021 | `validators/hem_v27/r1/AUDIT_CHECKLIST.md` | Required audit and evidence inventory |
| R1-MANIFEST-20261021 | `validators/hem_v27/r1/MANIFEST.json` | Completed manifest; pipeline PASS, model NOT APPLICABLE |
| R1-REPORT-20261021 | `validators/hem_v27/r1/evidence-2026-10-21/R1_AUDIT_REPORT.md` | Completed R1 PASS report; model NOT APPLICABLE |
| R1-RESULT-20261021 | `validators/hem_v27/r1/evidence-2026-10-21/R1_AUDIT_RESULT.json` | Machine-readable frozen-contract checks |
| R1-SHA256-20261021 | `validators/hem_v27/r1/evidence-2026-10-21/SHA256SUMS.txt` | Full H:-workspace manifest; 1,080/1,080 verified |
