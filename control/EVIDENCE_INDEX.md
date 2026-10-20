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
| R0-CHECKS-20261020 | `validators/hem_v27/r0/recovery-2026-10-20/RECOVERY_CHECKS.json` | Notebook equality, ZIP contents and source-hunk match; no scientific acceptance |

## Remaining Evidence Blocking R0

- provenance tying candidate source and binary to the historical numerical run;
- original compiler/dependency/environment identities and build/run logs;
- original per-run INIs and raw CLASS outputs, deleted by the runner;
- baseline convention audit, including N_ncdm=0 versus Σmν=0.06 eV;
- available clean-room environment, frozen rerun tolerances and independent rerun.

Tracking: [issue #3](https://github.com/Terry-Kao/TechOrb-Hubble-Validator/issues/3). R0 remains BLOCKED / NOT PASSED.
