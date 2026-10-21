# HEM v27 R1 — Baseline Definition Audit

Status: **PASS; NUMERICAL AUDIT COMPLETED 2026-10-21**

R1 asks whether the reference cosmology is correctly and unambiguously defined. It contains two strictly separated tracks:

1. `historical-v26.4`: the original runner semantics with implicit `N_ncdm=0` and RECFAST, used only as a historical control.
2. `canonical-modern`: an explicit modern baseline with one 0.06 eV massive neutrino and HyRec, used for the R1 gate.

The binding specification is `BASELINE_SPEC.md`; the only acceptance criteria are in `TOLERANCE_CONTRACT.md`. `AUDIT_CHECKLIST.md` defines required evidence and `MANIFEST.json` is the pre-execution evidence skeleton.

No target `H0` is a PASS criterion. R1 pipeline PASS would establish only a defined and reproducible baseline. The model verdict is `NOT APPLICABLE` or `UNDETERMINED`; R1 cannot validate HEM, Tech-Orb, a cosmological posterior, or a Hubble-tension solution.

Tracking: GitHub Issue #5.

The final report and compact verification records are in `evidence-2026-10-21/`. The full 1,080-file evidence workspace and verified SHA-256 manifest remain under `H:\\AI_Products\\AI_Factory_Workspace\\TechOrb-Hubble-Validator-R1-baseline-audit-20261021`.
