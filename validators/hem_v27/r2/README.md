# HEM v27 R2 — Published Model Reproduction

Status: **REFERENCE IDENTITY RE-FROZEN — NUMERICAL EXECUTION AUTHORIZED BUT NOT YET RUN**

R2 will test whether the repository can reproduce a published baryon-clumping recombination benchmark before any novel HEM calculation. The selected primary target is the fixed-cosmology M3 three-zone benchmark of Rashkovetskyi et al. (2021), especially Figs. 2–3 and the RECFAST track of Fig. 18.

No author raw table was located, but the official arXiv v3 source contains standalone vector PDFs for Figs. 2, 3 and 18. Two independent figure extractions have now frozen auditable Fig. 2/3 reference CSVs, row masks and geometry-derived tolerances. No local model output has been generated or inspected.

The 2026-10-22 identity repair verified that the manifest hashes identify the LF Git blobs already present in the original pre-execution commit. Windows CRLF working-tree/archive hashes are transport representations, not replacement references. See `R2_REFERENCE_IDENTITY_REPAIR_REPORT.md`.

Binding files: `BENCHMARK_SELECTION.md`, `BENCHMARK_SPEC.md`, `EQUATION_TRANSCRIPTION.md`, `REFERENCE_RECOVERY_REPORT.md`, `REFERENCE_DATA_MANIFEST.md`, `FIG2_REFERENCE.csv`, `FIG3_REFERENCE.csv`, `DIGITIZATION_METHOD_A.md`, `DIGITIZATION_METHOD_B.md`, `DIGITIZATION_UNCERTAINTY.md`, `TOLERANCE_CONTRACT.md`, `AUDIT_CHECKLIST.md`, `ENVIRONMENT_SPEC.md`, and `MANIFEST_SCHEMA.json`.

Tracking: GitHub Issue #7. R2 PASS would mean only that this published benchmark was reproduced; it would not validate HEM, Tech-Orb, PMFs, or a Hubble-tension solution.
