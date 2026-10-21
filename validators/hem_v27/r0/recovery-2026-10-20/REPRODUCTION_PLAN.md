# Reproduction plan — pending clean-room environment

Status: BLOCKED; no numerical run in this recovery.

1. Provide a clean Linux build/runtime with source, caches, temporary files, logs and outputs confined to the authorized H: workspace (or a bind mount entirely within it). Record exact platform/image identity, compiler, make, Python, package versions and relevant environment. Do not capture secrets or unrelated environment values.
2. Create a separate checkout of CLASS commit `aa92943e4ab86b56970953589b4897adf2bd0f99`; never execute the historical runner against the recovered source directory because it resets a source file, cleans the build and deletes outputs.
3. Freeze a justified rerun tolerance contract before execution. Record tolerances for baseline z_star, r_s, theta, multipole grids, spectra/residuals and summary fields, including the limited printed precision used by the historical parser.
4. Verify the original runner/notebook/source hashes against `MANIFEST.json`. Apply the scientific hunk from `CLASS_FULL_DIFF.patch` to clean upstream source and verify it matches the recovered wrap_recfast.c. Record any intentional omission of the three non-scientific executable-bit changes.
5. Build clean upstream and patched source separately, retaining complete stdout/stderr, exit codes, flags, build commands, binary hashes and dependency metadata.
6. Run b=1, eta=0 on clean upstream and patched source; compare their outputs before enabling either modification.
7. Use a provenance-preserving harness derived from the frozen runner. Document and hash its full diff. Retain all generated INIs, raw spectra/background/thermodynamics, stdout/stderr and environment controls for every root-finding iteration and final run. Do not treat a modified harness as historical source.
8. Reproduce Mode1 (eta=0; b=1.40,1.60) and Mode2 (eta=0.5; b=1.35,1.40), including the runner's iso-theta root-finding and n_s/A_s compensation. Record both matches and deviations from the frozen historical CSV and local residual table.
9. Perform the separately named massive-neutrino comparison after the historical track, preserving both sets. Do not tune the historical configuration to remove a discrepancy.
10. Write an auditable gate report with source/environment/input/output identities, commands, tolerances, deviations and independent reproduction record. Only then may R0 pipeline status be adjudicated. Model verdict remains NOT APPLICABLE; downstream scientific claims require later gates.

An artifact-identity check or numerical match obtained in the original dirty source tree is insufficient for R0 PASS.
