# HEM v27 R0 recovery — 2026-10-20

**Pipeline verdict: BLOCKED. Model verdict: NOT APPLICABLE.**

This package freezes candidate v26.4 artifacts actually inspected under `H:\AI_Products\AI_Factory_Workspace\TechOrb-Hubble-Validator`. The recovery resolves source/file identity for this candidate set. It does not prove which executable, build environment, or source bytes produced the historical CSV; missing run/build logs prevent that attribution. The controlling work item is [issue #3](https://github.com/Terry-Kao/TechOrb-Hubble-Validator/issues/3).

## Frozen material

- Original runner and unexecuted notebook, preserved byte-for-byte using Git blobs.
- Actual recovered RECFAST source, with its copyright/permission notice intact.
- Complete tracked CLASS working-tree diff, including three executable-bit changes.
- Small historical summary CSV; the larger residual CSV, PNG, and ZIP remain local with hashes in `MANIFEST.json`.
- The upstream Planck 2018 example INI and two clearly labelled reconstructed baseline inputs.
- Recovery checks, baseline comparison, environment check, and blockers below.

The upstream tag `v3.2.0` resolves on GitHub to `aa92943e4ab86b56970953589b4897adf2bd0f99`, matching the candidate nested checkout. The top-level local TechOrb directory has no Git metadata. The full diff changes scientific code only in `external/RecfastCLASS/wrap_recfast.c`; `CPU.py`, `doc/input/make1.sh`, and `doc/input/make2.sh` change mode 100755→100644 only.

## Recovery checks

- Runner SHA256: `260bbe8642a9a7d6b0c6ad1734697e4087990d8ffb2c305145fbbf2274ddfb9e`.
- Notebook code cell equals the runner text exactly. Execution count is null and stored outputs are empty; it supplies no runtime log.
- Summary CSV has the four declared Mode1/Mode2 cases. Its numbers are historical, not independently reproduced.
- ZIP inspection finds only the summary CSV, residual CSV, and PNG; it contains no raw CLASS output, generated per-run INI, or build/run log.
- No `.log`, `.out`, `.txt`, `.pre`, or precision-named files were found outside the nested CLASS trees in the inspected Colab subtree.
- The runner invokes CLASS without a separate precision file. Exact defaults are pinned by the CLASS commit and local `include/precisions.h` SHA256.
- No recovered file was executed or modified. No task artifact was created on C: or another drive.

## Blockers and stop point

1. **ENV-01:** WSL lists only `docker-desktop`. Docker client 29.2.1 returns Server=null and cannot connect to `dockerDesktopLinuxEngine`; the Linux engine pipe is absent. No clean-room runtime is available through the inspected tools. No service was started or software installed.
2. **PROV-01:** Original compiler/dependency versions, build log, runtime environment, executable identity, and source-to-output linkage are missing.
3. **RAW-01:** The runner deletes generated per-run INIs and raw CLASS files, and captures stdout without archiving it. Only derived outputs were recovered.
4. **BASE-01:** Historical neutrino settings inherit `N_ncdm=0` and `N_eff=3.044`. They differ from the explicit massive-neutrino Planck example. See `BASELINE_COMPARISON.md`.
5. **TOL-01:** Historical tolerances were not frozen. A justified comparison contract must be recorded before a new independent rerun and must not be tuned after seeing rerun deviations.

R1–R4 remain NOT STARTED. A recovered file/hash check cannot pass R0. Once a clean Linux environment with all writes confined to the authorized H: workspace is available, follow `REPRODUCTION_PLAN.md` and retain every input, environment setting, build/run log, and raw output.

## Historical wording

The original runner contains historical comments such as “BBN absolute safe region,” approximate `B(b)` labels, and “Planck 2018 standard baseline.” These bytes are preserved for provenance. They are superseded as scientific claims by the governance reset and the baseline audit. The effective clumping/magnetogenesis relation remains a Candidate Bridge / ANSATZ.

## Attribution

CLASS upstream: [fixed commit](https://github.com/lesgourg/class_public/tree/aa92943e4ab86b56970953589b4897adf2bd0f99). CLASS requests citation of [CLASS II](https://arxiv.org/abs/1104.2933) in publications. The RECFAST source snapshot retains its original University of British Columbia notice. Recovery annotations and reconstructed configs are distinct from original artifacts.
