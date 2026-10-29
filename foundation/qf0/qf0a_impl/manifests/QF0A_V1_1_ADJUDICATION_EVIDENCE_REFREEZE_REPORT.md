# QF-0A v1.1 Runtime Frozen-Source Identity Closure Re-Freeze

Status: `PRE-EXECUTION / NOT EXECUTED / INDEPENDENT REVIEW REQUIRED`

- Authoritative main: `a9e9a8ec0b637d4a724a6d3baaded83b54c0058e`
- Previous reviewed PR head: `c2a4e9fd5725a00a403e7eea35fbfe1a16083a1b`
- Repaired source baseline: `cca6707fd8410f355f9308b948eabf2efeac4c89`
- Protocol v1.1 changed: `NO`
- Protocol blob: `3af303a471d99397429d33f77ca512578f40f972`
- Protocol canonical SHA-256: `1cf9f7f3a8cfa6a5b0e12a3290d7e545f44586f9359aaa5367754415cdcc5f3a`
- Preregistration v1.1 blob at baseline: `4e303e39abe9f389257eb5e91e607a668ff25f8f`
- Preregistration v1.1 canonical SHA-256: `5c3ce46eac8459097754c2e52902cd1b785ea77c8fb7264d0683a1db113585e2`
- Source manifest: `QF0A_SOURCE_MANIFEST_v1.1.json`
- Source-manifest entries: `38`
- Source-manifest SHA-256: `6f603981cee63e120ca05a8fba377fe9fd9838c5d97b9cfb9126f61547e328fd`

The source baseline adds a stdlib-only pre-import provenance verifier and a fresh-process official entry point. Authorization binds the committed HEAD source-manifest SHA before its contents are trusted; every manifest entry is then checked at current HEAD by Git blob, byte length, and canonical SHA-256. The active source/schema/environment inventory must equal the manifest-derived inventory, the current runner/protocol/preregistration/environment must equal authorization identities, and baseline-to-HEAD changes are restricted to the manifest and this re-freeze report.

Validation at the baseline was static and synthetic only. Foundation validation, preregistration validation, Draft 2020-12 schemas with `FormatChecker`, synthetic fixtures, and 39 tests passed. Synthetic Git regressions reject a later clean source commit, an added alternate module, a changed-source plus changed-manifest attack, and a clean runner mutation. Python compilation, JSON parsing, canonical committed-byte verification, and `git diff --check` passed. No registered QF-0A scientific or adjudicative run was performed or inspected.

The manifest/report commit following this baseline may change only this source manifest and this report. The next gate is independent read-only final re-review of QF-0A v1.1 Runtime Frozen-Source Identity Closure. This report does not authorize scientific execution or merge.

## Post-governance execution-readiness re-freeze — 2026-10-29

Status: `PRE-EXECUTION / NOT EXECUTED / INDEPENDENT REVIEW REQUIRED`

- Accepted governance main used for preflight: `13ce10f447f377234648e94f2779e23502e8eeaa`
- Post-governance repair source baseline: `3c144f0f523800973bcff68f40d59a1e34ed6932`
- Protocol v1.1 changed: `NO`
- Preregistration v1.1 changed: `NO`
- Environment/dependency contract changed: `NO`
- Run-manifest schema changed: `YES` — the stale fixed authoritative-main SHA was replaced by a validated Git-SHA field so generated metadata can record the validated execution HEAD.
- Source manifest: `QF0A_SOURCE_MANIFEST_v1.1.json`
- Source-manifest entries: `39`
- Source-manifest SHA-256: `ed504e4ddec500de69ea5d2834b532ad17d8f45a5a1d8f8a4489d6a12b132e09`

The repair preserves exact manifest identity for every active protocol, preregistration, source, schema, and v1.1 environment path. Post-baseline governance changes are confined to the explicit six-path allowlist: `README.md`, `control/DECISION_LOG.md`, `control/EVIDENCE_INDEX.md`, `control/PROJECT_STATE.md`, the v1.1 source manifest, and this report. No wildcard or directory exemption exists.

Generated run metadata now takes the authoritative execution commit from the already validated `ExecutionSourceSeal.current_execution_head`. The historical source baseline remains separately recorded as `ExecutionSourceSeal.source_baseline_commit`. Attached HEAD records the actual symbolic branch name; detached HEAD records `DETACHED`; and a HEAD that differs from the validated seal is rejected.

Validation was static and synthetic only. Foundation validation passed. Preregistration validation passed with five Draft 2020-12 schemas plus `FormatChecker`, 8/8 positive fixtures, and 103/103 expected negative-fixture rejections. All 48 unit tests passed, including exact accepted-governance allowlist acceptance, rejection of unlisted governance and active source/schema/environment mutations, validated-head metadata, truthful attached/detached branch metadata, stale-metadata rejection, and provenance readiness without importing QF-0A scientific modules. Python compilation, JSON parsing, canonical committed-byte reconstruction, and `git diff --check` passed.

No execution authorization, evidence directory, registered conditional state, fidelity, trace distance, dimension/interaction/clock/control suite, machine verdict, or QF-0A scientific execution was created or performed. The manifest/report follow-up must contain zero executable/source/schema/test drift from the source baseline. The exact next gate is **Independent read-only review of QF-0A v1.1 Post-Governance Execution-Readiness Repository Repair / Re-Freeze**. This re-freeze does not authorize scientific execution or merge.
