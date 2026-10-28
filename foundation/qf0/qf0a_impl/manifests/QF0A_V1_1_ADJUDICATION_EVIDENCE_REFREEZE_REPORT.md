# QF-0A v1.1 Adjudication & Evidence Contract Re-Freeze

Status: `PRE-EXECUTION / NOT EXECUTED / INDEPENDENT REVIEW REQUIRED`

- Authoritative main: `a9e9a8ec0b637d4a724a6d3baaded83b54c0058e`
- Previous reviewed PR head: `7d6a7e7e119f0e791464157fa9ab1c93f163525c`
- Repaired source baseline: `68359bca1ee90bf5b6662f46f3e2f51f6b3a8bd6`
- Protocol v1.1 changed: `NO`
- Protocol blob: `3af303a471d99397429d33f77ca512578f40f972`
- Protocol canonical SHA-256: `1cf9f7f3a8cfa6a5b0e12a3290d7e545f44586f9359aaa5367754415cdcc5f3a`
- Preregistration v1.1 blob at baseline: `af15d76fbe8991e22f054b71c200de484470f7b7`
- Preregistration v1.1 canonical SHA-256: `c3bc076e0bc4498c7e5cc87f96164185f4378bb538c3898d0be9e1c09075d4d6`
- Source manifest: `QF0A_SOURCE_MANIFEST_v1.1.json`
- Source-manifest entries: `35`
- Source-manifest SHA-256: `5044510d628e7d3b3dc1334388c9569f307575ed714a4ccdacdce0c33e9fdd35`

The source baseline closes the named 22-threshold wiring, required diagnostics, run-manifest schema gate, deterministic human-report contract, write-once raw evidence, conditional Stage-B hash lock, evidence-before-final-verdict ordering, exact-byte `source_manifest.sha256`, Section 9 completeness, and synthetic/static regression coverage.

Validation at the baseline was static and synthetic only. Foundation validation, preregistration validation, Draft 2020-12 schemas with `FormatChecker`, synthetic fixtures, 34 source/adjudication/evidence tests (including direct validation of the runner-generated synthetic run manifest), Python compilation, JSON parsing, and `git diff --check` passed. No registered QF-0A scientific or adjudicative run was performed or inspected.

The manifest/report commit following this baseline may change only this source manifest and governance/report artifacts. The next gate is independent read-only review by ChatGPT/Terry. This report does not authorize scientific execution or merge.
