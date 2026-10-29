# QF-0A v1.1 Evidence Contract Serialization Repair / Re-Freeze

Date: 2026-10-29
Status: **DRAFT / PROSPECTIVE REPAIR / NO SCIENTIFIC RE-EXECUTION**

## Historical incident and preservation

`QF0A-v1.1-QUAL-001` remains immutable historical evidence of the first qualification attempt. It executed once at `f804e8cc0d553bd33ee71c2c96479f035a07e44b`; the frozen runner reported 22 numerical gates passed, six expected control failures, raw machine verdict `PASS`, and foundation verdict `NOT_APPLICABLE`. External governance adjudication remains **EVIDENCE CONTRACT BLOCKED** because `raw_output_index.json`, `negative_controls.json`, and `machine_verdict.json` failed their schemas. There was no retry, no accepted evidence PR, and this repair does not alter any QUAL-001 byte.

The original 22-file evidence directory was independently inventoried and copied outside the Git working tree. Its `SHA256SUMS.txt` SHA-256 is `6c4b81b8d2ceca6e878420565986f6972eadd220719b6da71d8f802eace62788`.

## Prospective repair

- `raw_output_index.json` now comes from one canonical serializer. It emits the frozen nine-array order with deterministic section/adjudicative metadata, complete typed identities, count equality, no exclusions, and the frozen protocol SHA.
- `negative_controls.json` now contains exactly six typed objects. Each points to the already-written immutable `diagnostics.json`, whose hash is captured before the controls record is written; this avoids circular dependence on the verdict or checksum manifest.
- `machine_verdict.json` now contains every schema-required field and projects completeness, control inventory, and leakage status from the final adjudication record.
- G2 validates authorization, run manifest, raw-output schema and array hashes, negative-control schema and referenced evidence hash, fixed pre-verdict inventory, immutable Stage-B hashes, exact committed source-manifest bytes, and leakage status before F2 can derive PASS.
- G3 writes the verdict and checksum manifest, then validates the machine-verdict schema, exact 22-file inventory, checksum coverage, uniqueness, and every sealed file hash without rewriting the verdict.
- Future `source_manifest.json` evidence is copied from exact committed bytes. Its evidence-copy SHA therefore equals the authorization-bound committed-manifest SHA; G2 checks both identities.

No schema, threshold, scientific control, physics/numerical definition, protocol byte, environment file, or dependency was changed. The fixed inventory remains 22 files.

## Re-freeze identities

- New source baseline: `5768ab55395f7e0edc8137a9dc2cc4f57d7b264d`
- Source manifest entries: 40
- Source manifest canonical committed-byte SHA-256: `b1415aa5ae92e1a73ad0bbfc1de555eec81eda103fd6f12adef81fb91dd6b90f`
- Protocol blob/SHA-256: `3af303a471d99397429d33f77ca512578f40f972` / `1cf9f7f3a8cfa6a5b0e12a3290d7e545f44586f9359aaa5367754415cdcc5f3a`
- Environment blob/SHA-256: `4bb6f73d5cfd96422829d3f0680ed52c9614e9e0` / `13bd8a95059363a5b2208f36e514e0e7d08b4467f4a0db35e2ce2bf37c41299a`
- Dependency lock SHA-256: `9ec9da7aaa0f7a7b85d9cb33909d03b0b934f0948c0fbb1c1182274e2edea0b2`

The follow-up commit after the source baseline is restricted to this report and `foundation/qf0/qf0a_impl/manifests/QF0A_SOURCE_MANIFEST_v1.1.json`; therefore baseline-to-final-head executable/source/schema/test drift is zero.

## Validation scope

Only static, schema, artificial, and synthetic integration validation is permitted. The tests exercise the actual evidence serializers and G2/G3 methods without invoking `qf0a.entrypoint`, `FrozenQF0ARunner.execute()`, or registered QF-0A science. Synthetic coverage includes all three repaired schemas, malformed pre-verdict evidence, missing evidence, Stage-B mutation invalidation, malformed final verdict sealing failure, exact six-control completeness, exact committed source-manifest bytes, full synthetic package sealing, PASS-before-completion prohibition, and no caller-supplied PASS path.

## Gate

No `QUAL-002` authorization was created and no QF-0A run was executed. The exact next gate is **Independent read-only review of QF-0A v1.1 Evidence Contract Serialization Repair / Re-Freeze**.
