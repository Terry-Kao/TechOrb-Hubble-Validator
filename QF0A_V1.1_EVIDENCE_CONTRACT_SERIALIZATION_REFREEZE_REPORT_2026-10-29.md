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

## Exact inventory closure

The final independent review identified one remaining blocker: required-file existence was enforced, but unregistered extra files were not rejected. This closure establishes a single canonical ordered inventory and derives two immutable sets from it: 20 exact pre-verdict files and 22 exact final files. Both file sets and the allowed directory set (`raw/`, `logs/`) must match exactly. Unexpected root, raw, log, nested files, and unexpected directories fail closed without deletion or normalization.

`SHA256SUMS.txt` must contain exactly the 21 frozen non-checksum paths. Equality between checksum paths and whatever files happen to exist is not sufficient; both the actual final file set and checksum path set are independently compared with the frozen inventory. Regression tests reject a correctly hashed 23rd file, missing checksum entries, duplicates, aliases outside the frozen set, and hash mismatches.

A machine-verdict file is not an accepted qualification result unless the official process exits successfully and G3 final sealing completes successfully. This is a governance/evidence clarification only and changes no scientific semantics.

No schema, threshold, scientific control, physics/numerical definition, protocol byte, environment file, or dependency was changed. The fixed inventory remains 22 files.

## Re-freeze identities

- New source baseline: `80403cb383807334066eec482012ed42d4ded0ca`
- Source manifest entries: 40
- Source manifest canonical committed-byte SHA-256: `a86d73cd80094711a4790c2e6944a400e59fe7e8d8833e62b20f754a7e48a5ce`
- Protocol blob/SHA-256: `3af303a471d99397429d33f77ca512578f40f972` / `1cf9f7f3a8cfa6a5b0e12a3290d7e545f44586f9359aaa5367754415cdcc5f3a`
- Environment blob/SHA-256: `4bb6f73d5cfd96422829d3f0680ed52c9614e9e0` / `13bd8a95059363a5b2208f36e514e0e7d08b4467f4a0db35e2ce2bf37c41299a`
- Dependency lock SHA-256: `9ec9da7aaa0f7a7b85d9cb33909d03b0b934f0948c0fbb1c1182274e2edea0b2`

The follow-up commit after the source baseline is restricted to this report and `foundation/qf0/qf0a_impl/manifests/QF0A_SOURCE_MANIFEST_v1.1.json`; therefore baseline-to-final-head executable/source/schema/test drift is zero.

## Validation scope

Only static, schema, artificial, and synthetic integration validation is permitted. The tests exercise the actual evidence serializers and G2/G3 methods without invoking `qf0a.entrypoint`, `FrozenQF0ARunner.execute()`, or registered QF-0A science. Synthetic coverage includes all three repaired schemas, malformed or missing pre-verdict evidence, Stage-B mutation invalidation, malformed final verdict sealing failure, exact six-control completeness, exact committed source-manifest bytes, full synthetic package sealing, PASS-before-completion prohibition, no caller-supplied PASS path, exact 20/22 inventory counts, unexpected root/raw/log/nested paths, unexpected directories, correctly hashed extra checksum entries, missing/duplicate checksum paths, and hash mismatch.

## Gate

No `QUAL-002` authorization was created and no QF-0A run was executed. The exact next gate is **Independent narrow read-only re-review of QF-0A v1.1 Exact Evidence Inventory Closure**.
