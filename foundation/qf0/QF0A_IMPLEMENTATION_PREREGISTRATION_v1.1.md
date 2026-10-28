# QF-0A Implementation Preregistration v1.1

**Status:** `PROPOSED / PRE-EXECUTION REPAIR / NOT EXECUTED`  
**Authoritative pre-repair main:** `a9e9a8ec0b637d4a724a6d3baaded83b54c0058e`

## Frozen identities

- Protocol: `foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.1.md`
- Protocol Git blob: `3af303a471d99397429d33f77ca512578f40f972`
- Protocol canonical committed-byte SHA-256: `1cf9f7f3a8cfa6a5b0e12a3290d7e545f44586f9359aaa5367754415cdcc5f3a`
- Canonical hash policy: `git-committed-bytes-v1`; authoritative bytes are `git show <commit>:<path>`, never checkout-transformed bytes.
- `.gitattributes` blob: `3565f993726838565e32016941c4ee2fec65a65a`; canonical SHA-256: `54f889317ed67cfd9dd599a7d03b9c8a97660cab570fe886ff116e722e322ab7`.

Working-tree hashes are diagnostic only. CRLF/LF differences cannot invalidate a run when canonical committed identity matches.

## Exact active source-tree contract

Historical v1.0 artifacts remain preserved. The v1.1 freeze must include exactly these active paths plus versioned governance/report artifacts:

```text
foundation/qf0/qf0a_impl/
  README.md
  LEAKAGE_AUDIT_CHECKLIST.md
  schemas/{QF0A_RUN_MANIFEST.schema.json,QF0A_MACHINE_VERDICT.schema.json,QF0A_RAW_OUTPUT.schema.json,QF0A_NEGATIVE_CONTROL_RESULT.schema.json,QF0A_EXECUTION_AUTHORIZATION.schema.json}
  environment/{QF0A_ENVIRONMENT_FREEZE_v1.1.json,QF0A_ENVIRONMENT_FREEZE_v1.1.md,requirements-scientific.txt,requirements-scientific.lock}
  manifests/QF0A_SOURCE_MANIFEST_v1.1.json
  audit/QF0A_STATIC_LEAKAGE_AUDIT_v1.1.json
  src/qf0a/{__init__.py,protocol_binding.py,analytic_reference.py,conditional_state.py,controls.py,metrics.py,adjudication.py,evidence.py,runner.py,audit.py,verdict.py}
  tests/fixtures/synthetic_fixtures.py
  tests/{test_source_freeze.py,test_adjudication_contract.py,test_evidence_contract.py,validate_preregistration.py}
  evidence/<run_id>/
```

No notebook, plugin registry, arbitrary control/metric list, plotting/GPU/symbolic framework, or alternate execution path is authorized.

## Environment and numeric contract

CPython 3.12.10, NumPy 2.1.3, SciPy 1.14.1, the existing hashed Windows AMD64 wheels, OpenBLAS identities, five thread variables fixed to `1`, `float64/complex128`, and validation-only `jsonschema[format]==4.23.0` remain unchanged. Environment v1.1 adds Git/canonical-hash metadata only.

## Automatic adjudication contract

The adjudicator consumes a complete typed record and derives the pipeline verdict. Public scientific code cannot accept a caller-supplied verdict. Precedence is `INVALIDATED > BLOCKED > FAIL > PASS`. Foundation verdict is restricted to `NOT_APPLICABLE` or `NOT_ESTABLISHED`.

Every one of the 22 thresholds and every required diagnostic is evaluated. Six controls are exact and non-configurable. Missing output/required evidence is BLOCKED; proven identity/provenance/post-hoc contamination is INVALIDATED; completed threshold or expected-control failure is FAIL; only a complete valid qualification is PASS.

## Result-blind ordering

The frozen runner sequence is A authorization/provenance; B direct conditional construction plus write-once persistence and hash; C analytic reference; D metrics and required diagnostics; E exact controls; F1 provisional named-threshold evaluation without a final PASS; G1 evidence materialization; G2 schema, inventory, hash, immutability, and completeness validation; F2 final machine-verdict derivation; and G3 final verdict/checksum sealing. Analytic output cannot enter the conditional path. A final PASS cannot exist before G2 establishes evidence completeness. There is no result-driven branch.

## Authorization contract

Future execution requires a separate schema-valid `execution_authorization.json` binding run ID, protocol/preregistration/environment/source identities, source baseline, runner identity, issuer/governance record, timestamp, and `authorized=true`. It contains no result and requires no source change. No `--force` or bypass exists.

## Evidence contract

The predetermined inventory includes authorization, run/environment/source manifests, conditional raw arrays, analytic arrays, metric arrays/index, diagnostic records, exactly six controls, leakage audit, machine verdict, deterministic machine-generated `human_report.md`, logs, and `SHA256SUMS.txt`. `source_manifest.sha256` is the SHA-256 of the exact bytes of the evidence copy `source_manifest.json`. Stage-B conditional arrays are written once and their hashes are rechecked before final adjudication. Completeness and canonical paths are machine checked; no post-inspection selection or overwrite is allowed.

## Gate

No frozen scientific result may be generated in this repair. The next gate is independent read-only review of v1.1.
