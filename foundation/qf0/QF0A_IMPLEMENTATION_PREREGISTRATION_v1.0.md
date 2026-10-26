# QF-0A Implementation Preregistration v1.0

**Status:** `PROPOSED / DRAFT / NOT EXECUTED`

**Scope:** implementation and evidence-contract freeze only; no scientific or numerical execution

**Authoritative main:** `4c2920871898add6254902fb3963784e2761d375`

## 1. Binding identity

This preregistration is byte-bound to the authoritative protocol and must be invalidated, not silently updated, if any identity differs:

| Field | Frozen value |
|---|---|
| Protocol version | `QF0A_FROZEN_PROTOCOL_v1.0` |
| Path | `foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.0.md` |
| Git blob SHA | `c11599a058caa8dfb0912c4e00a0656bef4688bd` |
| File SHA-256 | `fb7969c11e9e2d4b1eebdd2495692dc422eca418bbbaa85bbabcda2815811d8b` |
| Authoritative main SHA | `4c2920871898add6254902fb3963784e2761d375` |

No implementation phase may modify the protocol, thresholds, Hamiltonians, constraint, clock POVM, history state, negative controls, Section 9 window/projector rule, or clock-scramble permutation. A protocol change requires a separately reviewed protocol version and a new preregistration.

## 2. Runtime and dependency contract

- Language: CPython `3.12.x`; the exact patch version must be recorded in each future run manifest.
- Minimal scientific dependencies: NumPy `2.1.*` and SciPy `1.14.*` only. JSON Schema validation may use `jsonschema 4.23.*` in the validation environment.
- The future lock file must pin exact resolved versions and hashes before source freeze. No large framework, notebook runtime, GPU library, symbolic engine, or plotting stack is permitted without a new reviewed preregistration.
- Numeric types are `float64` and `complex128`. Silent downgrade, platform-default extended precision, mixed precision, or GPU/tensor precision substitution is forbidden.
- Every run manifest must capture NumPy/SciPy versions, BLAS/LAPACK vendor and version/build identity, compiler/runtime identity where exposed, and thread configuration.

## 3. Determinism and tolerance policy

- All random sources use explicitly named integer seeds recorded in the manifest; unseeded entropy is forbidden.
- Eigenvalue/eigenspace handling must use the protocol's full spectral projector, deterministic ordering, frozen phase convention, rank checks, boundary checks, idempotence checks, and crossing FAIL rules. A basis vector chosen from a degenerate eigenspace is forbidden.
- Record OS, kernel/build, CPU architecture and model, Python implementation/version, locale, timezone, package lock hash, Git commit, dirty-tree state, command line, current working directory, and BLAS/LAPACK identity.
- Record and fix `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `NUMEXPR_NUM_THREADS`; absent variables are recorded as `UNSET`, never inferred.
- Protocol tolerances are immutable. No adaptive, data-dependent, platform-relaxed, or post-hoc tolerance is allowed. A platform unable to satisfy the frozen numeric contract is `BLOCKED`, not retuned.

## 4. Source-tree contract and freeze

The only future implementation paths permitted without revising this preregistration are:

```text
foundation/qf0/qf0a_impl/
  README.md
  schemas/
  manifests/
  src/qf0a/{protocol_binding.py,analytic_reference.py,conditional_state.py,controls.py,audit.py,verdict.py}
  tests/{test_imports.py,test_schema_contracts.py,test_protocol_binding.py,test_static_leakage.py}
  evidence/<run_id>/
```

This checkpoint creates only documentation, schemas, and a structural validator. It does not create the scientific modules above. Before any future execution, the allowed source tree must be complete, reviewed, committed with a clean tree, and represented by a canonical UTF-8, slash-normalized source manifest containing path, byte length, Git blob SHA where applicable, and SHA-256. The manifest itself receives a SHA-256 and is immutable for that run.

Generated arrays, scientific result JSON, figures, adjudicative reports, executable result files, caches, and verdict instances are forbidden in this PR.

## 5. Analytic-reference separation

`analytic_reference.py` and `conditional_state.py` must be independent calculation paths. The analytic target may supply only comparison values to the adjudication layer after the conditional-state output is finalized and hashed. It may not supply state vectors, phases, fitted parameters, thresholds, branches, masks, or normalization into the conditional-state path. Shared code is limited to immutable schema/types and protocol-identity verification; no shared evolution or state-construction helper is allowed.

Unit tests must use structural fixtures or deliberately non-adjudicative toy values. They may not compute or snapshot the frozen protocol's conditional states, fidelity, trace distance, interaction suite, or clock-choice results.

## 6. No-external-time and leakage controls

It is forbidden to generate an external classical-time orbit and then package it as a global history state. The future implementation must construct and condition the registered global constrained state through the frozen protocol path. The audit plan comprises:

1. static source scan for prohibited external-time orbit generation, answer-bearing imports, mutable thresholds, hidden exclusions, and inverse-scramble access;
2. source-level traceability from every protocol parameter to the byte-bound protocol adapter;
3. runtime provenance showing that analytic targets enter only the comparison stage;
4. immutable raw-output index proving no dropped outcomes or hidden exclusions;
5. independent review of the completed source manifest, audit record, and negative controls before adjudication.

The binding checklist is `foundation/qf0/qf0a_impl/LEAKAGE_AUDIT_CHECKLIST.md`. HEM, Hubble, PFM, topology, and later QF-layer outcomes may not influence QF-0A implementation, thresholds, inclusion, or verdict.

## 7. Result blindness

During implementation and preregistration, adjudicative conditional-state outputs must not be generated or inspected. Allowed checks are JSON/schema parsing, Python syntax/import checks, static source checks, protocol/hash binding, link checks, and unit-structure tests with non-adjudicative fixtures. Any accidental computation, display, persistence, or inspection of a frozen-protocol scientific output invalidates the checkpoint and requires disclosure plus independent re-preregistration.

## 8. Environment and run manifest

The run manifest must validate against `QF0A_RUN_MANIFEST.schema.json` and record:

- protocol path/version/blob/SHA-256 and authoritative main SHA;
- implementation commit and branch, dirty-tree status, source-manifest path/hash;
- UTC and Asia/Taipei timestamps, exact command line, and cwd;
- OS/kernel/build, CPU architecture/model, Python identity, locale, timezone;
- exact package and lock identities, NumPy/SciPy, BLAS/LAPACK, numeric dtype;
- seeds and determinism/thread environment;
- evidence paths and hashes for raw output index, controls, machine verdict, human report, and hash manifest.

Dirty execution trees are `INVALIDATED`. Missing reproducibility identity or required evidence is `BLOCKED` unless the evidence proves contamination or contract violation, which is `INVALIDATED`.

## 9. Evidence directory contract

Future evidence is confined to `foundation/qf0/qf0a_impl/evidence/<run_id>/`:

```text
run_manifest.json
environment_manifest.json
source_manifest.json
source_manifest.sha256
raw/
raw_output_index.json
negative_controls.json
leakage_audit.json
machine_verdict.json
human_report.md
SHA256SUMS.txt
logs/
```

Every regular file except `SHA256SUMS.txt` must be listed in that hash manifest. Raw arrays are immutable little-endian NumPy `.npy` files with explicit `float64`/`complex128` dtype, shape, semantic name, and SHA-256 recorded in the raw-output index. Compressed archives, pickles, object arrays, lossy tables, and result-only screenshots are forbidden as primary evidence.

## 10. Verdict separation

`QF0A_MACHINE_VERDICT.schema.json` requires independent fields:

- `pipeline_reproduction_verdict`: `PASS`, `FAIL`, `BLOCKED`, or `INVALIDATED`;
- `foundation_model_verdict`: only `NOT_APPLICABLE` or `NOT_ESTABLISHED` at QF-0A.

A reproduction `PASS` is a known-result pipeline result only. It cannot establish the Foundation, Tech-Orb, emergent spacetime, QF-0B/C/D, QF-1/2/3, HEM, PFM, or topology.

## 11. Raw output and negative-control contracts

`QF0A_RAW_OUTPUT.schema.json` is an index contract: every array declares identifier, relative path, dtype, shape, byte size, SHA-256, protocol section, and whether it is adjudicative. The index records that no outcome was dropped and lists exclusions; hidden exclusions are forbidden.

`QF0A_NEGATIVE_CONTROL_RESULT.schema.json` requires one record for every frozen negative control, its protocol identity, expected failure mode, observed machine status, evidence hashes, and whether the expected failure occurred. A missing control is `BLOCKED`; a control that unexpectedly passes or cannot be interpreted under the frozen rule is `FAIL`; a mutated or post-hoc control is `INVALIDATED`.

## 12. Independent-review evidence requirements

Independent read-only review must receive the exact implementation commit, clean-tree proof, protocol/blob/file hashes, source manifest and hash, environment and lock identities, command line, raw index and arrays, negative controls, leakage audit, machine verdict, human report, logs, and complete hash manifest. Review must verify analytic/conditional separation, no external-time generation, no answer leakage, no post-hoc changes, and verdict separation. Missing access or an unverifiable hash is `BLOCKED`.

## 13. Explicit FAIL, BLOCKED, and INVALIDATED rules

- `FAIL`: a valid, complete run violates a frozen scientific acceptance condition or a required negative control does not fail as expected.
- `BLOCKED`: required evidence, environment identity, backend identity, deterministic setting, or review access is absent; the run cannot be adjudicated.
- `INVALIDATED`: protocol/source/tolerance/control mutation; dirty execution tree; external-time leakage; analytic-target injection; hidden exclusion/outcome dropping; inverse clock-scramble leakage; post-hoc threshold/window change; hash mismatch; or use of external HEM/Hubble/PFM/topology results.
- `PASS` requires every required scientific gate and negative control under a complete, valid evidence package. Missing evidence can never be treated as PASS.
- Precedence is `INVALIDATED` over `BLOCKED` over `FAIL` over `PASS` when multiple conditions occur.

## 14. Definition of Done for this checkpoint

This checkpoint is complete only when:

1. PR #9 is merged and authoritative main is recorded.
2. Protocol Git blob and file SHA-256 bindings are exact.
3. Runtime, dtype, dependency, backend, determinism, source, evidence, leakage, and verdict contracts are frozen.
4. Four machine-readable schemas parse and enforce required vocabularies and separation.
5. Governance states Foundation initialization as merged, protocol frozen/not executed, and preregistration draft/not executed.
6. Structural/static validation and `git diff --check` pass.
7. No scientific/numerical calculation or adjudicative output exists.
8. A Draft PR is opened and left unmerged for independent read-only review.

The exact next gate is independent read-only review of this preregistration package. Scientific implementation and execution remain unauthorized until a separate reviewed implementation authorization binds a clean source commit and environment lock to these contracts.
