# QUAL-002 authorization and execution gate specification v1.0

> **STATUS: CANDIDATE GOVERNANCE SPECIFICATION — NOT AUTHORIZATION**
> QUAL-002 remains UNAUTHORIZED; authorization absent, run NOT STARTED, evidence root absent. Review or merge of this document SHALL NOT authorize execution. Only a subsequent explicit Terry decision may do so.

MUST, MUST NOT, SHALL, SHALL NOT and MAY are normative. This specification is based on authoritative main 532f389dd54c3062f7d1be4a8cf6c3af79b926fb. Terry reports Claude's review APPROVE WITH NON-BLOCKING FOLLOW-UPS: no technical repair/re-freeze prerequisite remains before Terry Governance Decision. F-17 and F-23 remain mandatory execution conditions, not repair requests; F-22/F-24/F-25/F-26 remain non-blocking.

## State machine

| State | Entry and permitted transition |
|---|---|
| 0 FROZEN / UNAUTHORIZED | Authorization/evidence absent, run not started. Only Terry may transition out. |
| 1 TERRY AUTHORIZED / ARTIFACT PREPARATION | Requires explicit `AUTHORIZE QF0A-v1.1-QUAL-002`. MAY prepare external authorization and governance record; science remains prohibited. |
| 2 AUTHORIZATION MATERIALIZED / EXECUTION HOLD | External JSON, its SHA-256 and governance record exist; execution remains prohibited. |
| 3 PREFLIGHT PASS / READY FOR ONE-SHOT EXECUTION | P0–P9 all PASS in one continuous recorded session. Only this state permits formal launch. |
| 4 EXECUTION CONSUMED | Successful canonical evidence-root creation consumes run ID; every later failure prohibits retry and requires adjudication. |
| 5 MACHINE RESULT / NOT YET ACCEPTED | Exit 0 or machine PASS alone is insufficient. |
| 6 INDEPENDENT OFFLINE ACCEPTANCE REVIEW | Independent package validation and subsequent governance acceptance are mandatory. |

## Future authorization contract

No authorization instance is supplied here. The future external artifact MUST satisfy committed QF0A_EXECUTION_AUTHORIZATION.schema.json (blob c324b2e2831104a7805f8a3c55d65d1e55cae375), with exactly these required fields and no additional properties:

| Field | Required future value |
|---|---|
| authorization_version | 1.1 |
| run_id | QF0A-v1.1-QUAL-002 |
| authorized | true, only after Terry AUTHORIZE |
| protocol_blob | 3af303a471d99397429d33f77ca512578f40f972 |
| protocol_sha256 | 1cf9f7f3a8cfa6a5b0e12a3290d7e545f44586f9359aaa5367754415cdcc5f3a |
| preregistration_blob | ac74a7753b35d0f7a0c9ce2e4ebb16ccc05081d1 |
| preregistration_sha256 | 0fea7b3c88645b0af687feb79da3871b838b14de507ffafefc3784d1b28ef932 |
| environment_sha256 | f3f934b66f87627e436c825158d3cad05703f55a31b876634be19915f6bd16f2 |
| source_manifest_sha256 | fb072ab0abeb443baf52f15807a827e94b43524081b89ffb3873da7df9995241 |
| source_baseline_commit | cb1c44c0b888595ef07bf3bc68db6eb3fe43aa48 (M4) |
| runner_blob | d4113bfcacc2c7abc91c97f5b8e41ed0359f5f1a |
| issuer | Nonempty actual issuer, populated only at authorization time |
| governance_record | Nonempty actual record reference, populated only at authorization time |
| timestamp_utc | Actual UTC RFC3339 authorization timestamp |

The artifact MUST NOT add execution_head, nonce, transcript or package fields. F-09 remains LOW/ACCEPTABLE; the external transcript and governance record SHALL bind actual execution HEAD operationally.
The file MUST remain outside the worktree, at D:\AI_Project\TechOrb\QF0A\authorizations\QF0A-v1.1-QUAL-002.authorization.json or an explicitly recorded absolute external path. It MUST NOT be committed. Its byte SHA-256 MUST be recorded before execution.
After AUTHORIZE, a governance-only DECISION_LOG record SHALL contain Terry's decision, run ID, external artifact path/SHA, baseline, manifest SHA, runner blob, protocol/prereg blobs and SHAs, environment SHA, timestamp and execution HOLD pending preflight. It SHALL NOT claim run started. To avoid self-reference, the record need not contain its own future commit SHA; the external transcript MUST bind resulting execution HEAD immediately before launch. Artifact and record references MUST be resolved and mutually consistent before execution.

## Transcript and execution HEAD

A transcript MUST be created outside the worktree before preflight, under D:\AI_Project\TechOrb\QF0A\transcripts\QF0A-v1.1-QUAL-002\ or a recorded external root. It MUST capture timestamps, exact interpreter/repo/auth paths, auth SHA, run ID, exact 40-character HEAD, M4, every command/stdout/stderr/exit code, import/thread environment, verbatim formal command/exit code, and post-run inventory. It is HOST-LOCAL evidence, not CI.
Execution HEAD MUST equal freshly fetched authoritative main. Later governance commits than the reviewed main MAY qualify only if accepted, allowlisted and technically byte-identical. HEAD and environment MUST remain unchanged through launch; any change restarts all gates.

## Mandatory sequential gates

Every gate MUST PASS in one recorded session. Historical PASS SHALL NOT substitute. Order MUST be P0 repository audit, P1 bootstrap filesystem/import-hook audit, P2 controlled launch preparation, P3 interpreter self-observation, P4 package audit, then P5–P9 below. No normal site-enabled Python startup may occur before P1 PASS; before any repository Python import, P0–P5 MUST complete.

### P0 repository and ignored-source audit

Record `git fetch`, `git rev-parse HEAD`, `git rev-parse origin/main`, `git status --porcelain`. Worktree MUST be clean. `git status --porcelain --ignored -- foundation/qf0/qf0a_impl/src` MUST produce EMPTY OUTPUT. Any ignored/untracked source, __pycache__, .pyc/.pyo or stray module FAILS preflight. The controller MUST NOT delete automatically; separate sanitation and complete preflight restart are required. No root means no consumption.

### P1 bootstrap filesystem / import-hook audit — F-17

Primary inspection MUST use OS/filesystem tools independent of Python site initialization (PowerShell on this Windows host). A normal site-enabled Python process MUST NOT be used to establish that the bootstrap environment was clean before startup. In particular, `python -P -s -c "import site; ..."` MUST NOT be initial cleanliness proof: -s does not disable system/venv site processing.

Before normal Python startup, inspect and record exact resolved locations:

- Frozen venv D:\AI_Project\TechOrb\QF0A\env\qf0a-v1.1-py31210 and its Scripts\python.exe; full pyvenv.cfg contents, home, include-system-site-packages and version if present.
- Venv Lib\site-packages, and base interpreter executable/runtime directories resolved from home and the frozen environment relationship. Derivation MUST be recorded rather than guessed. Unexpected system-site enablement FAILS; if explicitly adjudicated enabled, every relevant base/system site-packages directory MUST also be audited.
- User-site candidate %APPDATA%\Python\Python312\site-packages and any configuration-derived alternative; record ABSENT when missing, despite formal -s defense.
- Interpreter-adjacent python._pth, python312._pth and all *._pth in venv interpreter/Scripts and base executable/runtime directories. Record every path and full contents. Unexpected ._pth FAILS unless explicitly adjudicated as part of the frozen environment.

Inspect all applicable startup/import locations for *.pth, *._pth, sitecustomize.py/.pyc/.pyo, usercustomize.py/.pyc/.pyo, cached startup-customization bytecode, stray/unexpected qf0a*, duplicate/unexpected jsonschema*, and other path/import hooks. For each .pth record exact path, size, SHA-256, full text where decodable, and whether a line begins with executable import content. Preserve undecodable raw bytes and STOP for adjudication. Executable .pth lines MUST NOT be silently accepted; explicit adjudication is mandatory. Unknown/unexplained hooks FAIL PREFLIGHT / STOP. pip metadata does not establish hook-file integrity.

If filesystem/config inspection cannot resolve a required path, a bootstrap-only stdlib -c discovery process MAY use the exact frozen interpreter with `-S -P -s -E`. It MUST NOT import site, repository code, jsonschema, NumPy or SciPy, execute qualification logic, or create evidence. -S prevents automatic site initialization; -E ignores ambient PYTHON* variables. Pure filesystem discovery MUST be used when sufficient. This exception is path/config discovery only; it does not waive filesystem audit.

### P2 controlled launch environment — after P1 PASS

Interpreter MUST be D:\AI_Project\TechOrb\QF0A\env\qf0a-v1.1-py31210\Scripts\python.exe with `-P -s`. PYTHONPATH MUST contain exactly `<repo>\foundation\qf0\qf0a_impl\src`. PYTHONPYCACHEPREFIX MUST identify a newly created empty external dedicated directory. Every other import-affecting PYTHON* variable MUST be unset unless explicitly adjudicated. Record all such variables, including PYTHONHOME, PYTHONSTARTUP, PYTHONINSPECT, PYTHONUSERBASE, PYTHONNOUSERSITE, PYTHONSAFEPATH and the two required variables. -P prevents unsafe automatic path prepending; -s disables user site/usercustomize.

### P3 interpreter self-observation — only after bootstrap clearance

With intended -P -s mode record sys.executable, sys._base_executable if present, sys.version, sys.implementation, sys.orig_argv, sys.flags.safe_path/no_user_site, sys.pycache_prefix and sys.path. Both flags MUST be true; prefix and source path MUST match intended paths. Unexpected preceding/duplicate paths FAIL. Record site.getsitepackages(), site.getusersitepackages() and actual relevant sys.path; every discovered startup location MUST already have P1 clearance. A new/unexplained location FAILS and requires complete preflight restart with expanded filesystem audit before normal startup.
After P3, re-audit P1 startup-hook paths using filesystem tools against recorded inventory/hashes: no unexpected changed/new .pth, ._pth, sitecustomize, usercustomize, stray source/import-hook artifact may appear. Any change FAILS and requires complete preflight restart. Expected observation-cache output is confined to its recorded external directory and MUST NOT alter startup-hook state.
Preflight observation cache SHALL NOT be reused for formal launch. Record separate external preflight cache and new formal-execution cache paths. Immediately before formal launch, the latter MUST still be empty; otherwise FAIL / STOP.

### P4 F-23 package audit

Record exact frozen-interpreter `-P -s -m pip freeze --all` and `-P -s -m pip check` output. pip check MUST succeed. jsonschema MUST be 4.23.0; rfc3339-validator and required format dependencies MUST exist. Record metadata/version/location/requirements for jsonschema, jsonschema-specifications, referencing, rpds-py, rfc3339-validator and every dependency required by installed [format] stack. No qf0a distribution may shadow source. Unexplained versions, duplicates, metadata or dependency state FAIL; expected packages plus behavioral validation require no repair/re-freeze.
This gate MUST follow P1 filesystem clearance; pip freeze/check MUST NOT replace it. Record actual imported module origins (__file__ and spec origin) for jsonschema, referencing, rpds and rfc3339_validator (or the actual applicable module name), and compare resolved paths with audited approved package locations. Metadata alone is insufficient; unexpected origins FAIL PREFLIGHT / STOP.
After bootstrap clearance, external preflight MUST also resolve qf0a non-executingly where feasible and require its package origin/search locations exclusively under `<repo>\foundation\qf0\qf0a_impl\src\qf0a`. It MUST NOT import scientific modules for this check; actual formal provenance remains authoritative.

### P5 threads

Before environment imports set and observe OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, VECLIB_MAXIMUM_THREADS and NUMEXPR_NUM_THREADS all exactly 1. Any mismatch FAILS.

### P6 strict committed provenance

M4 MUST exist and be ancestor of HEAD. Manifest blob MUST be f74f1bf4e9fd6e81b2c11c1bf2008198953d1673, with exact SHA and baseline above, 41 unique entries, git-committed-bytes-v1 and scientific_execution NOT_PERFORMED. Independently verify each current committed byte length/blob/SHA, exact active inventory, runner/protocol/prereg/environment binding, and entirely allowlisted M4-to-HEAD diff. The current provenance implementation MUST return success; the lifecycle unit test alone is insufficient. Any discrepancy FAILS.

### P7 non-scientific host satisfiability

Only after P0–P6 PASS, a separate observation process MAY import minimum existing runtime code and call environment observation only. It MUST NOT create root or execute Stage B+. Require FULL POSITIVE HOST SATISFIABILITY PASS: CPython/version/architecture, base SHA, OS, registry CPU/source/normalization, NumPy/SciPy/BLAS, threads, dtypes, jsonschema 4.23.0, valid date-time accepted and invalid rejected. Failure STOP; run not consumed. Formal launch MUST use another fresh process without preloaded scientific modules.

### P8 authorization validation

Validate actual external bytes against committed authorization schema using Draft 2020-12 and functional FormatChecker, then independently check every frozen identity. Transcript SHA MUST equal governance-record SHA and actual bytes. Any mismatch FAILS.

### P9 absent canonical root

Immediately before launch record absence of `<repo>\foundation\qf0\qf0a_impl\evidence\QF0A-v1.1-QUAL-002`. Any existing object causes STOP; MUST NOT overwrite, delete or retry the ID. Existence of a prior root MUST be adjudicated as possible prior consumption.

## Formal launch and consumption

Only State 3 permits the following conceptual command; it MUST NOT be executed while this specification is being drafted/reviewed:

```text
<frozen-python> -P -s -m qf0a.entrypoint --repo-root <repo> --authorization <external-authorization-json> --evidence-root <canonical-QUAL002-evidence-root>
```

These three required flags match committed entrypoint d666c9e42f799548dcff41844ac856262a12a259. Future controller MUST recheck committed CLI, preserve actual one-line command verbatim, maintain controlled environment and fresh external cache, and re-record HEAD/auth SHA/root absence immediately before launch. No bypass or alternate launcher is permitted.
Formal launch remains `-P -s`; -S MUST NOT be added to formal execution under this specification. The `-S -P -s -E` mode above is exclusively optional bootstrap path discovery, not a frozen runtime change. F-17/F-23 remain external execution gates, not implementation-repair requirements.
Before successful root creation, failure MUST be logged externally and does not consume the run; no root may remain. Once root is created, any exception, crash, numerical/evidence/G3 failure consumes QUAL-002. NO RETRY UNDER QUAL-002; no deletion/recreation.

## Result and acceptance

Machine PASS is output only. Acceptance requires process exit 0, G3 final seal, exact final inventory, valid schemas/checksums, independent offline review and governance acceptance. Reviewer MUST verify immutable inventory/no extra files, machine-verdict/negative-control/run-manifest schemas, raw-output index, checksums, evidence manifests, authorization bytes/SHA, source binding/frozen identities, external transcript, exit and G3 result. Current contract is exactly 22 final regular files and 21 non-checksum paths; MUST NOT add a new evidence file to satisfy this specification. Transcript remains external. Any failed condition blocks accepted qualification even with machine PASS.

## HOLD and failure classification

`HOLD QF0A-v1.1-QUAL-002` SHALL leave accepted freeze intact; no authorization, authorization decision record, execution preflight or evidence root SHALL be created. No rollback is needed.
PRE-AUTHORIZATION FAILURE has no run implications. PRE-ROOT PREFLIGHT FAILURE stops/reports without consuming run. POST-ROOT EXECUTION FAILURE consumes run with no retry. MACHINE PASS / ACCEPTANCE BLOCKED is consumed and unaccepted. ACCEPTED QUALIFICATION requires independent offline review and governance decision.
The controller MUST NOT silently repair Python paths/packages, ignored source/cache, registry values, versions, root collisions, authorization or manifest mismatch. Remediation MUST be separately adjudicated, then gates restarted.

## Epistemic and historical limits

Classification remains Software Qualification + Known-result Reproduction + Constructibility Demonstration. Even accepted PASS establishes no Foundation validity, ontology support, empirical emergent time, absence of meta-time, Page–Wootters physical truth, GR/Lorentz recovery, QF-0D support or empirical physics evidence. QUAL-001 remains immutable historical numerical PASS / evidence-contract BLOCKED, Foundation NOT_APPLICABLE, no retry or reinterpretation.

## Specification self-check

Read against current authorization schema, entrypoint, provenance d01dc3dbac9f84056b4baf01d7aba043b9337d0d, prereg, environment, manifest and governance at reviewed main. No technical changes required. Schema lacks direct HEAD binding (F-09 operational transcript compensation); historical prereg/governance text is preserved and not rewritten by this candidate. Claude verdict above is Terry-supplied review context, not a new acceptance decision created here.
Next gate: Independent Review of QF0A-v1.1-QUAL-002 Formal Authorization + Execution Gate Specification.
