# QUAL-002 P8 hash-binding clarification v1.0

Status: **NORMATIVE GOVERNANCE CLARIFICATION**

## Applies to and scope

This later Terry governance adjudication applies only to P8 SHA-binding semantics in `control/QF0A_QUAL002_AUTHORIZATION_EXECUTION_GATE_SPEC_v1.0.md`, accepted blob `4c629b70d6933733b7aa9298bd735accd8559bb9`. The accepted specification bytes, including the earlier ambiguous wording, MUST remain unchanged.

## Normative definition

Define `AUTH_SHA = SHA256(actual external authorization JSON bytes)`.

P8 SHALL require:

`Transcript-recorded AUTH_SHA == Governance-recorded AUTH_SHA == recomputed AUTH_SHA from actual external authorization JSON bytes`.

“Transcript SHA” means the external authorization artifact SHA-256 value recorded inside the execution transcript. “governance-record SHA” means that same authorization-artifact SHA-256 value recorded in the governance record / DECISION_LOG. “actual bytes” means the SHA-256 recomputed directly from the actual external authorization JSON bytes.

Comparison SHALL be exact equality of the 256-bit SHA-256 values represented by 64 hexadecimal digits; hexadecimal letter case MAY be normalized for comparison only. This does not alter any stricter schema/implementation representation requirement, permit truncated hashes, or permit JSON parsing/reserialization or line-ending normalization before hashing the actual artifact bytes.

## Explicit exclusion

P8 does **not** require:

`SHA256(transcript file bytes) == SHA256(DECISION_LOG.md bytes) == SHA256(authorization JSON bytes)`.

These are distinct objects; their object-byte hashes are not required to be identical.

## Non-modification

This clarification changes no protocol semantics, preregistration, source baseline, source manifest, runner, environment freeze, schemas or their existing definitions, evidence inventory, provenance requirements, numerical thresholds, controls, one-shot/no-retry semantics, acceptance criteria, or scientific interpretation. No frozen technical/scientific artifact may be changed under this clarification.

## Authorization continuity and event history

Terry's existing `AUTHORIZE QF0A-v1.1-QUAL-002` remains valid and has not been revoked. No second authorization is required solely because of this clarification.

The prior controller discovered the ambiguity before authorization artifact materialization and stopped fail-closed. This was a pre-execution governance/specification clarification event, not a QUAL-002 execution failure. That workflow created no authorization artifact, execution transcript or canonical evidence root, ran no P0–P9 or formal qualification command, performed no scientific execution, and did not consume the run ID.

This governance clarification task itself MUST NOT materialize the authorization artifact or its final SHA, create an execution transcript or canonical evidence root, complete formal A0, run P0–P9, or execute QUAL-002.

QUAL-001 remains immutable historical: EXECUTED / NUMERICAL GATES PASS / EVIDENCE CONTRACT BLOCKED; no retry or rewrite.

## Resume rule

After merge, future execution MUST fresh-fetch authoritative main and restart A0 from the beginning under the accepted specification plus this normative clarification. It MUST explicitly verify the governance-only main drift, verify all frozen technical identities byte-identically, and incorporate this clarification commit into the allowed governance lineage.

No partial A0 result from the stopped workflow may be reused as an A0 PASS. Only after fresh A0 PASS may a later separately bounded task materialize authorization and continue through governance SHA binding, recorded P0–P9, and (only after all gates PASS) one-shot execution. Independent offline review and subsequent governance acceptance remain mandatory.
