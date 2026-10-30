# QF-0A v1.1 Main-Lineage Runtime Re-Freeze

Candidate for independent re-freeze acceptance; not formally accepted.

PR #19 approved head a4a943f1a89470c9d8dc61f10440ac89260b72c3 merged as M1 bab7393b370c36b72b9f08f4e7eb4285bf49420c.
Transition test-only PR #20 head dbae584bf568208e91e6c94b2e29bc16e58a4463 merged as M2 ec6497164b89b1c050e0457ca8cb3770305990bf.
M1 to M2 changes only the lifecycle-aware readiness test. Runtime bytes are unchanged.

M2 is the exact authoritative-main source baseline. The old baseline 80403cb383807334066eec482012ed42d4ded0ca is superseded only for future execution identity; historical evidence is preserved.
F-02 is structurally closed by authoritative-main ancestry.

The manifest was regenerated from M2 Git object bytes using git-committed-bytes-v1, without self-reference.
Inventory: 40 to 41 entries. Only added path: foundation/qf0/qf0a_impl/tests/test_phase_a_runtime_closure.py, a Phase A test covered by the existing test inventory convention. No removals.
Manifest SHA-256: 8b05e8ef9a0adff3cec373bde6ac92625577898b5764c4ea86ef1d7623ea8667.

Phase B changes only this report and the source manifest. Runtime, scientific implementation, tests, preregistration, protocol, schemas, environment and dependency lock remain unchanged from M2.

Transition validation: 79/79 tests PASS; prereg PASS; five Draft 2020-12 schemas with FormatChecker PASS; eight positive fixtures PASS; 103 negative fixtures rejected; compile, repository JSON and diff-check PASS.

QUAL-001 remains IMMUTABLE HISTORICAL ATTEMPT / NUMERICAL GATES PASS / EVIDENCE CONTRACT BLOCKED. Raw machine PASS is historical output only; Foundation NOT_APPLICABLE. No retry or historical changes.
QUAL-002 authorization NOT CREATED / UNAUTHORIZED; run NOT STARTED; no evidence root.
Scientific execution NOT_PERFORMED. No official launcher, entrypoint or runner execute was invoked.

Next gate: Independent narrow read-only review of QF-0A v1.1 Phase B Main-Lineage Re-Freeze.
