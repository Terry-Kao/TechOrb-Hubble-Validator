# R2 governance policy v1.0

CANDIDATE / NOT ACCEPTED / NOT AUTHORIZED FOR FORMAL EXECUTION

Proposed publication scope: policy only. Historical QUAL-002 remains State 2.
There is no run allocation, authorization, or execution readiness in this index.

## Policy acceptance

Terry's five approved governance directions are L1. This Draft PR is an L2 candidate. L3 policy-version acceptance requires the state transitions below. L4 qualification and execution require separate approvals and operational gates. L1–L3 confer no execution authority.

- **P0 DRAFT_CANDIDATE.** Candidate bytes are under review.
- **P1 REVIEWED_CANDIDATE.** Independent review binds the exact PR head, diff SHA, policy version and semantics.
- **P2 MERGE_AUTHORIZED.** Separate Terry approval MUST bind that reviewed head, version and acceptance procedure.
- **P3 MERGED_NOT_YET_ACCEPTED.** Verify the actual GitHub merge and merged policy bytes against the approved head. Candidate status persists.
- **P4 ACCEPTED_G_POLICY.** Separately authorized acceptance-sync MUST append a versioned record binding the verified merge commit and exact policy bytes. Independent verification MUST confirm the acceptance commit, Terry approval, monotonic checkpoint and effective time. The record MUST NOT include its own commit SHA. A detached receipt can bind that SHA after the commit exists.

P4 identifies the accepted policy version via the append-only record. The historical candidate bytes remain intact. Missing, mismatched, stale, replayed or rolled-back evidence MUST fail closed. G-POLICY provides no G-AUTH, qualification authorization or execution permission.

- [AUTHORIZATION_BINDING.md](AUTHORIZATION_BINDING.md)
- [HISTORICAL_QUAL002.md](HISTORICAL_QUAL002.md)
- [MODEL_B.md](MODEL_B.md)
- [QUALIFICATION_ROUTE.md](QUALIFICATION_ROUTE.md)
- [TECHNICAL_BASELINE.md](TECHNICAL_BASELINE.md)
- [TRUSTED_G_SOURCE.md](TRUSTED_G_SOURCE.md)
