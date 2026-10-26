# QF-0A Leakage Audit Checklist

Every item is mandatory for a future implementation review. `NO`, `UNKNOWN`, missing evidence, or an unreviewable source path prevents PASS.

| ID | Required assertion | Failure class |
|---|---|---|
| L-01 | No external classical/meta-time orbit is generated and then packaged as a global history state. | INVALIDATED |
| L-02 | The analytic target is computed in a separate module and enters only after conditional-state output is finalized and hashed. | INVALIDATED |
| L-03 | Analytic values, phases, states, fitted parameters, masks, and branches are not fed into the conditional-state path. | INVALIDATED |
| L-04 | Protocol thresholds and tolerances are immutable and no post-hoc mutation exists. | INVALIDATED |
| L-05 | No hidden exclusions, outcome dropping, array trimming, or silent non-finite filtering occurs. | INVALIDATED |
| L-06 | The clock-scramble inverse is unavailable to implementation and adjudication code; the fixed permutation is used only as registered. | INVALIDATED |
| L-07 | Section 9 uses fixed `W=(-0.5,0.5)` and the full spectral projector; output-dependent window movement is impossible. | INVALIDATED |
| L-08 | Eigensolver basis rotation inside a degenerate subspace cannot change state selection. | FAIL |
| L-09 | All Hamiltonians, constraint, POVM, history state, controls, parameters, and thresholds are verified against the frozen protocol identity. | INVALIDATED |
| L-10 | No HEM, Hubble, PFM, topology, or later-QF result influences implementation, thresholds, exclusions, or verdicts. | INVALIDATED |
| L-11 | Static scan findings, reviewed paths, reviewer identity, and source-manifest hash are preserved. | BLOCKED |
| L-12 | Runtime provenance proves the executed source, environment, command, cwd, and evidence directory identities. | BLOCKED |

Reviewers must record each item as `PASS`, `FAIL`, `BLOCKED`, or `INVALIDATED`, cite immutable evidence paths and hashes, and apply the preregistered verdict precedence.
