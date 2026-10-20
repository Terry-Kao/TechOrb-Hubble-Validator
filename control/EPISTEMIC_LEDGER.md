# Epistemic Ledger

## Schema

| Field | Required values |
|---|---|
| ID | Stable identifier |
| Statement | One bounded claim |
| Class | FACT / DERIVED / HYPOTHESIS / ANSATZ / EXCLUDED / OPEN QUESTION |
| Exclusion subtype | CLAIM / IMPLEMENTATION / SUBCLASS / N/A |
| Evidence status | VERIFIED / SUPPORTED / PROVISIONAL / REPRO-BLOCKED / SUPERSEDED / HISTORICAL |
| Scope | Assumptions, domain, version |
| Evidence | Repository path or external citation |
| Promotion / falsification condition | Concrete next decision rule |

## Current Ledger

| ID | Statement | Class | Exclusion subtype | Evidence status | Scope | Evidence | Promotion / falsification condition |
|---|---|---|---|---|---|---|---|
| F-001 | Standard relativistic clock behavior and causal/null structure must be recovered by any emergent-time implementation. | FACT | N/A | VERIFIED | Known-physics constraint | `foundation/MATHEMATICAL_REQUIREMENTS.md` | Reject an implementation that contradicts verified SR/GR clock behavior. |
| F-002 | The BAO drag ruler `r_d` and last-scattering sound horizon `r_s(z_*)` are distinct quantities. | FACT | N/A | VERIFIED | Cosmological definitions | `foundation/MATHEMATICAL_REQUIREMENTS.md` | Any pipeline conflating them fails audit. |
| D-001 | At fixed physical matter density, increasing `h` lowers `Omega_m` because `omega_m = Omega_m h^2`. | DERIVED | N/A | SUPPORTED | Fixed-`omega_m` scans | Algebra; `validators/hem_v27/GATE_PROTOCOL.md` | Verify conventions in R1. |
| D-002 | HEM v26.4 reports a CLASS-level sensitivity/degeneracy direction, not a cosmological posterior. | DERIVED | N/A | REPRO-BLOCKED | Stated v26.4 setup | `control/PROJECT_STATE.md`; `validators/hem_v27/r0/recovery-2026-10-20/README.md` | Reproduce exact run and artifacts under R0. |
| H-001 | Physical reality admits a non-trivial Physics State Manifold description. | HYPOTHESIS | N/A | PROVISIONAL | F0 | `foundation/FOUNDATION_V1_SPEC.md` | Define states/equivalence and find coordinate-independent structure. |
| H-002 | Observed fields and geometry may be effective representations of a deeper state. | HYPOTHESIS | N/A | PROVISIONAL | F0 | `foundation/FOUNDATION_V1_SPEC.md` | Produce non-gauge observable consequences and transition laws. |
| H-003 | Time may emerge from relational state evolution. | HYPOTHESIS | N/A | PROVISIONAL | F0 | `foundation/FOUNDATION_V1_SPEC.md` | Recover SR/GR timing and causal structure. |
| H-004 | A deeper invariant, constraint, or symmetry organizes physical representations. | HYPOTHESIS | N/A | PROVISIONAL | F0 | `foundation/FOUNDATION_V1_SPEC.md` | Specify transformations and a non-trivial invariant. |
| A-001 | Taiji dual sectors/branch exchange describe global physical topology. | ANSATZ | N/A | PROVISIONAL | F1 | `topology/TAIJI_DUAL_TOPOLOGY.md` | Define topology/map, consistency, and discriminating observable. |
| A-002 | A specific magnetogenesis action bridges Foundation to HEM. | ANSATZ | N/A | PROVISIONAL | Candidate only | `foundation/BRIDGE_REGISTER.md` | Derive, benchmark, and pass physicality requirements before promotion. |
| A-003 | Single-zone clumping or phenomenological recombination deformation represents PMF physics. | ANSATZ | N/A | PROVISIONAL | HEM implementation candidates | `foundation/BRIDGE_REGISTER.md` | Establish PMF spectrum → MHD → density → recombination mapping. |
| X-001 | “HEM v26.4 solves the Hubble tension” is not supported by the available experiment. | EXCLUDED | CLAIM | SUPPORTED | v26.4 claims | `control/PROJECT_STATE.md`; `validators/hem_v27/README.md` | Requires reproducible multi-probe likelihood and physical model; do not revive from a geometric lock. |
| X-002 | The tested v1–v10 smooth late-time kinematic patch family is viable as a complete solution. | EXCLUDED | SUBCLASS | HISTORICAL | Recorded implementation/domain only | `validators/late_time/EXCLUSION_MAP_V1_V25.md` | Reopen only with new implementation and full constraint audit. |
| X-003 | The tested v11–v20 screened-gravity/SNIa implementation supplies the required luminosity shift. | EXCLUDED | IMPLEMENTATION | HISTORICAL | Recorded screening assumptions | `validators/late_time/EXCLUSION_MAP_V1_V25.md` | New subclass must satisfy stellar and Solar-System constraints. |
| X-004 | The tested v21–v25 smooth disformal photon-propagation implementation supplies the shift while respecting GW170817. | EXCLUDED | IMPLEMENTATION | HISTORICAL | Recorded coupling and ray trace | `validators/late_time/EXCLUSION_MAP_V1_V25.md` | New subclass must independently satisfy multi-messenger propagation bounds. |
| OQ-001 | What mathematical object, dimension, equivalence relation, and topology define the state manifold? | OPEN QUESTION | N/A | PROVISIONAL | F0 | `foundation/MATHEMATICAL_REQUIREMENTS.md` | Resolve in a versioned specification. |
| OQ-002 | Which Foundation structure yields a unique observable rather than a reparameterization? | OPEN QUESTION | N/A | PROVISIONAL | Foundation-to-validator bridge | `foundation/BRIDGE_REGISTER.md` | Register and test a discriminating bridge. |

