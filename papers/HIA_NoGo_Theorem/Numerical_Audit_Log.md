# Numerical Audit Log: Historical HIA v1–v25 Iterations

> **Governance notice (2026-10-20):** This is a historical narrative, not a universal proof. “Excluded” below means the recorded implementation or defined subclass under its stated assumptions. The current bounded interpretation is `validators/late_time/EXCLUSION_MAP_V1_V25.md`.

## Phase 1: Tested Kinematic Expansion Gains (v1–v10)

**Idea:** modify late-time `H(z)` with a localized gain near `z ≈ 0.5` while compensating the acoustic geometry.

**Recorded result:** for the tested parameterization and target, the optimizer drove `H_base` into roughly the 58–61 km/s/Mpc range, producing tension with physical constraints adopted by that analysis.

**Current verdict:** `EXCLUDED: SUBCLASS / HISTORICAL` for the tested patch family. This does not exclude all late-time expansion histories.

## Phase 2: Tested Conformal Coupling / Varying G (v11–v20)

**Idea:** change local `G_eff` to alter the Type-Ia-supernova absolute luminosity.

**Recorded result:** under the adopted Chameleon/Symmetron screening assumptions, the white-dwarf interior response remained far below the target while Solar-System bounds were imposed.

**Current verdict:** `EXCLUDED: IMPLEMENTATION / HISTORICAL`. Other theories require their own stellar and local-gravity audit.

## Phase 3: Tested Disformal Photon Propagation (v21–v25)

**Idea:** use a smooth disformal photon metric to obtain a target distance-modulus shift.

**Recorded result:** the stored implementation/parameter point produced an arrival-time difference of approximately `-3.1 × 10^8 s` for the adopted GW170817-distance test, incompatible with the adopted multi-messenger bound.

**Current verdict:** `EXCLUDED: IMPLEMENTATION / HISTORICAL` for the tested route.

## Audit State

The historical evidence is retained and the old universal No-Go conclusion is superseded. Future work must publish exact model scope, source/environment identity, input/output hashes, tolerances, and independent reproduction before promoting quantitative findings.

