# Frozen Tolerance Contract

Status: **INCOMPLETE / EXECUTION BLOCKED**. This file is committed before any R2 numerical output.

## Exact checks

- Source commit, paper/version identity and input-file SHA256: exact match.
- Algebraic `b`, volume conservation and mass conservation: absolute residual `<=1e-12`.
- Standard control repeatability: raw output hashes must be identical across two executions in one admitted environment.
- Exit status: zero; no unreviewed warnings or dirty source paths.

## Published-reference checks

Numerical tolerances for `x_e(z)` and `g(z)` are intentionally **UNSET** until REF-01/REF-02 digitization uncertainty is measured from official source bytes. The eventual tolerance must be at least the quadrature/envelope of axis-pixel calibration, curve thickness, point placement and solver-output interpolation, and must be committed in a new pre-execution amendment. It may not depend on reproduction residuals.

Because an adjudicative tolerance is not yet frozen, R2 numerical execution is forbidden and the pipeline verdict is BLOCKED, not FAIL.
