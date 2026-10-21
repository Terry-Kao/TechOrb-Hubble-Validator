# R1 Frozen Tolerance Contract

This contract is binding from its pre-execution commit. It may not be relaxed after numerical output is observed.

## Identity and definition checks — exact

The following require exact equality: source commit, clean-tree status, input bytes, build command, executable hash between the two canonical runs, runtime identity, species counts, neutrino masses/temperatures, recombination backend, gauge, curvature declaration, `YHe` treatment, primordial inputs, output/precision controls and successful exit codes.

Missing or ambiguous identity is `BLOCKED`; a conflicting resolved setting is `FAIL`.

## Canonical-modern numerical reproduction

Run 2 is compared with Run 1:

- Raw text outputs and resolved parameter files: byte-identical SHA-256 is required when paths/timestamps are absent from content. If metadata-bearing lines prevent byte identity, preserve both files and compare parsed numerical columns instead.
- Audited scalar quantities: `abs(delta) <= 5e-10 + 5e-10 * abs(reference)`.
- Background and thermodynamics numeric tables: every finite cell must satisfy `abs(delta) <= 5e-12 + 5e-10 * abs(reference)`.
- CMB and matter-spectrum numeric tables: every finite cell must satisfy `abs(delta) <= 5e-12 + 5e-9 * abs(reference)`.
- Row/column counts, headers, sampled independent-variable grids and non-finite-value masks: exact equality.

## Historical-v26.4 control

The baseline printed quantities and primary artifacts must match the accepted R0-H evidence exactly at the stored precision/hashes. Any deviation is `FAIL` for the historical control and blocks overall R1 PASS until explained without changing this contract.

## Consistency checks

- `h = H0 / 100`: absolute residual `<= 5e-12`.
- `Omega_i = omega_i / h^2` for baryons and CDM: absolute residual `<= 5e-10`.
- Reported `Omega_m` versus the sum of recorded matter components: absolute residual `<= 5e-8`; the report must state which massive-neutrino contribution is included.
- `r_d` must be evaluated at `z_d` and must not be substituted for `r_s(z_*)`; both must be present and positive.

## Verdict rule

R1 is `PASS` only if every exact identity/definition check, historical control, canonical reproduction and consistency check passes and all required evidence is present. Any exceeded frozen tolerance is `FAIL`. Missing required evidence or unavailable execution capability is `BLOCKED`. The HyRec-versus-RECFAST diagnostic has no acceptance threshold.

R1 PASS is a pipeline verdict only; the model verdict remains `NOT APPLICABLE` or `UNDETERMINED`.
