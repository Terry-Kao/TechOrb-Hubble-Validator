# R1 Audit Checklist

## Before execution

- [ ] Confirm main contains R0 PASS and read `validators/hem_v27/GATE_PROTOCOL.md` plus all R1 files.
- [ ] Record the pre-registration commit and verify no R1 numeric output predates it.
- [ ] Create a new H:-resident R1 workspace and inventory it before execution.
- [ ] Verify pristine CLASS commit and clean tree; record source archive/repository hashes.
- [ ] Freeze both final INI files and the exact build/run commands.

## Environment and build evidence

- [ ] OS/kernel/architecture, container or WSL identity.
- [ ] compiler, libc/libm, make, Python and relevant numerical libraries.
- [ ] CLASS commit, HyRec files/identity and executable SHA-256.
- [ ] stdout/stderr and exit code for every build and run.

## Baseline audit

- [ ] Physical and fractional densities, units, H0/h and curvature.
- [ ] Radiation/neutrino sector: N_ur, N_ncdm, masses, temperatures and N_eff.
- [ ] YHe and BBN treatment.
- [ ] A_s, n_s and pivot/default provenance.
- [ ] recombination backend, gauge and resolved precision settings.
- [ ] z_*, z_d, r_s(z_*), r_d=r_s(z_d), 100 theta_*.
- [ ] CLASS and HyRec identity.
- [ ] Historical control against R0-H.
- [ ] Canonical Run 1 versus Run 2 under every frozen tolerance.
- [ ] HyRec-versus-RECFAST diagnostic, clearly marked non-adjudicative.

## Evidence and governance

- [ ] Preserve all INIs, resolved parameter files, raw outputs, logs and derived tables.
- [ ] Generate complete SHA-256 manifest and verify every entry.
- [ ] Produce machine-readable verdict and human comparison report.
- [ ] Update PROJECT_STATE, DECISION_LOG, EVIDENCE_INDEX and Issue #5.
- [ ] Do not start R2 numeric work. If R1 passes, create only the R2 issue/spec entry.
