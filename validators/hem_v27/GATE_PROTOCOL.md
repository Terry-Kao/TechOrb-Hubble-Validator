# HEM v27 Canonical Gate Protocol

## Global Rules

1. Gates execute in order R0–R4.
2. Every gate separates pipeline verdict from model verdict.
3. No target `H0`, including `H0 > 71`, defines PASS.
4. Exact versions, inputs, outputs, tolerances, and evidence paths are mandatory.
5. Failure excludes only the declared claim, implementation, or subclass.

## R0 — Reproducibility Freeze

**Question:** Can an independent clean environment reproduce the exact baseline and modified v26.4 results?

Required package:

- source commit/tree and all patch locations;
- CLASS, HyRec, compiler, language, and dependency versions;
- platform/container description;
- baseline and modified parameter files;
- neutrino-sector assumptions, including mass convention;
- commands, seeds, data/input hashes;
- raw logs, derived tables, spectra, figures, and file hashes;
- numeric tolerances and an independent rerun record.

**Pipeline PASS:** all required artifacts are frozen and an independent rerun agrees within declared tolerances.  
**FAIL/BLOCKED:** missing identity, irreproducible output, or unexplained mismatch.  
**Model verdict:** NOT APPLICABLE at this gate.

## R1 — Baseline Definition Audit

**Question:** Is the reference cosmology correctly and unambiguously defined?

Audit physical vs fractional densities, units, curvature, radiation/species content, neutrino masses and `N_eff`, primordial spectrum, helium treatment, recombination backend, gauge, precision settings, `z_*`, `z_d`, `r_s(z_*)`, `r_d`, and derived parameters.

**Pipeline PASS:** the baseline reproduces declared CLASS/HyRec reference outputs within predeclared tolerances and all conventions are explicit.  
**Model verdict:** NOT APPLICABLE or UNDETERMINED.

## R2 — Published Model Reproduction

**Question:** Can the pipeline reproduce one or more relevant published recombination/clumping/PMF benchmark results before testing a novel model?

Pre-register paper/version, equations, parameter points, digitized/reference outputs, observables, and tolerances. Cite differences rather than tuning them away.

**Pipeline PASS:** selected benchmarks reproduce within justified tolerances.  
**Model verdict:** applies only to the reproduced benchmark, not Tech-Orb or a new HEM model.

## R3 — Full Multi-Probe Likelihood

**Question:** What do data say when the model is evaluated with a valid, reproducible likelihood?

Minimum design must declare CMB TT/TE/EE and lensing treatment, BAO using the correct drag ruler, supernova likelihood, calibration/nuisance parameters, covariance, priors, sampler settings, convergence, and posterior/predictive checks. Dataset licensing and likelihood versions must be recorded.

**Pipeline PASS:** likelihood implementation, sampling, covariance/nuisance treatment, and convergence are correct and reproducible. This remains PASS if the posterior returns `H0 ≈ 69`, does not relieve the tension, or disfavors the model.  
**Model verdict:** SURVIVES, DISFAVORED, EXCLUDED, or UNDETERMINED using predeclared comparison criteria.

Theory-to-theory TT RMS, an acoustic-angle lock, or optimizer convergence alone cannot pass R3.

## R4 — Physicality Benchmark

**Question:** Does the effective deformation follow from a viable physical model?

For magnetogenesis/PMF candidates, require an explicit action or sourced dynamics, stability/ghost/gradient analysis, generation history, field spectrum and coherence scale, backreaction, MHD evolution, mapping to baryon-density statistics, recombination coupling, and independent constraints.

BBN is used only as early-universe consistency for the magnetogenesis/PMF scenario (for example expansion/energy-density or abundance effects). It is not a direct retrospective test that closes an effective recombination result.

**Pipeline PASS:** the physicality tests are implemented and reproducible.  
**Model verdict:** SURVIVES, DISFAVORED, EXCLUDED, or UNDETERMINED.

Any magnetogenesis action remains a Candidate Bridge / ANSATZ until derived and promoted under `control/SCIENTIFIC_GOVERNANCE.md`.

## Gate Report Template

- Gate, date, operator/reviewer
- source and environment identities
- inputs, commands, outputs, hashes
- tolerances and deviations
- pipeline verdict and rationale
- model verdict and bounded scope
- unresolved anomalies
- ledger/evidence/decision updates

