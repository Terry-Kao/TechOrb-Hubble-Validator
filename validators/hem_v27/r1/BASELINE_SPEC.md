# R1 Frozen Baseline Specification

Freeze date: 2026-10-21. This file must be committed before any R1 numerical output is inspected.

## Common execution identity

- Code: pristine CLASS v3.2.0 commit `aa92943e4ab86b56970953589b4897adf2bd0f99`.
- Architecture: Linux x86-64 in a newly recorded reproducible runtime.
- Build: default CLASS release build; exact compiler, libc, make, Python and shared-library identities must be captured.
- Evidence root: a new directory under `H:\AI_Products\AI_Factory_Workspace\`; all controllable sources, configs, caches, logs, raw outputs, derived tables, reports and manifests remain there.
- Output: background, thermodynamics, scalar temperature/polarization/lensing spectra and resolved parameter files through `l_max_scalars = 2500`.
- Gauge: `synchronous`.
- Curvature: `Omega_k = 0`.
- No HEM/RECFAST source patch is permitted in the canonical-modern executable.

## Track A — historical-v26.4 control

This is non-adjudicative historical context and must use the byte-preserved `run_colab_v26_4.py` semantics:

- `h = 0.674`
- `omega_b = 0.022383`
- `omega_cdm = 0.11933`
- `n_s = 0.965`
- `A_s = 2.1e-9`
- implicit CLASS neutrino defaults from the historical runner, including `N_ncdm = 0`; record the resolved `N_ur`/`N_eff`
- `recombination = RECFAST`
- `YHe`: historical runner/default treatment; record the resolved value and whether it is inferred or explicit
- `HIA_B_CLUMPING = 1.0`, `HIA_LYA_EXP = 0.0`

Its required result is agreement with the already accepted R0-H baseline printout/artifacts. It cannot define the modern baseline.

## Track B — canonical-modern adjudicative baseline

The input is the CLASS v3.2.0 Planck-2018 example parameter point already frozen in the R0 package, with the following values made explicit:

- `H0 = 67.32117`; derived `h = 0.6732117`
- `omega_b = 0.02238280`
- `omega_cdm = 0.1201075`
- `Omega_k = 0`
- `N_ur = 2.046`
- `N_ncdm = 1`
- `m_ncdm = 0.06` eV (one species; sum `m_nu = 0.06` eV)
- `T_ncdm = 0.7137658555036082`
- `YHe = 0.2454006` explicitly; no BBN inference in the adjudicative run
- `tau_reio = 0.05430842`
- `n_s = 0.9660499`
- `A_s = 2.100549e-9`
- `recombination = HyRec`
- `gauge = synchronous`
- `non linear = halofit`
- `output = tCl,pCl,lCl,mPk`; `lensing = yes`; `l_max_scalars = 2500`

All other settings are the resolved defaults of the exact CLASS commit. The run must save CLASS's resolved parameter/precision files so these defaults are auditable. No parameter may be changed after output inspection.

## Required audited quantities

Both tracks must record: `H0`, `h`, `omega_b`, `omega_cdm`, `omega_m`, `Omega_b`, `Omega_cdm`, `Omega_m`, `Omega_k`, `N_ur`, `N_ncdm`, every `m_ncdm` and `T_ncdm`, CLASS-reported `N_eff`, `YHe` and BBN treatment, `A_s`, `n_s`, recombination backend, gauge, resolved precision settings, `z_*`, `z_d`, `r_s(z_*)`, `r_d = r_s(z_d)`, `100 theta_*`, and exact CLASS/HyRec identities.

## Reference and diagnostic comparisons

- Adjudicative reference: two independent executions of Track B from the same frozen source/input/runtime identity. Run 1 creates the reference outputs; Run 2 must reproduce them under the frozen tolerances.
- Input provenance: compare the canonical input byte-for-byte with the frozen R0 `upstream_planck2018.ini` values, except for the explicit additions listed above (`recombination`, `gauge`, `Omega_k`, `l_max_scalars`, and evidence-output controls).
- Backend diagnostic: run an otherwise identical canonical input with `recombination = RECFAST`. Report HyRec-minus-RECFAST differences, but do not use their magnitude as a PASS/FAIL criterion.
