# R0 clean-room tolerance contract

**Frozen:** 2026-10-20, before any independent numerical rerun.  
**Status:** binding for the first clean-room adjudication; changes require a new decision-log entry and cannot use rerun results as justification.

## Scope

This contract compares a provenance-preserving rerun of the recovered v26.4 algorithm against the frozen four-row historical summary. It does not test observational fit or physical validity. The historical track must retain the recovered runner's omitted-neutrino convention (`N_ncdm=0`, `N_eff=3.044`). The separately named standard comparison must set `N_ur=2.046`, `N_ncdm=1`, `m_ncdm=0.06`, and `T_ncdm=0.7137658555036082`; it has no historical PASS target.

## Exact identity requirements

The following must match exactly or the adjudication is `BLOCKED`:

- upstream CLASS commit `aa92943e4ab86b56970953589b4897adf2bd0f99`;
- the scientific hunk in `CLASS_FULL_DIFF.patch`, with the three executable-bit-only changes omitted;
- four case IDs, order, mode labels, `eta_lya`, and `b` values in the frozen CSV;
- historical cosmological inputs and implicit-neutrino convention;
- integer multipole grid and number of retained rows for `30 <= ell <= 2500`;
- successful finite outputs and zero exit status for every build and run.

Any missing raw output, generated INI, build/run log, dependency identity, command, or hash is `BLOCKED`, not a numerical PASS.

## Unmodified-versus-patched control

For `b=1`, `eta=0`, identical historical inputs, the clean upstream and patched binaries must agree within all of:

| Quantity | Tolerance |
|---|---:|
| `z_star` | absolute difference <= `5e-6` |
| `r_s` in Mpc | absolute difference <= `5e-6` |
| `100 theta_star` | absolute difference <= `1e-8` |
| each retained TT value | maximum relative difference <= `1e-7` |
| retained TT vector | RMS relative difference <= `1e-8` |

A zero denominator is compared by absolute difference using `1e-30` as the scale floor. Failure of this control is `FAIL` and stops candidate comparison.

## Historical four-case comparison

Every field below must satisfy its row-wise absolute tolerance; no averaging across cases and no post-run relaxation is allowed.

| Historical CSV field | Absolute tolerance |
|---|---:|
| `z_star` | `5e-5` |
| `rs_Mpc` | `5e-5` Mpc |
| `rs_reduction_pct` | `5e-5` percentage points |
| `H0_BAO_naive` | `5e-4` km/s/Mpc |
| `H0_CMB_exact` | `5e-4` km/s/Mpc |
| `theta100_error_ppm` | `5e-2` ppm |
| `ns_compensated` | `5e-6` |
| `As_compensated_1e9` | `5e-5` |
| each RMS residual field | `5e-4` percentage points |
| `max_residual_final_pct` | `5e-4` percentage points |

`est_B_nG` is an algebraic historical label, not a reproduced physical observable; it must match the frozen formula and value within `5e-12` but cannot support model validity.

## Adjudication

- **PASS:** provenance requirements are complete, the unmodified control passes, and all four rows pass every declared field tolerance.
- **FAIL:** the run is complete and auditable but at least one exact identity or numerical criterion fails.
- **BLOCKED:** the run cannot be completed or its provenance/raw evidence is incomplete.

A PASS promotes only the R0 pipeline execution. It does not validate HEM physics, a Planck likelihood, magnetogenesis, or a Hubble-tension solution. R1-R4 remain independently gated.
