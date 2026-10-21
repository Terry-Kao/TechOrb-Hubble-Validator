# HEM v27 R1 Baseline Definition Audit Report

- Pre-registration commit: `79247c1e97d433c38d0f1fc0f1e9366866f5e475`
- CLASS: v3.2.0 `aa92943e4ab86b56970953589b4897adf2bd0f99`
- Runtime: Alpine 3.20.10 / musl 1.2.5-r3 / GCC 13.2.1
- Pipeline verdict: **PASS**
- Model verdict: **NOT APPLICABLE**

## Admission record

The first canonical attempt is retained but rejected because its copied source tree had 241 dirty paths before build. No result from that attempt is used for adjudication. The accepted attempt was built from a Git archive of the frozen commit; scientific inputs and tolerances were unchanged.

## Historical-v26.4 control

All three accepted R0-H artifacts are byte-identical to their frozen hashes. The historical track remains an implicit-`N_ncdm=0`, RECFAST historical control only.

## Canonical-modern HyRec baseline

Two independent executions produced byte-identical background, thermodynamics, unlensed/lensed CMB, and linear/nonlinear matter-spectrum tables.

| Quantity | Value |
|---|---:|
| H0 | 67.32117 km/s/Mpc |
| h | 0.6732117 |
| Omega_b | 0.049386824645 |
| Omega_cdm | 0.265012779505 |
| Omega_ncdm | 0.001407493164 |
| Omega_m | 0.315807097315 |
| N_ur / N_ncdm / m_ncdm / T_ncdm | 2.046 / 1 / 0.06 eV / 0.7137658555036082 |
| CLASS-reported N_eff | 3.046000 |
| YHe / treatment | 0.2454006 / explicit, no BBN inference |
| A_s / n_s | 2.100549e-9 / 0.9660499 |
| recombination / gauge | HyRec / synchronous |
| z_* | 1085.136276 |
| z_d | 1059.927633 |
| r_s(z_*) | 144.493574 Mpc |
| r_d = r_s(z_d) | 147.054325 Mpc |
| 100 theta_* | 1.044142 |

## RECFAST diagnostic (non-adjudicative)

HyRec is the frozen canonical backend. The otherwise identical RECFAST run changed `z_*` by +0.020124, `z_d` by +0.009761, `r_s(z_*)` by -0.001712 Mpc, `r_d` by -0.000883 Mpc, and `100 theta_*` by -0.000013. These differences are reported without an acceptance threshold.

## Scope

R1 PASS means only that the historical and canonical baseline definitions are explicit and reproducible under the frozen contract. It is not evidence that HEM or Tech-Orb is physically valid, is not a cosmological posterior, and does not solve the Hubble tension. R2 numeric work has not started.
