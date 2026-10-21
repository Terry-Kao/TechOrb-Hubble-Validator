# Equation Transcription

From Sec. II of arXiv `2108.02747` v3 / PRD 104, 103517:

- `Delta_i = n_H,i / <n_H> = 1 + delta_i`.
- Volume conservation: `sum_i f_i = 1`.
- Baryon-mass conservation: `sum_i f_i Delta_i = 1`, equivalently `sum_i f_i delta_i = 0`.
- Average zone: `delta_0 = 0`.
- Clumping variance, Eq. (1): `b = <(n_H-<n_H>)^2>/<n_H>^2 = -delta_- delta_+ (1-f_0)`.
- Averaged ionization fraction, Eq. (2): `x_e(z) = sum_i f_i Delta_i x_e,i(z)`.

For `delta_-=-0.9`, `delta_+=5/3`, `f_0=1/3`:

- `f_-/f_+ = -delta_+/delta_- = 50/27`;
- `f_- = 100/231`, `f_+ = 18/77`;
- `b=1` exactly;
- all conservation identities equal one or zero algebraically.

Implementation mapping at author commit: `delta_m`, `delta_p`, `f2V`; recombination selector `recfast_3zones_lowlevel`. The code runs three RECFAST histories and mass-weights their intrinsic recombination tables with `f_i Delta_i`, consistent with Eq. (2). This transcription is frozen before any R2 run.
