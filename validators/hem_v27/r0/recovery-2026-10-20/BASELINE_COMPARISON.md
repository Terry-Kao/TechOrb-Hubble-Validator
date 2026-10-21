# Baseline reconstruction and neutrino comparison

These inputs are reconstructed from the recovered runner, not recovered historical per-run INIs. Do not substitute them for missing run provenance.

| Field | Historical runner | Pinned CLASS default / upstream Planck example | Implication |
|---|---|---|---|
| CLASS | v3.2.0 requested; recovered HEAD aa92943e4ab86b56970953589b4897adf2bd0f99 | GitHub tag matches that commit | Candidate source identity recovered |
| Recombination | Explicit RECFAST | CLASS also contains HyRec, but presence does not prove use | Do not label this a HyRec run |
| Massive species | N_ncdm omitted | source/input.c:5698 sets N_ncdm=0 | Historical baseline has no massive neutrino species |
| Relativistic species | N_ur omitted | source/input.c:5687 uses N_eff=3.044 | Default radiation convention must be preserved in historical track |
| Planck example neutrinos | Not used by runner | N_ur=2.046; N_ncdm=1; m_ncdm=0.06; T_ncdm=0.7137658555036082 | Explicit 0.06 eV comparison uses these four values together |
| h | 0.674 | Upstream example H0=67.32117 | Runner is not the exact upstream Planck best-fit input |
| omega_b / omega_cdm | 0.022383 / 0.11933 | 0.02238280 / 0.1201075 | More differences than neutrino mass alone |
| n_s / A_s | 0.965 / 2.1e-9 | 0.9660499 / 2.100549e-9 | Must not silently relabel comparison as historical reproduction |
| Helium / reionization | Omitted; CLASS defaults | Example YHe=0.2454006, tau_reio=0.05430842 | Freeze defaults and convention explicitly before R1 |
| Lensing / outputs | tCl,pCl; no explicit lensing | Example tCl,pCl,lCl,mPk; lensing=yes; halofit | Historical theory TT RMS is not a Planck likelihood |
| Precision | No separate .pre file or precision overrides | Pinned include/precisions.h | Preserve defaults; do not invent a historical precision config |

Local source/input.c SHA256: `f16826b597d361eeccd6a7bf360ae3c9cdb0ee544c326cb2cbb8f8acefc3171d`.
Local include/precisions.h SHA256: `b1d40edc6e649b3ec04688a27dcbdf4fb3c538b4f9b8c786e513b866e4157bb5`.
Upstream Planck example blob: `3b6808a0d18b32cfbf21af798aad5a1255e6cf81`; its bytes match the recovered example.

## Two tracks

1. **Historical reconstruction:** `configs/historical_baseline_reconstructed.ini`; omitted fields remain omitted. Use HIA_B_CLUMPING=1.0 and HIA_LYA_EXP=0.0. Recover exact historical behavior before changing conventions.
2. **Neutrino-only comparison:** `configs/massive_neutrino_comparison_reconstructed.ini`; same declared runner fields, plus the four neutrino settings from the pinned Planck example. Root path differs only to preserve outputs. This is a new controlled comparison, not a claim to reproduce the complete Planck example or its likelihood.

The untouched `configs/upstream_planck2018.ini` is a third reference artifact. R1 must audit all parameter differences before calling any setup the standard Planck baseline. No numerical neutrino comparison has been run.
