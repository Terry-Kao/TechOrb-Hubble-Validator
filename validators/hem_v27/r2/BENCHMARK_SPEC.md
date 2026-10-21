# Frozen Benchmark Specification

## Identity

- Paper: Rashkovetskyi, Muñoz, Eisenstein & Dvorkin (2021), “Small-scale clumping at recombination and the Hubble tension”.
- Journal/DOI: Physical Review D 104, 103517; `10.1103/PhysRevD.104.103517`.
- arXiv: `2108.02747` v3, 17 November 2021. No erratum located as of 2026-10-21.
- Author code: `misharash/class_public` commit `4fdab292a223a0ab331e98613ad98e38a3a2d0de` (CLASS 2.9 fork).
- Code license: **UNKNOWN**; GitHub reports no repository license. Local scientific inspection/execution only; do not redistribute source in this repository.

## Exact benchmark

- Locators: Sec. II Eqs. (1)–(2), Fig. 2 caption, Fig. 3 caption, Appendix A.1 and Fig. 18.
- Backend: `recfast_3zones_lowlevel` for the primary benchmark; `hyrec_3zones_lowlevel` is a non-adjudicative backend control unless reference data for Fig. 18 is admitted.
- Parameters: `delta_m=-0.9`, `delta_p=5/3`, `f2V=f0=1/3`; derived `b=-delta_m*delta_p*(1-f0)=1`.
- Derived volume fractions: `f_minus=100/231`, `f0=1/3`, `f_plus=18/77`; density factors `Delta_minus=0.1`, `Delta0=1`, `Delta_plus=8/3`.
- Cosmology: paper’s Planck 2018 best-fit parameter file `base_2018_plikHM_TTTEEE_lowl_lowE_lensing.ini` from the exact author commit, subject to byte hash admission. The paper uses massless neutrinos. BBN/YHe behavior and every resolved CLASS parameter must be captured before execution.
- Control: identical cosmology with standard `recfast` and no zone modification.
- R1 control: rerun the R1 canonical-modern HyRec input unchanged in the R1 source/runtime; it is provenance control only and is not the paper baseline.
- Observables: `x_e(z)` and visibility `g(z)` over the published Fig. 2 domain (approximately `800<=z<=1600`), plus zone and averaged `x_e(z)` over Fig. 3 (`250<=z<=1500`). CMB spectra are secondary until an auditable Fig. 18 reference table exists.

## Comparison procedure

Interpolate reproduction output to admitted reference redshifts without extrapolation. Compare `x_e` in linear space for Fig. 2 and log10 space for Fig. 3. Preserve every point, mask decision and residual. Do not optimize parameters or alter cosmology. A PASS requires all mandatory control checks and every admitted reference point to satisfy the frozen contract.

## Reference admission

No author machine-readable Fig. 2/3 table was located. The official arXiv v3 source supplies standalone vector figure PDFs, from which independent vector and 1200 dpi raster extractions have been frozen. `FIG2_REFERENCE.csv`, `FIG3_REFERENCE.csv`, their masks/hashes and `TOLERANCE_CONTRACT.md` now authorize a separate numerical-reproduction work. No numerical execution occurred in the reference-recovery work.
