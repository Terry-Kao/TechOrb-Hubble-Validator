# Historical Exclusion Study: Tested HIA Late-Time Subclasses

> **Governance notice (2026-10-20):** This directory preserves a February 2026 negative-result study. Its evidence applies only to the tested kinematic, screened-gravity, and smooth-disformal implementations under their recorded assumptions. It is **not** a universal no-go theorem for every late-time Hubble-tension model. Universal wording in historical filenames and manuscript text is superseded by this notice and by `validators/late_time/EXCLUSION_MAP_V1_V25.md`.

## Overview

Through HIA v1–v25, this project explored selected late-time (`z < 0.5`) modifications. The retained code, manuscript, plot, and audit log are historical evidence of those experiments and their failure modes; they do not demonstrate exhaustive coverage of theory space.

## Files

- `manuscript.tex`: historical manuscript, now explicitly retitled and scoped as an exclusion study.
- `Evidence_Code_v25.py`: stored disformal ray-tracing implementation used for the recorded numerical result.
- `Numerical_Audit_Log.md`: historical iteration narrative; current interpretation is bounded by the exclusion map.
- `hia_disformal_raytrace.png`: stored output image; full R0-style provenance has not been frozen.

## Bounded Findings

1. **Tested smooth disformal implementation:** the coupling used to obtain the target distance-modulus shift produced a recorded photon/GW timing difference incompatible with the adopted GW170817 constraint.
2. **Tested screened-gravity implementation:** under its stated screening assumptions, the white-dwarf interior did not obtain the required `G_eff` shift while Solar-System constraints were imposed.
3. **Tested kinematic patch family:** acoustic-angle compensation drove its baseline parameters into tension with the physical constraints adopted in that analysis.

These are `EXCLUDED: IMPLEMENTATION` or `EXCLUDED: SUBCLASS` historical results—not statements that all late-time physics is impossible.

## Reuse Warning

Before reuse, independently audit model equations, parameter domain, units, code, provenance, external constraints, and numerical tolerances. The old optimizer and ray-tracing results are not a multi-probe likelihood and do not validate Tech-Orb Foundation.

