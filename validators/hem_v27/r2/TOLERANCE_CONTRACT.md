# Frozen Tolerance Contract — PRE-EXECUTION REFERENCE IDENTITY RE-FREEZE

Status: **REFERENCE IDENTITY RE-FROZEN / NUMERICAL EXECUTION AUTHORIZED BUT NOT YET RUN**. Frozen on 2026-10-21 and byte-identity reverified on 2026-10-22 without producing, opening, or using local R2 model output.

Canonical references are the LF Git blobs recorded in `REFERENCE_DATA_MANIFEST.md`: Fig. 2 `a079416f...` and Fig. 3 `10a160e5...`. Windows CRLF materializations have different transport/checkout hashes and are non-canonical. The identity repair changes no curve value, row, mask, uncertainty, interpolation rule, aggregation rule, or numerical tolerance.

## Reference observables and domains

| Figure | Observable / curves | Grid and domain | Comparison space | Per-point tolerance |
|---|---|---|---|---|
| Fig. 2 | `x_e`: standard dashed, M3 solid | `z=800..1600`, step 10; admitted rows only | linear | absolute `0.012` |
| Fig. 2 | `g [Mpc^-1]`: standard dashed, M3 solid | `z=800..1600`, step 10; admitted rows only | linear | absolute `0.00025 Mpc^-1` |
| Fig. 3 | `x_e`: M3, minus zone, zero/LCDM zone, plus zone, simple clumping | `z=250..1500`, step 10; admitted rows only | `log10(x_e)` | absolute `0.045 dex` |

The reference source is the author-submitted standalone vector figure PDF inside the official arXiv v3 source tar. Method A maps vector path coordinates through tick-derived affine/log transforms. Method B independently renders the same source at 1200 dpi and traces color/stroke pixels. No reproduction curve participates in calibration, identity assignment, masking, uncertainty, or tolerance selection.

## Uncertainty rationale

- Published stroke width is 1.5 pt. Half-stroke uncertainty is approximately `0.0028` in Fig. 2 `x_e`, `0.000062 Mpc^-1` in Fig. 2 `g`, and `0.0111 dex` in Fig. 3.
- 1200 dpi raster quantization is much smaller than half a vector stroke.
- Method A/B median dispersions are below one half-stroke for the continuously drawn curves. Dashed curves show missing raster samples and occasional overlap ambiguity; those rows are excluded rather than guessed.
- The frozen tolerances are conservative two-full-stroke geometric envelopes, rounded outward: `0.012`, `0.00025 Mpc^-1`, and `0.045 dex`. They do not use future model residuals.
- Vector interpolation to the 10-redshift grid is piecewise linear between retained Matplotlib vertices; no extrapolation is allowed.

## Masks and aggregation

`FIG2_REFERENCE.csv` and `FIG3_REFERENCE.csv` contain `admitted=YES/NO` and an exclusion reason. A row is inadmissible if the independent raster method has no stroke at the sample (usually a dash gap) or if reference-vs-reference dispersion exceeds the geometry limit. The frozen admitted counts are:

- Fig. 2: M3 `x_e` 81/81; standard `x_e` 77/81; M3 `g` 81/81; standard `g` 78/81.
- Fig. 3: M3 126/126; minus zone 97/126; zero/LCDM zone 117/126; plus zone 111/126; simple clumping 126/126.

PASS requires every admitted point for every mandatory curve to satisfy its per-point tolerance. There is no mean/RMS escape clause. Missing reproduction coverage, extrapolation, non-finite values, changed masks, or any admitted-point exceedance is FAIL. Exact source/input identities, zero exit status, clean source state, and declared repeatability checks remain mandatory. Fig. 18 is non-adjudicative in this freeze.

Any change to reference bytes, rows, masks, tolerances, interpolation or aggregation requires a new reviewed pre-execution amendment before another numerical run.

The 2026-10-22 Independent Replication Audit independently supported all admitted reference geometry and the unchanged tolerances (`0.012`, `0.00025 Mpc^-1`, `0.045 dex`) while reporting overall `PARTIALLY CONFIRMED — ANOMALIES FOUND`. Its R2 identity anomaly is resolved by direct Git-blob recovery documented in `R2_REFERENCE_IDENTITY_REPAIR_REPORT.md`; the audit's independent scientific comparisons remain supporting evidence.
