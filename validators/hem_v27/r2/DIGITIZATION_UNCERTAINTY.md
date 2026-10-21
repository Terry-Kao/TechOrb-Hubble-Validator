# Digitization Uncertainty and Admission Mask

Uncertainty is derived only from figure geometry and Method A/B reference agreement.

| Curve | A/B median dispersion | A/B p95 | Maximum before masking |
|---|---:|---:|---:|
| Fig. 2 `x_e` standard | 0.00172 | 0.00963 | 0.0711 |
| Fig. 2 `x_e` M3 | 0.000530 | 0.00136 | 0.00356 |
| Fig. 2 `g` standard | 0.0000327 Mpc^-1 | 0.000205 Mpc^-1 | 0.000672 Mpc^-1 |
| Fig. 2 `g` M3 | 0.0000208 Mpc^-1 | 0.0000427 Mpc^-1 | 0.0000712 Mpc^-1 |
| Fig. 3 M3 | 0.00310 dex | 0.0118 dex | 0.0253 dex |
| Fig. 3 zero/LCDM | 0.00603 dex | 0.0568 dex | 0.135 dex |
| Fig. 3 minus zone | 0.00213 dex | 0.0110 dex | 0.0138 dex |
| Fig. 3 plus zone | 0.00373 dex | 0.0143 dex | 0.0201 dex |
| Fig. 3 simple clumping | 0.00196 dex | 0.00412 dex | 0.00434 dex |

Large maxima are localized dash-gap/overlap or wrong-stroke ambiguity and are excluded, not absorbed into tolerance. Admission geometry limits are `0.0113` for Fig. 2 `x_e`, `0.00025 Mpc^-1` for Fig. 2 `g`, and `0.045 dex` for Fig. 3. These are approximately two full 1.5 pt strokes in the corresponding data coordinate.

The final tolerance is the same conservative outward-rounded envelope: `0.012`, `0.00025 Mpc^-1`, and `0.045 dex`. Pointwise masks and reasons are frozen in the CSVs. Future reproduction values cannot change them.
