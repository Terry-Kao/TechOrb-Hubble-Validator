# Digitization Method A — PDF Vector Paths

Source assets are the author-submitted standalone Matplotlib PDFs `M3demo-rec.pdf` and `M3demo-rec-details.pdf` from arXiv `2108.02747v3` source. Their hashes are frozen in `REFERENCE_DATA_MANIFEST.md`.

The PDFs were converted to SVG without rasterization. Data paths were identified by stroke color, width and dash array, separately from axes, ticks and legend samples. Axis transforms were solved from embedded tick geometry:

- Fig. 2 redshift: `(x=94.153311 -> z=800)` and `(x=348.247843 -> z=1600)`.
- Fig. 2 `x_e`: `(y=56.617479 -> 0)` and `(y=322.804115 -> 1)`.
- Fig. 2 `g`: `(y=56.617479 -> 0)` and `(y=297.700875 -> 0.020 Mpc^-1)`.
- Fig. 3 redshift: `(x=100.175899 -> z=250)` and `(x=415.035993 -> z=1500)`.
- Fig. 3 log axis: `(y=56.617479 -> log10(x_e)=-4)` and `(y=327.348431 -> 0)`.

Curve identity comes from the paper caption/legend and PDF styling: black/purple solid versus dashed in Fig. 2; black solid, blue dash-dot, black dashed, dark-red dash-dot and olive solid in Fig. 3. Short legend samples are excluded by path span. Native vertices are preserved in `method-a-fig*-vector.csv`; frozen reference rows use piecewise-linear interpolation to a 10-redshift grid without extrapolation.

No CLASS/HyRec output or remembered numerical value is an input to this method.
