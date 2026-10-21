# R2 Reference Recovery Report

Date: 2026-10-21
Decision: **REFERENCE ADMISSION COMPLETE — NUMERICAL EXECUTION AUTHORIZED BUT NOT YET RUN**

## Boundary observed

This work performed source recovery, figure forensics, independent digitization and tolerance freezing only. It did not compile or execute CLASS/HyRec and did not produce, inspect or use any local reproduction curve.

## Primary-source recovery

Official arXiv v3 source/PDF and the APS/CHORUS accepted manuscript were preserved and hashed. The arXiv source contains author-submitted standalone Matplotlib PDFs for Figs. 2, 3 and 18. No machine-readable Fig. 2/3 arrays, original plotting code, exact plotting command, journal supplement, or ancillary table was found.

The full visible `misharash/class_public` Git history and refs, relevant deleted filenames, release state, author-linked mock-CMB repository, other relevant author repositories, and public data-deposit searches were checked. No primary raw Fig. 2/3 table emerged. `cobaya_mock_cmb` data are for mock forecast spectra and a different clumping fiducial; they are not substituted for the frozen M3 point.

## Figure forensics and independent extraction

`pdfimages -list` reports no raster objects in the standalone Fig. 2/3 PDFs. Both were produced by Matplotlib 3.4.x and preserve path vertices, strokes, dash arrays, clipping and axes.

- Method A parses vector paths and maps PDF coordinates to data coordinates using embedded ticks. Fig. 2 uses independent left `x_e` and right `g` axes; Fig. 3 uses a log10 `x_e` axis.
- Method B renders the original figure PDF at 1200 dpi, then traces pixels by curve color and line geometry on the independently calibrated raster.
- Only A-vs-B reference dispersion is measured. The row-level mask rejects raster dash gaps and geometry/overlap ambiguity.

The resulting mandatory references are `FIG2_REFERENCE.csv` and `FIG3_REFERENCE.csv`; exact hashes are in `REFERENCE_DATA_MANIFEST.md`.

## Admission

All four Fig. 2 curves are admitted with narrow row masks. All five Fig. 3 curves are admitted, with dashed/occluded samples masked explicitly. Fig. 18 is preserved and vector-extractable but deferred from the mandatory gate because Fig. 2/3 already provide a complete deterministic recombination benchmark and the Fig. 18 spectra are secondary under `BENCHMARK_SPEC.md`.

## License boundary

The author CLASS fork has no repository-level LICENSE/COPYING file at the pinned commit. Its README identifies upstream CLASS and the fork authors; isolated files contain component-specific notices (for example BSD or RECFAST notices), but these do not establish a license for the whole fork. Therefore author source bytes are not committed here. Local scientific inspection/execution only; record commit, tree/diff and hashes without inferring redistribution rights.

## Gate conclusion

Reference quality is sufficient to predeclare auditable pointwise tolerances without seeing a local model result. The selected benchmark remains unchanged. This work authorizes a separate `R2 Numerical Reproduction` work but performs none of it.
