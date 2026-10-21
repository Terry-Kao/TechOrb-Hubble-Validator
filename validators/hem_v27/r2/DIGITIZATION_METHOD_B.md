# Digitization Method B — Independent 1200 dpi Raster Trace

The same frozen standalone figure PDFs were rendered independently at 1200 dpi. This method does not read Method A vertices. It calibrates pixel coordinates from the rendered axes and traces narrow vertical slices using RGB masks for black, purple, blue, dark red and olive strokes.

At each 10-redshift sample, the tracer records candidate stroke centers. Dashed-line gaps remain missing rather than being filled. Curve candidates are associated to caption/legend color and line style, then compared only against Method A reference geometry. The raw output is preserved in `method-b-fig2-raster.csv` and `method-b-fig3-raster.csv`.

Raster rows are not silently repaired. Missing dashes, line overlap, legend interference or excess reference-vs-reference dispersion produce an explicit `admitted=NO` row in the final reference CSV.

OCR was not used for curve values. Embedded PDF text and vector ticks supplied the axis labels and calibration cross-check.
