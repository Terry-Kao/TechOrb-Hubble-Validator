# R1 Compact Evidence Index

- `R1_AUDIT_REPORT.md`: human adjudication report.
- `R1_AUDIT_RESULT.json`: machine-readable checks and verdict.
- `CANONICAL_COMPARISON.csv`: Run 1/Run 2 hashes for eight output families.
- `run1.ini` and `run1_00_parameters.ini`: frozen canonical input and CLASS-resolved parameters.
- runtime and executable identity text files.
- `SHA256SUMS.txt`: full H:-workspace manifest (1,080 entries; independently verified with 0 missing and 0 mismatch).

The first copied-source attempt is preserved in the full H: workspace and rejected because it had 241 dirty paths before build. The accepted run used a Git archive of the exact frozen CLASS commit with unchanged scientific inputs and tolerances.
