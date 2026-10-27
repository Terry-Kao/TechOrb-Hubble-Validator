# QF-0A Environment Freeze v1.0

**Status:** `PROPOSED / DRAFT / NOT EXECUTED`

This checkpoint fixes CPython 3.12.10 (64-bit AMD64), NumPy 2.1.3, SciPy 1.14.1, and validation-only `jsonschema[format]==4.23.0`. The scientific wheels and deterministic lock are SHA-256 bound in the JSON companion.

The intended host is Windows 11 Home build 26200 on an AMD Ryzen 7 9700X. NumPy reports scipy-openblas 0.3.27 (ILP64); SciPy reports scipy-openblas 0.3.27.dev. All five registered thread variables are frozen to `1`. Numeric types are `float64` and `complex128`.

Installation uses the python.org CPython 3.12.10 AMD64 installer and binary wheels only. Scientific dependencies are installed with `--require-hashes`; validation dependencies remain a separate scope.

No frozen QF-0A scientific calculation, result, evidence run, or machine verdict was produced or inspected while collecting this identity.
