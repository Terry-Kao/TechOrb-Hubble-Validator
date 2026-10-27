# QF-0A Environment + Source Freeze Report — 2026-10-27

**Status:** `PROPOSED / DRAFT / NOT EXECUTED`

## Authority and bindings

- PR #10 approved head: `9b29b40156a68797dc36c6ebd7555e9a3515ae4f`
- PR #10 merge / authoritative main: `2d125671c0d882df6d1de4eb2a0066d5bad989de`
- Protocol Git blob: `c11599a058caa8dfb0912c4e00a0656bef4688bd`
- Protocol SHA-256: `fb7969c11e9e2d4b1eebdd2495692dc422eca418bbbaa85bbabcda2815811d8b`
- Preregistration Git blob: `0b7e7b7046044e1bb9310df566342073dd46090b`
- Preregistration SHA-256: `01ec67801dbe1626c4012234e0182284a30b747ed30ea839842a84df052fd395`
- Source baseline commit: `946965d47fa66d0822cd8023534ee282fddc8f98`
- Source manifest: `foundation/qf0/qf0a_impl/manifests/QF0A_SOURCE_MANIFEST_v1.0.json`
- Source manifest SHA-256: `c8d06cb95095a022870152115c591774c64f4df73e54c5a576054e7ad4d35f27`

## Exact environment

- CPython 3.12.10, AMD64, python.org installer SHA-256 `67b5635e80ea51072b87941312d00ec8927c4db9ba18938f7ad2d27b328b95fb`
- Base executable SHA-256 `4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a`
- NumPy 2.1.3 wheel SHA-256 `0d30c543f02e84e92c4b1f415b7c6b5326cbe45ee7882b6b77db7195fb971e3a`
- SciPy 1.14.1 wheel SHA-256 `2ff38e22128e6c03ff73b6bb0f85f897d2362f8c052e3b8ad00532198fbdae3f`
- Scientific dependency lock SHA-256 `9ec9da7aaa0f7a7b85d9cb33909d03b0b934f0948c0fbb1c1182274e2edea0b2`
- Validation-only `jsonschema[format]==4.23.0`
- NumPy BLAS/LAPACK: scipy-openblas 0.3.27, ILP64, dynamic architecture, Haswell build
- SciPy BLAS/LAPACK: scipy-openblas 0.3.27.dev, dynamic architecture, Zen build
- `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`
- Windows 11 Home 10.0.26200, AMD Ryzen 7 9700X, AMD64, 8 cores / 16 logical processors, Traditional Chinese Taiwan locale, Asia/Taipei timezone

## Source freeze and leakage audit

The source inventory contains protocol binding, an independently implemented analytic reference, a direct constrained/density-conditioning path, the exact six frozen control implementations, provenance/static-audit helpers, and verdict-construction logic. The Option-B scientific API has signature `(operator, state)`, internally fixes `W=(-0.5,0.5)`, checks rank/boundaries/projector residuals/projection norm, and never selects a lowest-absolute-eigenvalue eigenvector. Exact interaction, Clock B, registered map, and coarse-clock construction paths are frozen. No execution CLI exists.

Static leakage audit: `PASS`. The conditional module neither imports nor accepts analytic-reference output; it exposes no external trajectory input, exclusion API, mutable threshold interface, inverse scramble, or cross-program result import. Package import performs no computation.

## Non-adjudicative validation

- Foundation package validator: PASS.
- Preregistration validator: PASS; Draft 2020-12 schemas 4/4, FormatChecker PASS, positive fixtures 8/8, negative fixtures 103/103 expected rejection.
- Artificial/static source tests: 20/20 PASS.
- Protocol blob/SHA-256 binding: PASS.
- Source manifest: 23/23 entries PASS.
- Python compilation and JSON parsing: PASS.
- `git diff --check`: PASS.

## Execution blindness

**Scientific/numerical execution NOT PERFORMED.**

No frozen conditional states, fidelity, trace distance, interaction results, clock-comparison results, negative-control scientific behavior, evidence run, raw array, real machine verdict, or scientific verdict was generated or inspected.

The exact next gate is independent read-only review of this Environment + Source Freeze package. QF-0A execution remains unauthorized.
