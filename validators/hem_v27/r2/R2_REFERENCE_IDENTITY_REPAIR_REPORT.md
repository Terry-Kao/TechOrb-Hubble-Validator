# R2 Reference Identity Repair / Re-Freeze Report

Date: 2026-10-22 (Asia/Taipei)

Repository: `Terry-Kao/TechOrb-Hubble-Validator`

Branch: `hem-v27-r2-published-model-reproduction`

Boundary: reference artifact identity/provenance only; no R2 CLASS/HyRec published-model numerical reproduction was executed, opened or used.

## Verdict

**REFERENCE IDENTITY RE-FROZEN — NUMERICAL EXECUTION AUTHORIZED BUT NOT YET RUN**

The declared hashes were not orphaned. They are the exact LF Git blobs already present in the original pre-execution freeze commit `1f2ca42ec4f3b1a112108fa41f72103e323d1f44`. The apparent conflicting hashes are Windows CRLF materializations of the same rows and fields.

## Before / after identity table

| Artifact | Audit-observed Windows CRLF representation | Recovered canonical LF Git blob | Canonical decision |
|---|---|---|---|
| Fig. 2 | 28,824 bytes; `e2ac0de1ef8e937150c1dcdf12887d2c3c0c8666576efa3be365735c99e2c56f` | blob `6ac9cccff730f8110d9392d4ab75e3e81b5f3146`; 28,499 bytes; `a079416fdf59e040233f4fecc67f5d8f920ff038a654c8c7758a4c7aab051a05` | retain original canonical identity |
| Fig. 3 | 63,428 bytes; `cdde7d52657906789a80fa4444c92f7658d137ae008c29110de55d7424a2c79b` | blob `602dedb0d62246829b8d837232eb5dae33c3b807`; 62,797 bytes; `10a160e50d5f67b11d678ce7beb709b2c782846eb2f8fff9f39177782c32c719` | retain original canonical identity |

“After” does not mean that curve bytes were rewritten in this repair. It means the existing canonical Git blob identity is now unambiguously named and re-frozen. The CSV content itself is unchanged.

## Phase A — forensic recovery

The search covered the PR branch and its reachable history, reflog-only amended commits, the local R2 recovery workspace and its evidence tarball, the current checkout, the Independent Audit workspace and its PR-evidence tar, the task workspace, and matching temporary/backup filenames. Git `fsck` and reflogs were inspected for unreachable prior versions.

Findings:

1. Pre-execution commit `1f2ca42`, amended predecessors `8cbfde4` and `7489239`, and PR head `223096b` all point to the same Fig. 2/3 blobs: `6ac9ccc...` and `602dedb...`.
2. Direct raw blob extraction produces the declared SHA-256 values exactly.
3. The local recovery workspace, its tarball, the Independent Audit PR evidence, and Windows checkouts contain CRLF materializations. No conflicting curve/content variant was found.
4. Exact byte diff: Fig. 2 has 325 CR bytes added before LF; Fig. 3 has 631 CR bytes added before LF. Removing those CR bytes gives the canonical sizes/hashes exactly. Encoding remains UTF-8 without BOM, newline-at-EOF remains present, and every other byte is identical.
5. The declared bytes therefore have complete provenance: generated and committed during reference recovery, frozen at `1f2ca42` on 2026-10-21 before any R2 numerical run, and derived only from the official arXiv figure assets through the recorded Method A/B process.

## Phase B — canonical selection rationale

The canonical selection rule was fixed independently of any future model output: prefer bytes that are pre-execution, directly traceable to the recovery method, independently supported from official arXiv figure geometry, and consistent with the frozen masks/tolerances. The recovered LF Git blobs satisfy every condition. No numerical reproduction output exists to influence selection.

## Full schema and content validation

Both files decode as UTF-8 without BOM, use the declared 11-column ordering, end with a newline, contain no duplicate `(figure, curve, z)` keys, and have strictly sorted 10-redshift grids.

| File / curve | Rows | Admitted | Validation |
|---|---:|---:|---|
| Fig. 2 `xe_m3` | 81 | 81 | PASS |
| Fig. 2 `xe_standard` | 81 | 77 | PASS |
| Fig. 2 `g_m3` | 81 | 81 | PASS |
| Fig. 2 `g_standard` | 81 | 78 | PASS |
| Fig. 3 `M3` | 126 | 126 | PASS |
| Fig. 3 `minus_zone` | 126 | 97 | PASS |
| Fig. 3 `zero_zone_LCDM` | 126 | 117 | PASS |
| Fig. 3 `plus_zone` | 126 | 111 | PASS |
| Fig. 3 `simple_clumping` | 126 | 126 | PASS |

All `admitted=NO` rows have an explicit recorded reason; admitted rows have none. Numeric fields are finite and preserve the committed decimal formatting. Method A reference values, Method B values, dispersion fields, masks and exclusion reasons are unchanged.

## Independent Audit reconciliation

The 2026-10-22 Independent Replication Audit overall verdict is recorded as `PARTIALLY CONFIRMED — ANOMALIES FOUND`. Its independent vector comparison supports every admitted curve within tolerance; its 1200 dpi raster check remains within the same envelopes. It independently supports the frozen absolute tolerances:

- Fig. 2 `x_e`: `0.012`;
- Fig. 2 `g`: `0.00025 Mpc^-1`;
- Fig. 3 `log10(x_e)`: `0.045 dex`.

The audit's artifact-identity anomaly arose from hashing CRLF materializations instead of the raw Git blobs. This report corrects that provenance interpretation without discarding the audit's scientific comparisons or rewriting its historical report.

## R0 bounded provenance correction

Direct object extraction also resolves the R0 text-byte discrepancy:

- `run_colab_v26_4.py`: canonical LF Git blob/historical original 14,068 bytes, `260bbe8642a9a7d6b0c6ad1734697e4087990d8ffb2c305145fbbf2274ddfb9e`; Windows CRLF working representation 14,358 bytes, `b911db1b2d7fd15ef5d3f59b0cabb47fd5a61fdad86376bf046868bdb8ce0209`.
- `wrap_recfast.c`: canonical LF Git blob/historical original 15,174 bytes, `ab460fc673088078cca27b250652a4698ced1802c7b10aeadb4cd658a47cf75f`; Windows CRLF working representation 15,547 bytes, `382b93b8b2ca8df2b6932ca40b18ab201145008a1308ef5c115a6e84ee83eb27`.

CRLF-to-LF conversion reproduces both canonical hashes exactly. This clarification changes neither R0 PASS / 52-of-52, binary/output exact reproduction, nor R1 PASS.

## Post-repair verification and gate

The pre-execution identity re-freeze commit is `f9037a9a19d1591d25e3ea98acfa8ce6468bdb43`. After push, GitHub reported that exact PR #8 head while the PR remained Draft. Fresh GitHub Contents/Git Blobs API reads returned:

- Fig. 2 blob `6ac9cccff730f8110d9392d4ab75e3e81b5f3146`, 28,499 bytes, SHA-256 `a079416fdf59e040233f4fecc67f5d8f920ff038a654c8c7758a4c7aab051a05`;
- Fig. 3 blob `602dedb0d62246829b8d837232eb5dae33c3b807`, 62,797 bytes, SHA-256 `10a160e50d5f67b11d678ce7beb709b2c782846eb2f8fff9f39177782c32c719`.

Both equal `REFERENCE_DATA_MANIFEST.md` exactly. The re-freeze remains earlier than any R2 numerical execution because no such execution occurred. PR #8 remains Draft/unmerged and R2 numerical execution remains outside this work.

Gate conclusion: **READY FOR R2 NUMERICAL REPRODUCTION**. This authorizes only a separate bounded future work; it does not mean R2 has run or passed.
