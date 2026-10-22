# HEM v27 R0 clean-room reproduction report

**Execution date:** 2026-10-20  
**Frozen contract:** `TOLERANCE_CONTRACT.md` committed at `2736cfa87b667945bb6b75ea90b33bc9f7793968` before execution  
**R0 pipeline verdict:** **FAIL**  
**HEM physical-validity verdict:** **NOT APPLICABLE**

## Conclusion

The frozen candidate source and recovered runner behavior were executed in a clean Docker Linux environment. The unmodified-versus-patched control passed exactly, and all four declared Mode1/Mode2 cases completed with retained INIs, logs, raw CLASS background/thermodynamics/spectra files, derived tables, and hashes.

The candidate does **not** reproduce the frozen historical summary within the pre-run tolerance contract. Of 52 row/field comparisons, 50 pass and 2 fail:

| Case | Field | Historical | Clean room | Absolute delta | Frozen tolerance | Verdict |
|---|---|---:|---:|---:|---:|---|
| `Mode1_b1.40` | `max_residual_final_pct` | 2.2655280649581884 | 2.2625240219549823 | 0.003004043003206114 | 0.0005 | FAIL |
| `Mode2_b1.35` | `max_residual_final_pct` | 2.210704735821735 | 2.2116819090465785 | 0.000977173224843586 | 0.0005 | FAIL |

The tolerance was not changed after observing results. This is a completed numerical mismatch, not an environment blocker and not an `UNKNOWN` quantity.

## Identity and environment

- Upstream CLASS commit: `aa92943e4ab86b56970953589b4897adf2bd0f99` (`v3.2.0`).
- Patched checkout: same commit plus only the frozen RECFAST scientific hunk.
- Canonical historical/frozen `wrap_recfast.c` is the LF Git blob (15,174 bytes), SHA-256 `ab460fc673088078cca27b250652a4698ced1802c7b10aeadb4cd658a47cf75f`. A Windows CRLF working-tree materialization is 15,547 bytes and hashes to `382b93b8b2ca8df2b6932ca40b18ab201145008a1308ef5c115a6e84ee83eb27`; it is content-equivalent after CRLF-to-LF conversion but is not the canonical byte identity.
- The three executable-bit-only differences in `CPU.py`, `doc/input/make1.sh`, and `doc/input/make2.sh` were excluded.
- Docker Desktop: 4.61.0; engine/client 29.2.1; Linux `amd64`; kernel `6.6.87.2-microsoft-standard-WSL2`.
- Image: `techorb-r0@sha256:14d858b5c2009fd6e9d5f0621c6d9554f66782dfa3201dd21bf4b26218c48a0a`.
- Base OS: Debian 12.12 slim, base digest `sha256:d5d3f9c23164ea16f31852f95bd5959aad1c5e854332fe00f7b3a20fcc9f635c`.
- Network was disabled for the numerical container. All controllable source, inputs, logs, raw outputs, derived artifacts, and manifests were bind-mounted to H:. Docker/WSL system metadata, layers, VHDX, and engine caches used the normal C: installation under the user's explicit 2026-10-20 authorization.

Exact compiler, make, Git, Python, pip, NumPy, pandas, and Matplotlib identities are stored in `cleanroom/ENVIRONMENT_MANIFEST.json`; Docker and image identity are stored in `cleanroom/DOCKER_ENVIRONMENT.json`.

## Baselines and controls

The historical track preserved the recovered runner convention: no `N_ncdm` line, CLASS default `N_ncdm=0`, and reported `N_eff=3.044`. A separate standard comparison used explicit `N_ur=2.046`, `N_ncdm=1`, `m_ncdm=0.06`, and `T_ncdm=0.7137658555036082`. The standard comparison has no historical PASS target.

The clean upstream and patched binaries were both run at `b=1`, `eta=0` with identical historical inputs. Results were exact across 2499 multipoles:

| Check | Observed | Frozen limit | Verdict |
|---|---:|---:|---|
| `z_star` absolute difference | 0 | 5e-6 | PASS |
| `r_s` absolute difference | 0 | 5e-6 Mpc | PASS |
| `100 theta_star` absolute difference | 0 | 1e-8 | PASS |
| maximum TT relative difference | 0 | 1e-7 | PASS |
| RMS TT relative difference | 0 | 1e-8 | PASS |

## Evidence preservation

The complete H:-resident raw package contains 371 files and 401,161,493 bytes before ZIP compression. It includes every generated INI, environment record, exit code, stdout/stderr log, raw CLASS background/thermodynamics/Cl output, derived CSV/figure, build command/log, and SHA-256 entry.

- Full archive: `H:\AI_Products\AI_Factory_Workspace\TechOrb-Hubble-Validator-R0-cleanroom-20261020\R0_CLEANROOM_RAW_EVIDENCE_2026-10-20.zip`
- Archive bytes: `107752158`
- Archive SHA-256: `23dfceb04e68cba4677186e4984c9b3c784259232d5b0f817ce6472c098f15d6`

The archive exceeds GitHub's ordinary single-file limit, so the branch commits the complete file-level SHA-256 manifest, all commands/logs/INIs/parsed records, the harness and Dockerfile, and derived comparison products; the 400 MB raw `.dat` bytes remain on H: under the archive hash above. This storage distinction does not affect the verdict because the raw bytes were retained and hashed before adjudication.

## Provenance separation

- `PROV-01` remains a historical limitation: the original historical compiler, dependencies, binary identity, and source-to-output linkage were not recovered.
- `RAW-01` remains a historical limitation: the original historical generated INIs, CLASS stdout/stderr, and raw numerical files were not recovered.
- The new clean-room run has complete provenance and raw evidence. Therefore `PROV-01` and `RAW-01` do not convert the observed candidate mismatch to `UNKNOWN` or indefinitely block rerun adjudication.
- Candidate-source adjudication: **FAIL** under the frozen tolerance contract.
- Claim scope: this result does not assess HEM physical validity, Planck likelihood fit, magnetogenesis, or the Tech-Orb mother hypothesis.

R1-R4 remain **NOT STARTED**.

