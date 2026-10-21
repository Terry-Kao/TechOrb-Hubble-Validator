# Tech-Orb / HEM v26.4 — R0-H Historical Environment Reconciliation Final Report

- **Experiment ID**: `R0-H — Historical Environment Reconciliation`
- **Date**: `2026-10-21`
- **Repository**: `Terry-Kao/TechOrb-Hubble-Validator`
- **Frozen Tolerance Contract Commit**: `2736cfa87b667945bb6b75ea90b33bc9f7793968` (unmodified)
- **Prior R0 Clean-Room Adjudication Commit**: `0b9e74ecebe05b0fe9f3cddf902ae79d6b6d4500`
- **Workspace Root (`R0H_ROOT`)**: `H:\AI_Products\AI_Factory_Workspace\TechOrb-Hubble-Validator-R0H-historical-reconciliation-20261021`

---

## 1. Executive Summary

- **R0-H Verdict**: **`PASS — HISTORICAL ENVIRONMENT RECONCILED`**
- **52-Field Frozen Tolerance Summary (Candidate B — Alpine 3.20.10 Primary Adjudication Run)**:
  - **52 PASS / 0 FAIL** (out of 52 frozen fields in `TOLERANCE_CONTRACT.md`)
  - **Maximum absolute difference across all 52 fields**: `0.0` (exact bit-for-bit reproduction)
- **Previously Failing Debian 12 Clean-Room Fields Reconciled**:
  1. `Mode1_b1.40` `max_residual_final_pct` (frozen tolerance `5.0e-4`):
     - Historical value: `2.2655280649581884`
     - Debian 12 R0 Clean-Room value: `2.2625240219549823` (`|Δ| = 0.003004043003206114` $\rightarrow$ `FAIL`)
     - Alpine 3.20 R0-H Candidate B value: `2.2655280649581884` (`|Δ| = 0.0` $\rightarrow$ `PASS`)
  2. `Mode2_b1.35` `max_residual_final_pct` (frozen tolerance `5.0e-4`):
     - Historical value: `2.210704735821735`
     - Debian 12 R0 Clean-Room value: `2.2116819090465785` (`|Δ| = 0.000977173224843586` $\rightarrow$ `FAIL`)
     - Alpine 3.20 R0-H Candidate B value: `2.210704735821735` (`|Δ| = 0.0` $\rightarrow$ `PASS`)
- **Byte-Identical Primary Adjudication Artifacts**:
  - `hem_class_v26_4_exact_lock_results.csv`: `1,319 bytes`, `SHA-256: 81d840b89fd1888c306f53fd02a4ac9ef678eecc239039fa4e458f574f2f83d1` (**100% byte-identical to historical artifact**)
  - `hem_class_v26_4_residual_curves.csv`: `573,881 bytes`, `SHA-256: 50a9334e5f833eb0af793c130e349bf4472dbedc1b4403fb4ccad520ca806d2e` (**100% byte-identical to historical artifact across all 2,471 $\ell$ rows $\times$ 12 columns = 29,652 values**)
  - `hem_class_v26_4_summary.png`: `522,605 bytes`, `SHA-256: 4bc87d7841e940c3264ac04defec5703b19d648e882641d105f6db121272b607` (**100% byte-identical to historical artifact**)
- **Scope & Scientific Guardrail Statement**:
  - Per Section 0 rules, this `PASS — HISTORICAL ENVIRONMENT RECONCILED` verdict **solely** establishes numerical and forensic reproducibility of the historical v26.4 script outputs when executed in the historical WSL2 Alpine 3.20 (`musl` / `GCC 13.2.1`) environment.
  - It **does NOT** validate HEM as a physical model (`R0 physical-model verdict` remains `NOT APPLICABLE`), **does NOT** constitute a solution to the Hubble Tension, and **does NOT** initiate or authorize `R1`–`R4`.

---

## 2. Historical Binary Forensics (`01_binary_forensics/`)

Forensic inspection of the preserved historical executable `HRS_Research\Colab\class_hia_v26\class` (copied read-only to `01_binary_forensics/class_historical`) established the following facts:

| Property | Value | Evidence Status |
|---|---|---|
| Historical Path | `H:\AI_Products\AI_Factory_Workspace\TechOrb-Hubble-Validator\HRS_Research\Colab\class_hia_v26\class` | `VERIFIED` |
| Size | `3,854,200 bytes` | `VERIFIED` |
| Created Timestamp | `2026-10-19T07:20:13.5529608+08:00` | `VERIFIED` |
| LastWrite Timestamp | `2026-10-19T07:20:14.4062412+08:00` | `VERIFIED` |
| SHA-256 | `a17d2197f33db122729e058dc3d4e9e15ef7c9efc554a408dfb96ca3d01d5150` | `VERIFIED` |
| SHA-1 | `e9a75a10f3e1cebf65f34a68dde13e8210bc102c` | `VERIFIED` |
| ELF Class / Type | `ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV), dynamically linked, with debug_info, not stripped` | `VERIFIED` |
| GNU Build-ID | `4d8e404396dcf56022069af2dc22344f0c9b22ae` | `VERIFIED` |
| Dynamic Loader (`PT_INTERP`) | `/lib/ld-musl-x86_64.so.1` | `VERIFIED` |
| `.comment` Compiler String | `GCC: (Alpine 13.2.1_git20240309) 13.2.1 20240309` | `VERIFIED` |
| `DT_NEEDED` Shared Libraries | `libgomp.so.1`, `libc.musl-x86_64.so.1` | `VERIFIED` |
| DWARF `musl` Source Path | `/home/buildozer/aports/main/musl/src/musl-1.2.5` | `VERIFIED` |
| Embedded `__CLASSDIR__` Path | `/mnt/host/h/AI_Products/AI_Factory_Workspace/TechOrb-Hubble-Validator/HRS_Research/Colab/class_hia_v26` | `VERIFIED` |
| Embedded Environment Symbols | `HIA_B_CLUMPING`, `HIA_LYA_EXP` | `VERIFIED` |

### Bit-for-Bit Binary Reproduction Proof (`04_source_builds/`)
- When clean upstream CLASS `v3.2.0` (`aa92943e4ab86b56970953589b4897adf2bd0f99`) + the frozen `wrap_recfast.c` (`SHA-256: ab460fc673088078cca27b250652a4698ced1802c7b10aeadb4cd658a47cf75f`) is compiled inside Candidate B at `04_source_builds/class_hia_v26`, the resulting binary (`04_source_builds/class_candidate_b`) is `3,854,240 bytes` (`SHA-256: a8c71c76b68b3286a7ff77d4bdabc1b63c93b058b944bcc45ac95d3709c6b2be`) solely because its directory path string (`__CLASSDIR__` and DWARF `DW_AT_comp_dir`) is 37 bytes longer.
- When compiled inside Candidate B using an isolated chroot bind-mount at the exact historical directory path string `/mnt/host/h/AI_Products/AI_Factory_Workspace/TechOrb-Hubble-Validator/HRS_Research/Colab/class_hia_v26` (`04_source_builds/path_matched_class_hia_v26/class`), the freshly compiled binary is **`3,854,200 bytes` with `SHA-256: a17d2197f33db122729e058dc3d4e9e15ef7c9efc554a408dfb96ca3d01d5150` — 100% bit-for-bit identical to `class_historical`** (`VERIFIED`).
- This elevates the provenance of `class_historical` from `STRONG CANDIDATE` to **`VERIFIED` exact historical production executable compiled from clean upstream `v3.2.0` (`aa92943e4ab86b56970953589b4897adf2bd0f99`) + frozen `wrap_recfast.c` (`ab460fc673088078cca27b250652a4698ced1802c7b10aeadb4cd658a47cf75f`) with zero hidden source modifications**.

---

## 3. Historical Environment Evidence (`02_environment_evidence/` & `03_environment_locks/`)

In addition to `HRS_Research\Colab\run_wsl_fallback.sh` (`SHA-256: 86e1c354ceb69ed5ed0b0f2631beb55fc7969c87c701c3fe313d81e96d7e8175`), the preserved historical execution log `task-125.log` from `2026-10-19T07:19:50+08:00` (`02_environment_evidence/historical_task_125_v26_4_execution.log`, `18,662 bytes`, `SHA-256: 9b8745034ccc0da82b0e9f6394a2c5d6c7a92ce80ec1160a10bcce0eafe0d6f4`) records the exact `apk` package installation, `gcc` compile commands, and `run_colab_v26_4.py` console output.

| Component | Historical `task-125.log` (`2026-10-19`) | R0-H Candidate B (`2026-10-21`) | R0 Clean-Room (Debian 12) | Evidence Status |
|---|---|---|---|---|
| Host / Kernel | `Linux 6.6.87.2-microsoft-standard-WSL2 x86_64` | `Linux 6.6.87.2-microsoft-standard-WSL2 x86_64` | `Linux 6.6.87.2-microsoft-standard-WSL2 x86_64` | `VERIFIED` |
| Distro / Rootfs | Alpine Linux `3.20.10` (`alpine-base 3.20.10-r0`) | Alpine Linux `3.20.10` (`alpine-base 3.20.10-r0`) | Debian `12.12 (bookworm)` | `VERIFIED` |
| C Library (`libc`/`libm`) | `musl-1.2.5-r3` (`/lib/ld-musl-x86_64.so.1`) | `musl-1.2.5-r3` (`/lib/ld-musl-x86_64.so.1`) | `glibc 2.36` (`libc.so.6` / `libm.so.6`) | `VERIFIED` |
| C Compiler | `gcc-13.2.1_git20240309-r1` | `gcc-13.2.1_git20240309-r1` | `gcc 12.2.0 (Debian 12.2.0-14+deb12u1)` | `VERIFIED` |
| OpenMP Runtime | `libgomp-13.2.1_git20240309-r1` | `libgomp-13.2.1_git20240309-r1` | `libgomp1 (12.2.0-14+deb12u1)` | `VERIFIED` |
| Binutils / Make | `binutils-2.42-r1`, `make-4.4.1-r2` | `binutils-2.42-r1`, `make-4.4.1-r2` | `binutils 2.40`, `make 4.3` | `VERIFIED` |
| Python | `python3-3.12.13-r0` | `python3-3.12.13-r0` | `python3 3.11.2` | `VERIFIED` |
| NumPy | `py3-numpy-1.25.2-r1` | `py3-numpy-1.25.2-r1` | `numpy 1.24.2` | `VERIFIED` |
| Pandas | `py3-pandas-2.0.3-r2` | `py3-pandas-2.0.3-r2` | `pandas 1.5.3` | `VERIFIED` |
| Matplotlib | `py3-matplotlib-3.7.3-r1` | `py3-matplotlib-3.7.3-r1` | `matplotlib 3.6.3` | `VERIFIED` |
| BLAS / LAPACK | `openblas-0.3.28-r0` | `openblas-0.3.28-r0` | `libopenblas0 0.3.21` | `VERIFIED` |
| Total `apk` Packages | `137 packages` (`359 MiB`) | `137 packages` (`137/137` exact version match) | N/A (`apt`) | `VERIFIED` |

- **Candidate A (Persistent on-disk `/run/linux_env` rootfs)**: `NOT AVAILABLE` (`VERIFIED`), because `/run` inside `docker-desktop` is an in-RAM `tmpfs` mount that resets when WSL2 terminates idle VMs.
- **Candidate B (`CANDIDATE_B_WSL2_ALPINE_3_20`)**: Reconstructed in `wsl -d docker-desktop` using the exact `run_wsl_fallback.sh` recipe and locked prior to numerical execution in `03_environment_locks/ENV_LOCK_CANDIDATE_B.json` (`apk` DB SHA-256: `b902b53083a2f85ba7f17ee8866e32118cea8555d5370ca8316fcfb2e57b3b58`). All 137 installed packages match `task-125.log` 100%.
- **Candidate C**: Not required because Candidate B matched all 137 historical package versions and reproduced all 52 fields with `0.0` error.

---

## 4. Candidate Environment Results (`05_primary_runs/`)

The Primary Adjudication Run executed the unmodified `run_colab_v26_4.py` (`SHA-256: 260bbe8642a9a7d6b0c6ad1734697e4087990d8ffb2c305145fbbf2274ddfb9e`) against upstream CLASS `v3.2.0` (`aa92943e4ab86b56970953589b4897adf2bd0f99`) patched solely with frozen `wrap_recfast.c` (`SHA-256: ab460fc673088078cca27b250652a4698ced1802c7b10aeadb4cd658a47cf75f`).

| Artifact | Historical SHA-256 | Candidate B Primary Run SHA-256 | Byte Size | Match |
|---|---|---|---|---|
| `hem_class_v26_4_exact_lock_results.csv` | `81d840b89fd1888c306f53fd02a4ac9ef678eecc239039fa4e458f574f2f83d1` | `81d840b89fd1888c306f53fd02a4ac9ef678eecc239039fa4e458f574f2f83d1` | `1,319` | **EXACT (`0.0` diff)** |
| `hem_class_v26_4_residual_curves.csv` | `50a9334e5f833eb0af793c130e349bf4472dbedc1b4403fb4ccad520ca806d2e` | `50a9334e5f833eb0af793c130e349bf4472dbedc1b4403fb4ccad520ca806d2e` | `573,881` | **EXACT (`0.0` diff)** |
| `hem_class_v26_4_summary.png` | `4bc87d7841e940c3264ac04defec5703b19d648e882641d105f6db121272b607` | `4bc87d7841e940c3264ac04defec5703b19d648e882641d105f6db121272b607` | `522,605` | **EXACT (`0.0` diff)** |

---

## 5. Full 52-Field Comparison Table (`07_comparisons/FIELD_COMPARISON_CANDIDATE_B.csv`)

All 52 frozen fields from `TOLERANCE_CONTRACT.md` are listed below comparing Historical v26.4, Debian 12 R0 Clean-Room, and Alpine 3.20 R0-H Candidate B:

| Case | Field | Historical Value | R0 Clean-Room (Debian 12) | R0-H Candidate B (Alpine 3.20) | R0 `|Δ|` | R0-H `|Δ|` | Frozen Tolerance | R0 Status | R0-H Status |
|---|---|---|---|---|---|---|---|---|---|
| `Mode1_b1.40` | `z_star` | `1113.85456` | `1113.854562` | `1113.85456` | `2.000e-06` | `0.0` | `5e-05` | `PASS` | `PASS` |
| `Mode1_b1.40` | `rs_Mpc` | `142.551899` | `142.551899` | `142.551899` | `0.0` | `0.0` | `5e-05` | `PASS` | `PASS` |
| `Mode1_b1.40` | `rs_reduction_pct` | `1.489947630528621` | `1.489947630528621` | `1.489947630528621` | `0.0` | `0.0` | `5e-05` | `PASS` | `PASS` |
| `Mode1_b1.40` | `H0_BAO_naive` | `68.4194134292101` | `68.4194134292101` | `68.4194134292101` | `0.0` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode1_b1.40` | `H0_CMB_exact` | `72.86658241314156` | `72.86658241314156` | `72.86658241314156` | `0.0` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode1_b1.40` | `theta100_error_ppm` | `0.6334415486278256` | `0.6334687408280032` | `0.6334415486278256` | `2.719e-05` | `0.0` | `0.05` | `PASS` | `PASS` |
| `Mode1_b1.40` | `ns_compensated` | `0.9406695203760442` | `0.9406705483481465` | `0.9406695203760442` | `1.028e-06` | `0.0` | `5e-06` | `PASS` | `PASS` |
| `Mode1_b1.40` | `As_compensated_1e9` | `2.010590753514204` | `2.010587400989518` | `2.010590753514204` | `3.353e-06` | `0.0` | `5e-05` | `PASS` | `PASS` |
| `Mode1_b1.40` | `rms_residual_naive_pct` | `11.318428443017252` | `11.318429046249404` | `11.318428443017252` | `6.032e-07` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode1_b1.40` | `rms_residual_theta_only_pct` | `6.342968753209839` | `6.343000549491091` | `6.342968753209839` | `3.180e-05` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode1_b1.40` | `rms_residual_final_pct` | `0.9261969639340124` | `0.925977809717088` | `0.9261969639340124` | `2.192e-04` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode1_b1.40` | **`max_residual_final_pct`** | **`2.2655280649581884`** | **`2.2625240219549823`** | **`2.2655280649581884`** | **`3.004e-03`** | **`0.0`** | **`0.0005`** | **`FAIL`** | **`PASS`** |
| `Mode1_b1.40` | `est_B_nG` | `0.2846049894151541` | `0.2846049894151541` | `0.2846049894151541` | `0.0` | `0.0` | `5e-12` | `PASS` | `PASS` |
| `Mode2_b1.35` | `z_star` | `1114.371589` | `1114.371589` | `1114.371589` | `0.0` | `0.0` | `5e-05` | `PASS` | `PASS` |
| `Mode2_b1.35` | `rs_Mpc` | `142.508241` | `142.508241` | `142.508241` | `0.0` | `0.0` | `5e-05` | `PASS` | `PASS` |
| `Mode2_b1.35` | `rs_reduction_pct` | `1.520117357459763` | `1.520117357459763` | `1.520117357459763` | `0.0` | `0.0` | `5e-05` | `PASS` | `PASS` |
| `Mode2_b1.35` | `H0_BAO_naive` | `68.44037400475669` | `68.44037400475669` | `68.44037400475669` | `0.0` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode2_b1.35` | `H0_CMB_exact` | `72.98089388801586` | `72.98089388801581` | `72.98089388801586` | `4.263e-14` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode2_b1.35` | `theta100_error_ppm` | `0.6065883789859653` | `0.6065883789859653` | `0.6065883789859653` | `0.0` | `0.0` | `0.05` | `PASS` | `PASS` |
| `Mode2_b1.35` | `ns_compensated` | `0.941893256717326` | `0.9418923011231118` | `0.941893256717326` | `9.556e-07` | `0.0` | `5e-06` | `PASS` | `PASS` |
| `Mode2_b1.35` | `As_compensated_1e9` | `2.0158568372045096` | `2.015854861654633` | `2.0158568372045096` | `1.976e-06` | `0.0` | `5e-05` | `PASS` | `PASS` |
| `Mode2_b1.35` | `rms_residual_naive_pct` | `11.10957841940066` | `11.109345967275315` | `11.10957841940066` | `2.325e-04` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode2_b1.35` | `rms_residual_theta_only_pct` | `5.979272695229032` | `5.979457770433189` | `5.979272695229032` | `1.851e-04` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode2_b1.35` | `rms_residual_final_pct` | `0.9184594194387046` | `0.9184932273326624` | `0.9184594194387046` | `3.381e-05` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode2_b1.35` | **`max_residual_final_pct`** | **`2.210704735821735`** | **`2.2116819090465785`** | **`2.210704735821735`** | **`9.772e-04`** | **`0.0`** | **`0.0005`** | **`FAIL`** | **`PASS`** |
| `Mode2_b1.35` | `est_B_nG` | `0.2662235902394828` | `0.2662235902394828` | `0.2662235902394828` | `0.0` | `0.0` | `5e-12` | `PASS` | `PASS` |
| `Mode2_b1.40` | `z_star` | `1117.555282` | `1117.555282` | `1117.555282` | `0.0` | `0.0` | `5e-05` | `PASS` | `PASS` |
| `Mode2_b1.40` | `rs_Mpc` | `142.240047` | `142.240047` | `142.240047` | `0.0` | `0.0` | `5e-05` | `PASS` | `PASS` |
| `Mode2_b1.40` | `rs_reduction_pct` | `1.7054519981801748` | `1.7054519981801748` | `1.7054519981801748` | `0.0` | `0.0` | `5e-05` | `PASS` | `PASS` |
| `Mode2_b1.40` | `H0_BAO_naive` | `68.5694185182602` | `68.5694185182602` | `68.5694185182602` | `0.0` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode2_b1.40` | `H0_CMB_exact` | `73.6863198891052` | `73.6863198746445` | `73.6863198891052` | `1.446e-08` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode2_b1.40` | `theta100_error_ppm` | `0.4337438026468395` | `0.4337438026468395` | `0.4337438026468395` | `0.0` | `0.0` | `0.05` | `PASS` | `PASS` |
| `Mode2_b1.40` | `ns_compensated` | `0.9389545578193772` | `0.9389543080604826` | `0.9389545578193772` | `2.498e-07` | `0.0` | `5e-06` | `PASS` | `PASS` |
| `Mode2_b1.40` | `As_compensated_1e9` | `2.0061412435675225` | `2.006139989774764` | `2.0061412435675225` | `1.254e-06` | `0.0` | `5e-05` | `PASS` | `PASS` |
| `Mode2_b1.40` | `rms_residual_naive_pct` | `12.561742071755628` | `12.561804358582346` | `12.561742071755628` | `6.229e-05` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode2_b1.40` | `rms_residual_theta_only_pct` | `6.740555516949118` | `6.740636055189731` | `6.740555516949118` | `8.054e-05` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode2_b1.40` | `rms_residual_final_pct` | `1.0420934710358458` | `1.0420801660356993` | `1.0420934710358458` | `1.331e-05` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode2_b1.40` | `max_residual_final_pct` | `2.525798135050048` | `2.526151461391174` | `2.525798135050048` | `3.533e-04` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode2_b1.40` | `est_B_nG` | `0.2846049894151541` | `0.2846049894151541` | `0.2846049894151541` | `0.0` | `0.0` | `5e-12` | `PASS` | `PASS` |
| `Mode1_b1.60` | `z_star` | `1124.061487` | `1124.061487` | `1124.061487` | `0.0` | `0.0` | `5e-05` | `PASS` | `PASS` |
| `Mode1_b1.60` | `rs_Mpc` | `141.695358` | `141.695358` | `141.695358` | `0.0` | `0.0` | `5e-05` | `PASS` | `PASS` |
| `Mode1_b1.60` | `rs_reduction_pct` | `2.081857660198578` | `2.081857660198578` | `2.081857660198578` | `0.0` | `0.0` | `5e-05` | `PASS` | `PASS` |
| `Mode1_b1.60` | `H0_BAO_naive` | `68.8330051913204` | `68.8330051913204` | `68.8330051913204` | `0.0` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode1_b1.60` | `H0_CMB_exact` | `75.13604149868428` | `75.13604149360135` | `75.13604149868428` | `5.083e-09` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode1_b1.60` | `theta100_error_ppm` | `0.0832304995630976` | `0.0832304995630976` | `0.0832304995630976` | `0.0` | `0.0` | `0.05` | `PASS` | `PASS` |
| `Mode1_b1.60` | `ns_compensated` | `0.930868120152704` | `0.930868547056143` | `0.930868120152704` | `4.269e-07` | `0.0` | `5e-06` | `PASS` | `PASS` |
| `Mode1_b1.60` | `As_compensated_1e9` | `1.9784020342521456` | `1.978401728694943` | `1.9784020342521456` | `3.056e-07` | `0.0` | `5e-05` | `PASS` | `PASS` |
| `Mode1_b1.60` | `rms_residual_naive_pct` | `16.148952870815318` | `16.148780923816567` | `16.148952870815318` | `1.719e-04` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode1_b1.60` | `rms_residual_theta_only_pct` | `8.92341482968813` | `8.923377130046427` | `8.92341482968813` | `3.770e-05` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode1_b1.60` | `rms_residual_final_pct` | `1.3310583017881097` | `1.3309960697300556` | `1.3310583017881097` | `6.223e-05` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode1_b1.60` | `max_residual_final_pct` | `3.2973152337228124` | `3.297352080766752` | `3.2973152337228124` | `3.685e-05` | `0.0` | `0.0005` | `PASS` | `PASS` |
| `Mode1_b1.60` | `est_B_nG` | `0.3485685011586675` | `0.3485685011586675` | `0.3485685011586675` | `0.0` | `0.0` | `5e-12` | `PASS` | `PASS` |

---

## 6. Residual Curve Analysis (`07_comparisons/CURVE_COMPARISON_CANDIDATE_B.csv`)

Across all 12 residual curve columns (`4 cases × 3 stages: naive, theta_only, final`) and all 2,471 multipoles $\ell \in [30, 2500]$:

| Curve Column | Debian R0 vs Hist Max `|Δ|` (%) | $\ell$ at Max `|Δ|` | Debian R0 vs Hist RMS `Δ` (%) | Debian R0 vs Hist $\ell=30$ `Δ` (%) | Debian R0 vs Hist $\ell\in[315,317]$ Max `|Δ|` (%) | Alpine R0-H vs Hist Max `|Δ|` (%) | Alpine R0-H vs Hist RMS `Δ` (%) |
|---|---|---|---|---|---|---|---|
| `Mode1_b1.40_naive_pct` | `0.002380` | `318` | `0.000432` | `-0.000107` | `0.002377` | `0.0` | `0.0` |
| `Mode1_b1.40_theta_only_pct` | `0.003281` | `315` | `0.000801` | `-0.002501` | `0.003281` | `0.0` | `0.0` |
| `Mode1_b1.40_final_pct` | `0.003411` | `315` | `0.000752` | **`-0.003004`** | `0.003411` | **`0.0`** | **`0.0`** |
| `Mode2_b1.35_naive_pct` | `0.002474` | `971` | `0.000668` | `-0.000058` | `0.002039` | `0.0` | `0.0` |
| `Mode2_b1.35_theta_only_pct` | `0.003169` | `317` | `0.000655` | `+0.000820` | `0.003169` | `0.0` | `0.0` |
| `Mode2_b1.35_final_pct` | `0.003132` | `317` | `0.000603` | **`+0.000977`** | `0.003132` | **`0.0`** | **`0.0`** |
| `Mode2_b1.40_naive_pct` | `0.004250` | `318` | `0.000691` | `+0.000290` | `0.004248` | `0.0` | `0.0` |
| `Mode2_b1.40_theta_only_pct` | `0.001970` | `316` | `0.000418` | `+0.000348` | `0.001970` | `0.0` | `0.0` |
| `Mode2_b1.40_final_pct` | `0.001957` | `316` | `0.000389` | `+0.000353` | `0.001957` | `0.0` | `0.0` |
| `Mode1_b1.60_naive_pct` | `0.003040` | `358` | `0.000605` | `+0.000207` | `0.001968` | `0.0` | `0.0` |
| `Mode1_b1.60_theta_only_pct` | `0.002183` | `317` | `0.000450` | `+0.000157` | `0.002183` | `0.0` | `0.0` |
| `Mode1_b1.60_final_pct` | `0.002132` | `317` | `0.000417` | `+0.000037` | `0.002132` | `0.0` | `0.0` |

### Root-Cause Mechanism of the Two Debian 12 `max_residual_final_pct` Discrepancies (`VERIFIED`)
1. In all 4 candidate cases, the peak absolute value of the final compensated residual curve (`max_residual_final_pct`) occurs at the low-$\ell$ boundary **$\ell = 30$** (`2.265528%` for `Mode1_b1.40`, `2.210705%` for `Mode2_b1.35`, `2.525798%` for `Mode2_b1.40`, `3.297315%` for `Mode1_b1.60`).
2. Between Debian 12 (`GCC 12.2.0` + `glibc 2.36 libm` + `numpy 1.24.2` + `OpenBLAS 0.3.21`) and Alpine 3.20 (`GCC 13.2.1` + `musl 1.2.5-r3 libm` + `numpy 1.25.2` + `OpenBLAS 0.3.28`), minor floating-point trajectory differences in CLASS (`-O3`) produce localized $C_\ell^{TT}$ differences of $\sim 0.002\text{--}0.0034\%$ around $\ell \in [315, 318]$ and a `2e-6` stdout rounding difference in `z_star` for `Mode1_b1.40` (`1113.854560` vs `1113.854562`).
3. Through the unweighted log-linear least-squares regression across $\ell \in [30, 2500]$, those small $C_\ell^{TT}$ shifts alter the fitted tilt `ns_compensated` by $+1.028 \times 10^{-6}$ (`Mode1_b1.40`) and $-0.956 \times 10^{-6}$ (`Mode2_b1.35`). Because $\ell = 30$ has the longest logarithmic lever arm from the pivot scale ($\ln(30/700) \approx -3.15$), the boundary residual at $\ell = 30$ shifts by `-0.003004%` (`Mode1_b1.40`) and `+0.000977%` (`Mode2_b1.35`), exceeding the tight `0.0005%` frozen tolerance in Debian 12 while vanishing identically (`0.0`) in Alpine 3.20.

---

## 7. Historical Binary Comparison (`06_instrumented_runs/PHASE_O_HISTORICAL_BINARY_MANIFEST.json`)

- **Direct Execution of `class_historical` (`SHA-256: a17d2197f33db122729e058dc3d4e9e15ef7c9efc554a408dfb96ca3d01d5150`)**:
  - Baseline (`b = 1.0, eta = 0.0, h = 0.674`) output files (`background.dat`, `thermodynamics.dat`, `cl.dat`) from `class_historical` are **100% byte-identical** to those produced by both the freshly compiled Candidate B binary (`class_candidate_b`) and the unmodified upstream CLASS `v3.2.0` binary (`class_unmodified_upstream`):
    - `background.dat`: `SHA-256: 625d97c331dda6f197220908c695a248e808275684590c82f457a8ebc38800ff`
    - `thermodynamics.dat`: `SHA-256: b2b4522f526ccbe1fd1c7e447ce4fd7ee6b2c64ccdb806f1ca51997fef135aac`
    - `cl.dat`: `SHA-256: 752054dcf732f45a8f12fb1158a8a70d8c2e4fb7bad9b1bfbb51dd049989d0ac`
  - Running the full 4-candidate v26.4 pipeline against `class_historical` in `06_instrumented_runs/historical_binary_full_test/` produces:
    - `hem_class_v26_4_exact_lock_results.csv`: `SHA-256: 81d840b89fd1888c306f53fd02a4ac9ef678eecc239039fa4e458f574f2f83d1` (exact match)
    - `hem_class_v26_4_residual_curves.csv`: `SHA-256: 50a9334e5f833eb0af793c130e349bf4472dbedc1b4403fb4ccad520ca806d2e` (exact match)
    - `hem_class_v26_4_summary.png`: `SHA-256: 4bc87d7841e940c3264ac04defec5703b19d648e882641d105f6db121272b607` (exact match)
- **Diagnostic Neutrino Track (`06_instrumented_runs/CONTROL_AND_DIAGNOSTIC_SUMMARY.json`, Non-Adjudicative)**:
  - The unmodified runner `run_colab_v26_4.py` omits explicit `N_ncdm` / `m_ncdm` lines in its `.ini` template, yielding baseline `z_* = 1088.74222` (`1088.7422`), `r_s = 144.707972 Mpc` (`144.7080 Mpc`), `100*theta_* = 1.03972813` (`1.039728`), which matches line 198 of the historical `task-125.log` (`基準線: z_* = 1088.7422, r_s = 144.7080 Mpc, 100*theta_* = 1.039728`) to every printed digit.
  - Adding explicit `N_ur = 2.046, N_ncdm = 1, m_ncdm = 0.06, T_ncdm = 0.7137658555036082` shifts the baseline to `z_* = 1088.7471, r_s = 144.6914 Mpc, 100*theta_* = 1.041195`, confirming (`VERIFIED`) that the historical v26.4 run executed the exact `.ini` template inside `run_colab_v26_4.py`.

---

## 8. Governance Recommendation

1. **R0-H Adjudication Status**: Record `R0-H` as **`PASS — HISTORICAL ENVIRONMENT RECONCILED`** (`52/52` fields `PASS` with `0.0` difference under the frozen tolerance contract `2736cfa87b667945bb6b75ea90b33bc9f7793968`).
2. **Interpretation of Prior Debian 12 R0 Clean-Room Result**: The 2/52 field failures in the Debian 12 R0 clean-room reproduction (`Mode1_b1.40` and `Mode2_b1.35` `max_residual_final_pct`, which missed the `0.0005%` tolerance by `0.0025%` and `0.00048%` at the $\ell = 30$ boundary) were caused purely by cross-environment OS/compiler/libc/library differences (`Debian 12 glibc 2.36 + GCC 12.2.0 + Python 3.11/NumPy 1.24` vs `Alpine 3.20.10 musl 1.2.5 + GCC 13.2.1 + Python 3.12/NumPy 1.25`), **not** by unrecorded source edits, hidden parameters, or manual post-processing in the historical v26.4 run.
3. **Environment Pinning for Future Reproducibility**: Any future numerical regression gate targeting sub-$10^{-3}\%$ point-wise $L_\infty$ tolerances on CLASS + Python post-processing outputs must pin the container/rootfs image (`Alpine 3.20.10` / `musl 1.2.5-r3` / `gcc 13.2.1_git20240309-r1` / `py3-numpy 1.25.2-r1`) or distinguish between environment-exact bitwise reproduction and cross-distro/cross-libc floating-point tolerance bands.
4. **Strict Separation of Numerical Reproducibility from Physical Validity**: `R0-H PASS` reconciles the historical v26.4 numerical environment only. It does **not** alter `R0 physical-model verdict: NOT APPLICABLE`, does **not** establish physical validity of HEM, and does **not** start `R1`–`R4`.

---

## 9. Evidence Package Info

- **Workspace Root (`R0H_ROOT`)**: `H:\AI_Products\AI_Factory_Workspace\TechOrb-Hubble-Validator-R0H-historical-reconciliation-20261021`
- **Master Checksum Manifest**: `09_manifests/SHA256SUMS.txt` (mirrored at `R0H_ROOT/SHA256SUMS.txt` and `10_report/SHA256SUMS.txt`)
- **Archive Path**: `H:\AI_Products\AI_Factory_Workspace\TechOrb-Hubble-Validator-R0H-historical-reconciliation-20261021\R0H_HISTORICAL_ENVIRONMENT_RECONCILIATION_2026-10-21.zip`
- **Archive Size (bytes)**: `183518797` (`183,518,797 bytes`)
- **Archive SHA-256**: `2bd2937a5bb6e8ea3cae8fc774dea29b2cdf907b27feaae80079c39cd68fbb48`
