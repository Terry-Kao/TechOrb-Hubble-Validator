# Historical Binary Forensic Examination (`HISTORICAL_BINARY_FORENSICS.md`)

- **Phase**: `PHASE A`
- **Target Artifact**: Historical compiled `class` executable from `v26.4` workspace
- **Inspection Method**: Strictly read-only (`Get-Item`, `Get-FileHash`, `file`, `readelf`, `objdump`, `strings`) performed on isolated copy `01_binary_forensics/class_historical` prior to any execution.

---

## 1. File Identity & Cryptographic Hashes (`VERIFIED`)

| Property | Value | Evidence Status |
| :--- | :--- | :--- |
| **Original Full Path** | `H:\AI_Products\AI_Factory_Workspace\TechOrb-Hubble-Validator\HRS_Research\Colab\class_hia_v26\class` | `VERIFIED` |
| **Forensic Copy Path** | `H:\AI_Products\AI_Factory_Workspace\TechOrb-Hubble-Validator-R0H-historical-reconciliation-20261021\01_binary_forensics\class_historical` | `VERIFIED` |
| **File Size** | `3,854,200` bytes (`3.68 MiB`) | `VERIFIED` |
| **Created Timestamp (NTFS)** | `2026-10-19T07:20:13.5529608+08:00` | `VERIFIED` |
| **Modified Timestamp (NTFS)** | `2026-10-19T07:20:14.4062412+08:00` | `VERIFIED` |
| **Full SHA-256** | `a17d2197f33db122729e058dc3d4e9e15ef7c9efc554a408dfb96ca3d01d5150` | `VERIFIED` |
| **Full SHA-1** | `e9a75a10f3e1cebf65f34a68dde13e8210bc102c` | `VERIFIED` |
| **GNU Build-ID (`NT_GNU_BUILD_ID`)** | `4d8e404396dcf56022069af2dc22344f0c9b22ae` | `VERIFIED` |
| **Debian R0 Clean-room Binary SHA-256 (for contrast)** | `c4adc6aeeea1428b26b5fcebba2ea4d1db1ca50d2e3cc960faf1d8e7bac18f30` (`3,588,944` bytes) | `VERIFIED` |

---

## 2. ELF Header & Program Structure (`VERIFIED`)

From `file`, `readelf -h`, `readelf -l`, and `objdump -f -p` (logged in [`historical_binary_readelf.txt`](historical_binary_readelf.txt)):

- **`file` output**:
  `ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib/ld-musl-x86_64.so.1, BuildID[sha1]=4d8e404396dcf56022069af2dc22344f0c9b22ae, with debug_info, not stripped`
- **ELF Class**: `ELF64` (`2's complement, little endian`)
- **Architecture / ISA**: `Advanced Micro Devices X86-64`, `.note.gnu.property` specifies `x86 feature used: x86, XMM`, `x86 ISA used: x86-64-baseline`
- **ELF Type**: `DYN (Position-Independent Executable file)` with `FLAGS_1: NOW PIE` (`BIND_NOW`)
- **Entry Point**: `0x29d4`
- **Debug Sections**: Present and unstripped (`.debug_aranges`, `.debug_info`, `.debug_abbrev`, `.debug_line`, `.debug_frame`, `.debug_str`, `.debug_line_str`, `.debug_loclists`, `.debug_rnglists`, `.symtab`, `.strtab`)

---

## 3. Toolchain, C Library & Dynamic Linking Evidence (`VERIFIED`)

| Component | Extracted Binary Evidence | Evidence Status |
| :--- | :--- | :--- |
| **Program Interpreter (`PT_INTERP`)** | `/lib/ld-musl-x86_64.so.1` | `VERIFIED` |
| **C Standard & Math Library (`DT_NEEDED`)** | `libc.musl-x86_64.so.1` (in `musl`, `libm` is integrated directly into `libc.musl-x86_64.so.1`) | `VERIFIED` |
| **musl Version (DWARF CRT paths)** | `/home/buildozer/aports/main/musl/src/musl-1.2.5` (strings lines 14497, 18022) | `VERIFIED` |
| **OpenMP Runtime (`DT_NEEDED`)** | `libgomp.so.1` (`GOMP_parallel`, `GOMP_loop_nonmonotonic_dynamic_start`, etc.) | `VERIFIED` |
| **Compiler (`.comment` section)** | `GCC: (Alpine 13.2.1_git20240309) 13.2.1 20240309` | `VERIFIED` |
| **Compiler Target Triple (DWARF)** | `/usr/lib/gcc/x86_64-alpine-linux-musl/13.2.1/include` (strings line 18085) | `VERIFIED` |
| **Fortran / BLAS / LAPACK in `class`** | None (`libgfortran`, `libblas`, `liblapack` are absent from `DT_NEEDED`) | `VERIFIED` |

---

## 4. Embedded Build Path, CLASS Version & v26.4 Patch Markers (`VERIFIED`)

From [`historical_binary_strings.txt`](historical_binary_strings.txt):

1. **CLASS Version**:
   - Line 9346: `v3.2.0` (`VERIFIED`)
2. **Embedded v26.4 Environment Variable Strings**:
   - Line 10658: `HIA_B_CLUMPING` (`VERIFIED`)
   - Line 10659: `HIA_LYA_EXP` (`VERIFIED`)
   - *Forensic significance*: Earlier versions (`v26.2`) only had `HIA_B_CLUMPING`, and `v26.3` used `HIA_LYA_ETA` (and built in `class_hia_v26_3`). The presence of `HIA_LYA_EXP` proves beyond doubt that `class_hia_v26/class` was compiled specifically from the `v26.4` patched `wrap_recfast.c`.
3. **Embedded Build Directory (`__CLASSDIR__` and DWARF `DW_AT_comp_dir`)**:
   - Line 10925: `/mnt/host/h/AI_Products/AI_Factory_Workspace/TechOrb-Hubble-Validator/HRS_Research/Colab/class_hia_v26`
   - Line 18026: `/mnt/host/h/AI_Products/AI_Factory_Workspace/TechOrb-Hubble-Validator/HRS_Research/Colab/class_hia_v26/build`
   - *Forensic significance*: The `/mnt/host/h/...` mount path is the signature mount point of the Windows WSL2 `docker-desktop` distribution (`/mnt/host/h` rather than standard WSL Ubuntu's `/mnt/h`), confirming that this binary was compiled inside WSL2 `docker-desktop` (`Alpine Linux`) mounted over `H:\AI_Products\AI_Factory_Workspace`.
4. **Python Strings in Binary**:
   - Lines 10596–10597, 12772–12777: Standard CLASS `external/HyRec2020` / `distortions` fallback strings (`python --version`, `The command 'python %s' failed.`). The `class` C binary itself does not link against `libpython`.

---

## 5. Chronological & Provenance Assessment

- **Source-to-Binary-to-Output Timeline (Local Time `+08:00` on `2026-10-19`)**:
  1. `07:19:37.833`: `run_wsl_fallback.sh` updated to invoke `python3 run_colab_v26_4.py`.
  2. `07:19:55.279`: `run_colab_v26_4.py` finalized (`SHA-256: 260bbe86...`).
  3. `07:20:07.259`: `class_hia_v26/external/RecfastCLASS/wrap_recfast.c` patched with `/* HIA_V26_DUAL_PATCH */` (`SHA-256: ab460fc6...`).
  4. `07:20:14.406`: `class_hia_v26/class` binary linked (`SHA-256: a17d2197...`).
  5. `07:23:24.000`: `hem_class_v26_4_exact_lock_results.csv`, `hem_class_v26_4_residual_curves.csv`, `hem_class_v26_4_summary.png`, and `hem_class_v26_4_bundle.zip` written at the completion of the 41-run pipeline (~3 minutes 10 seconds after compilation).
- **Evidence Classification**:
  - Binary identity, Alpine 3.20 / `musl 1.2.5` / `GCC 13.2.1_git20240309` toolchain identity, and `HIA_LYA_EXP` patch presence: **`VERIFIED`**.
  - Status of `class_hia_v26/class` as the historical production executable that generated the v26.4 CSVs: **`STRONG CANDIDATE`** (supported by exact embedded path `/mnt/host/h/...`, `HIA_LYA_EXP` symbol, and 7-second / 190-second timestamp chain; not marked `VERIFIED` in isolation only because historical `run_colab_v26_4.py` did not cryptographically hash the binary inside a run manifest at runtime).
