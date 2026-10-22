# HEM v27 R2 — Published Model Reproduction — NUMERICAL REPRODUCTION REPORT

**Execution date:** 2026-10-22 / 2026-10-23 (UTC+8)  
**Workspace (`R2_NUM_ROOT`):** `H:\AI_Products\AI_Factory_Workspace\TechOrb-Hubble-Validator-R2-numerical-reproduction-20261022`  
**Benchmark:** Rashkovetskyi, Muñoz, Eisenstein & Dvorkin (2021), PRD 104, 103517, DOI `10.1103/PhysRevD.104.103517`, arXiv `2108.02747v3` — M3 three-zone model.

```text
R2 = BLOCKED
Blocker: MANDATORY OBSERVABLE NOT LEGALLY EXTRACTABLE UNDER FROZEN SPEC
         (Fig.3 minus_zone, plus_zone, simple_clumping are not exposed by the pinned
          author binary and the frozen pre-registration defines no extraction procedure)
Pipeline numerical status: every admitted point of every curve that CAN be adjudicated
         (6/9 curves, 560/560 points) is within frozen tolerance; 0 FAIL.
HEM physical validity = NOT ESTABLISHED / NOT APPLICABLE
Tech-Orb = NOT VALIDATED
R3 = NOT AUTHORIZED
```

---

## A. Governance identity

| item | value |
|---|---|
| repository | `Terry-Kao/TechOrb-Hubble-Validator` |
| `origin/main` SHA at snapshot | `402b6e908986bcbdfc62f74df15f50001a08fab3` |
| R2 branch | `hem-v27-r2-published-model-reproduction` @ `e6a8a4f016254a1acd22f110f75fb46b017d26c7` |
| R2 pre-registration | `85821e5581d6464746e457e5af9f721dd17d0201` |
| original reference/tolerance freeze | `1f2ca42ec4f3b1a112108fa41f72103e323d1f44` |
| artifact identity re-freeze | `f9037a9a19d1591d25e3ea98acfa8ce6468bdb43` |
| `FIG2_REFERENCE.csv` | blob `6ac9cccff730f8110d9392d4ab75e3e81b5f3146`, 28,499 B, SHA-256 `a079416fdf59e040233f4fecc67f5d8f920ff038a654c8c7758a4c7aab051a05` — **re-verified at comparison time** |
| `FIG3_REFERENCE.csv` | blob `602dedb0d62246829b8d837232eb5dae33c3b807`, 62,797 B, SHA-256 `10a160e50d5f67b11d678ce7beb709b2c782846eb2f8fff9f39177782c32c719` — **re-verified at comparison time** |
| Tolerances (unchanged) | Fig.2 x_e linear 0.012; Fig.2 g linear 0.00025 Mpc⁻¹; Fig.3 log10(x_e) 0.045 dex |

Snapshot: `R2_EXECUTION_GOVERNANCE_SNAPSHOT.json` (taken before any numerical execution).

## B. Source

- Author repo `https://github.com/misharash/class_public`, exact commit `4fdab292a223a0ab331e98613ad98e38a3a2d0de` (CLASS v2.9.4 fork), tree `cebbcc1fc3133fb1d951d381e568fa1932a1aae4`.
- Pristine `git archive`: `01_source_admission/misharash_class_public_4fdab292a223.tar`, 7,813,120 B, SHA-256 `0efba26e2bfbdd6f689c796b3c711b589ccf63d128fca03c0cc70389282f1637`.
- Dirty paths: **0**.
- License: **UNKNOWN** (no repository license). Local inspection/execution only; **no author source bytes committed to GitHub**.

## C. Environment

- WSL2 kernel `6.6.87.2-microsoft-standard-WSL2`, x86_64; Alpine Linux 3.20.10 chroot `/run/linux_env` in `wsl -d docker-desktop`.
- gcc/g++ 13.2.1_git20240309, musl 1.2.5, GNU Make 4.4.1, Python 3.12.13, numpy 1.25.2, scipy 1.11.4, pandas 2.0.3, matplotlib 3.7.3, openblas 0.3.27.
- Build flags (author Makefile): `-O4 -ffast-math -fopenmp`, `-g -fPIC -DHYREC`.
- Runtime: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `LC_ALL=C`, `TZ=UTC`; network namespace isolated (`unshare -n`).
- Installed-DB digest `3cc664a5…1786f1`, rootfs tree digest `89819caa…a987` (`R2_ENVIRONMENT_LOCK.json`).
- Pristine binary: 3,202,256 B, SHA-256 `306713df0c2d6a5976f37871ca07e9943c2948ee9581c129701138587035e464`.
- Post-processing note: the WSL `/run` tmpfs was reset between the runs and the comparison step; the chroot was re-provisioned (`02_environment/reprovision_postprocessing/`) and its installed-DB digest, rootfs tree digest and package list all **MATCH** the Phase C lock. Only comparison/figure code ran in the re-provisioned chroot.

## D. Paper baseline

`base_2018_plikHM_TTTEEE_lowl_lowE_lensing.ini` from author commit (blob `9107122999cccb3bb7a82e616fcdc9333953203a`, 1,432 B, SHA-256 `223d409e…0a5fe`), used byte-unchanged plus frozen output/selector additions:

`H0=67.32117`, `omega_b=0.02238280`, `omega_cdm=0.1201075`, `N_ur=2.03066666667`, `N_ncdm=1`, `omega_ncdm=0.0006451439` (m≈0.0601 eV), `YHe=0.2454006` (explicit, no BBN lookup), `tau_reio=0.05430842` (reio_camb), `n_s=0.9660499`, `A_s=2.100549e-9`, halofit, `T_cmb=2.7255`; RECFAST defaults `recfast_z_initial=1e4`, `recfast_Nz0=20000`, fudge_H 1.14, delta_fudge_H −0.015, fudge_He 0.86. Full resolution: `PAPER_BASELINE_RESOLVED_CONFIG.json`.

M3: `recombination=recfast_3zones_lowlevel`, `delta_m=-0.9`, `delta_p=1.6666666666666667`, `f2V=0.3333333333333333` → b=1, f=(100/231, 1/3, 18/77), Δ=(0.1, 1, 8/3).  
Control: identical cosmology, `recombination=recfast`.

## E. Control (standard run1/run2)

All 8 output tables byte-identical (max abs/rel/RMS Δ = 0; first mismatch NONE). Exit codes 0, stderr empty. See `REPEATABILITY_REPORT.md`.

## F. M3 (run1/run2)

All 8 output tables byte-identical (max abs/rel/RMS Δ = 0; first mismatch NONE). Exit codes 0, stderr empty.

## G. Fig. 2 (linear space; interpolation `numpy.interp` linear-in-z at fixed reference z, no extrapolation; every reference z is an exact native grid node, Δz-grid = 0.5)

| curve | N total | N admitted | N fail | max |Δ| (z) | tolerance | verdict |
|---|---|---|---|---|---|---|
| xe_standard | 81 | 77 | 0 | 3.850e-4 (1330) | 0.012 | **PASS** |
| xe_m3 | 81 | 81 | 0 | 1.274e-3 (1370) | 0.012 | **PASS** |
| g_standard | 81 | 78 | 0 | 4.634e-5 Mpc⁻¹ (1180) | 2.5e-4 | **PASS** |
| g_m3 | 81 | 81 | 0 | 5.844e-5 Mpc⁻¹ (1210) | 2.5e-4 | **PASS** |

## H. Fig. 3 (log10 space)

| curve | extraction track | N total | N admitted | N fail | max |Δ| dex (z) | tol | verdict |
|---|---|---|---|---|---|---|---|
| M3 | unmodified binary (averaged x_e) | 126 | 126 | 0 | 1.760e-3 (940) | 0.045 | **PASS** |
| zero_zone_LCDM | unmodified binary (`recfast`, Δ=1) | 126 | 117 | 0 | 2.819e-3 (930) | 0.045 | **PASS** |
| minus_zone | read-only diagnostic dump (non-pre-registered) | 126 | 97 | 0 | 9.216e-4 (970) | 0.045 | PASS — **DIAGNOSTIC ONLY, NOT ADJUDICATIVE** |
| plus_zone | read-only diagnostic dump (non-pre-registered) | 126 | 111 | 0 | 1.312e-3 (930) | 0.045 | PASS — **DIAGNOSTIC ONLY, NOT ADJUDICATIVE** |
| simple_clumping | diagnostic extra RECFAST run, Δ=1+b=2 (non-pre-registered) | 126 | 126 | 0 | 3.116e-3 (930) | 0.045 | PASS — **DIAGNOSTIC ONLY, NOT ADJUDICATIVE** |

Excluded rows retained with frozen exclusion reasons. Pointwise CSVs: `FIG2_POINTWISE_COMPARISON.csv`, `FIG3_POINTWISE_COMPARISON.csv`; summary `R2_COMPARISON_SUMMARY.csv`. Figures in `11_figures/` are diagnostic only.

## I. Anomalies (all)

1. **[VERIFIED] Mandatory Fig.3 zone curves not exposed by pinned author code.** In `source/thermodynamics.c` (`thermodynamics_recombination_3zones`) at `4fdab292`, `reco1` (Δ=0.1, minus zone) and `reco3` (Δ=8/3, plus zone) are computed, mass-weighted into the averaged table and immediately `free()`d without output (an earlier author commit `f5401e42` dumped them; `9ab87673` removed the dump).
2. **[VERIFIED] `simple_clumping` (Δ=1+b) is not an `.ini`-selectable mode** at `4fdab292` (`recfast` hard-codes Δ=1; all zone modes enforce ⟨Δ⟩=1).
3. **[VERIFIED] Frozen pre-registration defines no extraction procedure** for these curves (`BENCHMARK_SPEC.md` lists "zone and averaged x_e(z)" as observables but no extraction route; `EQUATION_TRANSCRIPTION.md` states only that the code runs three RECFAST histories). Under spec §19/§35 this is "mandatory observable cannot be legally extracted" → BLOCKED.
4. **[VERIFIED] A read-only diagnostic build was created** (`03_build/diagnostic_zone_dump/`, patch `DIAGNOSTIC_EXPORT_PATCH.diff`, binary SHA-256 `4bb94403…72f5`). It writes the already-computed zone tables plus one extra RECFAST call with Δ=2 to a side file, behind an environment variable. All 8 standard outputs are byte-identical to the unmodified M3 run. Per §19 it is **classified NON-ADJUDICATIVE**. Note: the `simple_clumping` definition (RECFAST with Δ=1+b) is the executor's reading of the paper, not a frozen definition — **SUPPORTED**, not VERIFIED.
5. **[POSSIBLE] Neutrino-treatment text discrepancy.** `BENCHMARK_SPEC.md` says "The paper uses massless neutrinos", but the frozen, hash-admitted author `.ini` specifies `N_ncdm=1`, `omega_ncdm=0.0006451439` (one 0.06 eV massive species, N_eff=3.044). The frozen `.ini` was used unchanged as required (no cosmology changes). Unresolved; recorded for governance review. It did not prevent agreement within tolerance.
6. **[VERIFIED] WSL `/run` reset** between execution and comparison; chroot re-provisioned with matching digests (see §C). Scientific runs were not repeated.
7. **[VERIFIED] Process note.** The diagnostic build was performed before the §19/§35 stop rule was fully applied; strict reading of §35 would have stopped at blocker detection. Its outputs are preserved but carry no adjudicative weight.
8. Compile warnings in author code (prototype notes in `perturbations.h`) — benign, present in pristine build as well.

## J. Verdict

```text
R2 = BLOCKED
```

Reason: Gate §24 PASS requires *all mandatory admitted reference points* to pass. Three mandatory Fig.3 curves (minus_zone, plus_zone, simple_clumping; 334 admitted points) cannot be obtained from the pinned unmodified implementation by any frozen procedure. The non-pre-registered diagnostic dump cannot become adjudicative evidence (§19). No FAIL exists among the 560 adjudicable admitted points (6 curves), and the diagnostic track also shows 0 FAIL in 334 points, but neither fact converts BLOCKED into PASS.

Unblocking requires a governance decision (outside this work) to pre-register — before any further execution — an extraction procedure for zone histories and a definition of `simple_clumping` (e.g. formally admitting the read-only dump patch), then re-adjudication. No tolerance, mask, backend, cosmology or b change is proposed.

## K. Scientific boundary

> R2 tests reproduction of a selected published recombination/clumping benchmark. It does not establish observational preference for HEM, solve the Hubble tension, derive a primordial magnetic field, or validate the Tech-Orb mother hypothesis.
