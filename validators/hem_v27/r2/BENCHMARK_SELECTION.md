# R2 Benchmark Selection

Primary sources were rechecked on 2026-10-21. No remembered numerical value is used as evidence.

| Candidate | Identity and locator | Model / public implementation | Quantitative target | Cost and decision |
|---|---|---|---|---|
| Jedamzik & Pogosian, “Relieving the Hubble Tension with Primordial Magnetic Fields” (2020), PRL 125, 181302; DOI `10.1103/PhysRevLett.125.181302`; arXiv `2004.09487` v2 | M1/M2 three-zone models; Supplemental Tables I–II and Figs. 1–3 | M1/M2 equations are published. The later author fork identifies `recfast_3zones`, but the Letter does not provide machine-readable spectra/ionization histories or a fully frozen likelihood environment. License of the fork is not declared at repository level. | Posterior values and likelihood summaries exist, but reproducing them requires legacy Planck likelihoods and unprovided chains/configuration. | High relevance to v26.4 Mode1, but not the first target because the reference package is incomplete and the full fit is expensive. |
| Thiele et al., “Can small-scale baryon inhomogeneities resolve the Hubble tension? An investigation with ACT DR4” (2021), PRD 104, 063535; DOI `10.1103/PhysRevD.104.063535`; arXiv `2105.03003` | M1/M2 clumping constrained with ACT DR4/Planck | Published analysis uses the same model family; the Rashkovetskyi fork states compatibility. No paper-specific machine-readable benchmark outputs or exact executable bundle was located. | Tables/posteriors are likelihood products, not a compact deterministic recombination benchmark. | Valuable cross-check, but data/likelihood cost and reference-output gap make it secondary. |
| **Rashkovetskyi, Muñoz, Eisenstein & Dvorkin, “Small-scale clumping at recombination and the Hubble tension” (2021), PRD 104, 103517; DOI `10.1103/PhysRevD.104.103517`; arXiv `2108.02747` v3** | **Sec. II Eqs. (1)–(2); Fig. 2 caption; Fig. 3 caption; Appendix A.1 Fig. 18** | **M3 three-zone model. Author fork `misharash/class_public`, exact commit `4fdab292a223a0ab331e98613ad98e38a3a2d0de`, states it is CLASS 2.9 used for this paper and documents `recfast_3zones_lowlevel`. Repository-level license is not declared.** | **Fixed Planck-best-fit cosmology; `delta_m=-0.9`, `delta_p=5/3`, `f2V=f0=1/3`, hence `b=1`; published ionization/visibility curves and RECFAST-vs-HyRec relative CMB spectra.** | **Selected: closest deterministic published analogue to v26.4 Mode1, explicit equations and public source, no MHD simulation. Reference admission is complete through frozen independent vector/raster extraction; numerical execution has not occurred.** |
| Jedamzik, Abel & Ali-Haimoud, “Cosmic Recombination in the Presence of Primordial Magnetic Fields” (2024), arXiv `2312.11448` v3 | ENZO MHD + recombination + Monte-Carlo Lyman-alpha transport | Public ENZO is extended, but a complete public reference-output package sufficient for exact rerun was not identified. | Simulation figures and ionization histories; strong physical relevance. | Not first target: high MHD/transport cost and no compact public benchmark package. |

## Selection rationale

The selected M3 benchmark is deterministic, directly concerns baryon clumping during recombination, has exact equations and an author-linked implementation, runs in CLASS with RECFAST or HyRec, and avoids a new physical PMF/MHD model. It remains the best first R2 target. Official vector-figure bytes, independent reference extraction, masks and tolerances are now frozen; a separate numerical-reproduction work is authorized but has not run.

## Sources

- APS accepted manuscript and journal record for DOI `10.1103/PhysRevD.104.103517`.
- arXiv `2108.02747` v3.
- author repository `https://github.com/misharash/class_public` at commit `4fdab292a223a0ab331e98613ad98e38a3a2d0de`.
- APS/arXiv records for `2004.09487`, `2105.03003`, and `2312.11448`.
