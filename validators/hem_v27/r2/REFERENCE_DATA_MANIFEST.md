# Reference Data Manifest

| ID | Source | Locator | Intended bytes | Status |
|---|---|---|---|---|
| PAPER-01 | APS accepted manuscript, DOI `10.1103/PhysRevD.104.103517` | PDF, 21 pages | metadata and SHA256 in H: only; do not commit full paper | Identified, bytes not yet admitted |
| PAPER-02 | arXiv `2108.02747` v3 | source/PDF | version metadata and SHA256 in H: only | Identified, bytes not yet admitted |
| CODE-01 | `misharash/class_public` | commit `4fdab292a223a0ab331e98613ad98e38a3a2d0de` | pristine Git archive and tree/hash records in H: | Identified; license UNKNOWN |
| REF-01 | Paper Fig. 2 | `x_e(z)` and `g(z)` standard vs M3 | `reference/fig2_points.csv` plus extraction metadata | **MISSING / BLOCKER** |
| REF-02 | Paper Fig. 3 | zone/M3/simple-clumping `x_e(z)` | `reference/fig3_points.csv` plus extraction metadata | **MISSING / BLOCKER** |
| REF-03 | Paper Fig. 18 | RECFAST/HyRec TT/EE residuals | optional digitized CSV | Deferred |

## Digitization admission rule

Prefer author-supplied raw plotting data. Otherwise use the highest-resolution official/arXiv vector PDF, record URL, retrieval timestamp, PDF SHA256, page/figure, axis calibration points, curve/color identity, digitizer name/version, extraction settings, point CSV, and a second independent calibration check. The committed repository may contain derived point CSVs and metadata but not the full copyrighted paper.

No local model result may be opened before REF-01 and REF-02 are frozen with hashes. Their absence blocks numerical execution.
