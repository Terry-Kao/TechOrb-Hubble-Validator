# R2 Reference Data Manifest — PRE-EXECUTION FREEZE

Frozen on 2026-10-21 before any R2 numerical execution. No local CLASS/HyRec reproduction output was generated or inspected.

| ID | Primary source / derived asset | Bytes | SHA-256 | Provenance | Adjudicative use |
|---|---|---:|---|---|---|
| PAPER-ARXIV-PDF | arXiv `2108.02747v3` PDF | 2,224,808 | `27673ee09234c8bcdb9dc18a41d0a91d8295ccd760a60834c4a6c0dedeb34272` | VERIFIED official arXiv | Text/figure cross-check; H: only |
| PAPER-ARXIV-SRC | arXiv `2108.02747v3` source tar | 4,662,972 | `1c6d917257c4d209981191f8fb2e2a49335950eab5d7efc3f900b5b8657895fb` | VERIFIED official arXiv | Author-submitted figure assets; H: only |
| PAPER-APS-AM | APS/CHORUS accepted manuscript | 2,031,834 | `633f98522903079dada24715cee248815f5f6277c7788a9d319211a01c095652` | VERIFIED journal-controlled | Publication cross-check; H: only |
| FIG2-PDF | `img/M3demo-rec.pdf` from arXiv source | 193,789 | `5615bb21b212eca8f641d7833a289e0ba3c96db531ccd54ab8a1f5a465206b47` | AUTHOR-CONTROLLED | Vector geometry source; H: only |
| FIG3-PDF | `img/M3demo-rec-details.pdf` from arXiv source | 194,399 | `e1a29d0fe80c82672ddb7397ee89d0f33230e4115c083aba5e67e2645bc7b8d4` | AUTHOR-CONTROLLED | Vector geometry source; H: only |
| FIG18-PDF | `img/M3demo-reccodes.pdf` from arXiv source | 204,619 | `be199bc417b1ec9eacae0bec4f31902ff7b47e333d728aa70287b7979abd9a44` | AUTHOR-CONTROLLED | Preserved; not mandatory in this freeze |
| REF-FIG2 | committed `FIG2_REFERENCE.csv` bytes | 28,499 | `a079416fdf59e040233f4fecc67f5d8f920ff038a654c8c7758a4c7aab051a05` | DERIVED from FIG2-PDF | Mandatory reference with row masks |
| REF-FIG3 | committed `FIG3_REFERENCE.csv` bytes | 62,797 | `10a160e50d5f67b11d678ce7beb709b2c782846eb2f8fff9f39177782c32c719` | DERIVED from FIG3-PDF | Mandatory reference with row masks |

## Recovery result

No author-supplied machine-readable `x_e(z)` or `g(z)` raw table, original plotting script, or complete plotting command was found. The official arXiv v3 source does, however, contain standalone Matplotlib vector PDFs for Figs. 2, 3 and 18. The PDF object inventory shows no raster image objects in those figure files, so their plotted paths, strokes, dash patterns, axes and transforms are preserved as vectors.

The full visible history, branches, tags, releases, deleted-file names and relevant data/notebook extensions of `misharash/class_public` were searched. The author-linked `cobaya_mock_cmb`, `cobaya_H0_ext`, and thesis repositories were also inspected. `cobaya_mock_cmb` contains author-controlled mock CMB spectra and generation code, but its clumping fiducial is a different parameter point and is not a Fig. 2/3/18 raw-data package. No release asset, LFS pointer, submodule, or author repository supplied the missing plotted arrays.

## Storage and redistribution boundary

Copyrighted paper/figure bytes and author source repositories remain only in the H: evidence workspace. The repository commits derived numeric coordinates, method records, hashes and provenance. These are not represented as author-supplied raw data.

H: workspace: `H:\AI_Products\AI_Factory_Workspace\TechOrb-Hubble-Validator-R2-reference-recovery-20261021`.
