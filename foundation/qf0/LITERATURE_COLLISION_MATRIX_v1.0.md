# Tech-Orb QF Literature & Collision Matrix v1.0

**Status:** skeleton only; no large-scale search or literature verdict completed  
**Machine structure:** `LITERATURE_COLLISION_MATRIX_v1.0.json`  
**Created:** 2026-10-26

## Entry rule

One entry represents one primary source/version. It may record only:

- what the work actually proves;
- its assumptions and regime;
- related Tech-Orb claim IDs;
- collision relation: `COMPATIBLE`, `CONFLICT`, or `IRRELEVANT`;
- what it does **not** prove;
- publication status: `published`, `accepted`, or `preprint`;
- primary-source URL and DOI/arXiv identifiers;
- the date the primary source and publication status were checked.

`COMPATIBLE` never means supported, confirmed, equivalent, or novel. Similar wording is not evidence. Review articles may aid discovery but cannot replace the primary source. A newer version must be a new/reviewed entry or an explicit supersession; silent citation drift is prohibited.

## Priority domains

| Domain ID | Domain | Collision focus |
|---|---|---|
| LIT-POT | Problem of time | Constraints, timeless formulations, and the limits of claims about external time |
| LIT-PW | Page–Wootters | Conditional states, clock quality/interactions, and QF-0A reproduction targets |
| LIT-QRF | Quantum reference frames | Operational reference changes and prediction consistency |
| LIT-EG | Entanglement / geometry | Exact regime, dictionary, and non-generalizability of information–geometry results |
| LIT-CE | Causal emergence | Lorentzian order/signature, locality, null structure, and reconstruction limits |
| LIT-LR | Lorentz recovery | Preferred frames, emergent symmetry, universality, and precision constraints |
| LIT-DR | Decoherence / redundant records / Quantum Darwinism | Robustness, redundancy, accessibility, objectivity limits, and false positives |

## Human entry template

### LIT-<DOMAIN>-<NNN> — <short source name>

- **Citation:**
- **Publication status:** published / accepted / preprint
- **Primary source:**
- **DOI:**
- **arXiv:**
- **Checked at:** YYYY-MM-DD
- **What it actually proves:**
- **Assumptions/regime:**
- **Related claim IDs:**
- **Relation:** COMPATIBLE / CONFLICT / IRRELEVANT
- **What it does not prove:**

## Governance

The matrix can identify prior art, incompatibility, missing assumptions, or a narrower residual claim. It cannot promote a registry claim by authority, citation count, analogy, or vocabulary overlap. Claim changes require the registry, evidence, and Decision Log to be updated through review.
