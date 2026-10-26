"""Structural validation only for the QF-0 Foundation initialization package.

This script parses governance artifacts and checks identifiers, references, and
required protocol sections. It performs no scientific or numerical execution.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QF0 = ROOT / "foundation" / "qf0"

ALLOWED_LABELS = {
    "ESTABLISHED",
    "COMPATIBLE",
    "HYPOTHESIS",
    "ANSATZ",
    "REJECTED-LITERAL-VERSION",
}

ALLOWED_VALIDATORS = {
    "Quantum",
    "Reference",
    "Geometry",
    "Dimension",
    "Causality",
    "GR",
    "Classicality",
    "Information",
    "Predictivity",
    "Validator Leakage",
    "Analogy",
    "Novelty",
}

CLAIM_FIELDS = {
    "claim_id",
    "statement",
    "label",
    "mathematical_definition",
    "dependencies",
    "supporting_evidence",
    "conflicting_evidence",
    "falsification_condition",
    "validator",
    "current_verdict",
    "last_reviewed",
}

REQUIRED_CLAIMS = {
    "QF-FND-001",
    "QF-TIME-001",
    "QF-DIST-001",
    "QF-DIM-001",
    "QF-CAUS-001",
    "QF-GR-001",
    "QF-CLASS-001",
    "QF-TAIJI-001",
    "QF-BHWH-001",
    "QF-REJ-001",
    "QF-REJ-002",
}

PROTOCOL_MARKERS = {
    "H_C": "### 3.1 Clock space and Hamiltonian",
    "H_S": "### 3.2 System space and Hamiltonian",
    "factorization": "## 2. Factorization and ansatz boundary",
    "constraint": "### 3.3 Global constraint and physical state",
    "clock POVM": "### 3.1 Clock space and Hamiltonian",
    "conditional state": "### 3.4 Conditional state",
    "analytic target": "## 4. Analytic reference target",
    "numerical plan": "## 5. Numerical reproduction plan",
    "normalization": "## 6. Normalization, positivity, and constraint checks",
    "unitarity": "## 7. Approximate unitarity checks",
    "finite clock": "## 8. Finite/imperfect clock error metric",
    "interaction": "## 9. Interaction sensitivity",
    "two clocks": "## 10. Two clock-choice comparison",
    "seeds": "## 11. Randomness and deterministic settings",
    "thresholds": "1e-10",
    "negative controls": "## 12. Negative controls",
    "fail gates": "## 13. Explicit FAIL gates",
    "definition of done": "## 14. Definition of Done",
    "no new physics": "NO NEW-PHYSICS CLAIM",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    errors: list[str] = []

    registry_path = QF0 / "CLAIM_REGISTRY_v1.0.json"
    registry = load_json(registry_path)
    require(set(registry["allowed_labels"]) == ALLOWED_LABELS,
            "Registry label vocabulary differs from the frozen five-label set", errors)

    claims = registry.get("claims", [])
    ids = [claim.get("claim_id") for claim in claims]
    require(len(ids) == len(set(ids)), "Duplicate claim_id", errors)
    require(REQUIRED_CLAIMS.issubset(set(ids)), "Required core claims missing", errors)

    for claim in claims:
        claim_id = claim.get("claim_id", "<missing>")
        require(CLAIM_FIELDS.issubset(claim), f"{claim_id}: required field missing", errors)
        require(claim.get("label") in ALLOWED_LABELS,
                f"{claim_id}: invalid label {claim.get('label')!r}", errors)
        require(bool(claim.get("falsification_condition")),
                f"{claim_id}: empty falsification condition", errors)
        require(bool(claim.get("validator")), f"{claim_id}: no validator", errors)
        require(set(claim.get("validator", [])).issubset(ALLOWED_VALIDATORS),
                f"{claim_id}: unknown validator", errors)
        for dependency in claim.get("dependencies", []):
            require(dependency in ids, f"{claim_id}: broken dependency {dependency}", errors)

    human_registry = (QF0 / "CLAIM_REGISTRY_v1.0.md").read_text(encoding="utf-8")
    for claim_id in ids:
        require(claim_id in human_registry,
                f"{claim_id}: absent from human-readable registry", errors)

    canon = (QF0 / "FOUNDATION_CANON_v1.0.md").read_text(encoding="utf-8")
    for label in ALLOWED_LABELS:
        require(f"`{label}`" in canon, f"Canon missing label {label}", errors)
    for gate in ALLOWED_VALIDATORS:
        require(f"| {gate} |" in canon, f"Canon missing global gate {gate}", errors)

    protocol = (QF0 / "QF0A_FROZEN_PROTOCOL_v1.0.md").read_text(encoding="utf-8")
    for name, marker in PROTOCOL_MARKERS.items():
        require(marker in protocol, f"QF-0A protocol missing {name}", errors)
    for explicit_fail in (
        "secretly adding classical external/meta-time",
        "Conditional state normalization is unstable",
        "depend wholly on arbitrary clock choice",
        "changed after viewing results",
    ):
        require(explicit_fail in protocol,
                f"QF-0A missing minimum FAIL gate: {explicit_fail}", errors)

    benchmark = (QF0 / "QF0BC_BENCHMARK_SPEC_SKELETON_v1.0.md").read_text(encoding="utf-8")
    for marker in (
        "1D line", "1D ring", "2D square/rectangular lattice",
        "Monotone transform candidate", "Diffusion/resistance candidate",
        "degree-preserving edge shuffle", "intrinsic dimensions `1`, `2`, and `3`",
        "Spectral/diffusion estimator", "Local scaling estimator",
        "Hidden generator", "Reconstruction pipeline",
    ):
        require(marker in benchmark, f"QF-0B/C skeleton missing {marker}", errors)

    literature = load_json(QF0 / "LITERATURE_COLLISION_MATRIX_v1.0.json")
    require(set(literature["allowed_relations"]) == {"COMPATIBLE", "CONFLICT", "IRRELEVANT"},
            "Literature relation vocabulary mismatch", errors)
    require(set(literature["allowed_publication_status"]) == {"published", "accepted", "preprint"},
            "Publication-status vocabulary mismatch", errors)
    require(len(literature.get("domains", [])) == 7, "Expected seven priority literature domains", errors)
    require(literature.get("entries") == [], "Skeleton unexpectedly contains reviewed literature entries", errors)

    # Check relative Markdown links in the added package and changed root/control documents.
    markdown_paths = list(QF0.glob("*.md")) + [
        ROOT / "README.md",
        ROOT / "control" / "PROJECT_STATE.md",
        ROOT / "control" / "SCIENTIFIC_GOVERNANCE.md",
        ROOT / "control" / "DECISION_LOG.md",
        ROOT / "control" / "EVIDENCE_INDEX.md",
    ]
    link_pattern = re.compile(r"\[[^\]]+\]\((?!https?://|#)([^)]+)\)")
    for path in markdown_paths:
        if not path.exists():
            errors.append(f"Expected governance file missing: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        for target in link_pattern.findall(text):
            clean_target = target.split("#", 1)[0]
            if not clean_target:
                continue
            resolved = (path.parent / clean_target).resolve()
            require(resolved.exists(),
                    f"Broken link in {path.relative_to(ROOT)}: {target}", errors)

    if errors:
        print("QF-0 Foundation package validation: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("QF-0 Foundation package validation: PASS")
    print(f"- claims: {len(claims)}")
    print(f"- claim dependencies: {sum(len(c['dependencies']) for c in claims)}")
    print(f"- global gates: {len(ALLOWED_VALIDATORS)}")
    print(f"- literature domains: {len(literature['domains'])}")
    print("- scientific/numerical execution: NOT PERFORMED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
