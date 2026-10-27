"""Non-adjudicative provenance and source-structure helpers."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

FORBIDDEN_IMPORT_ROOTS = {"hem", "hubble", "pfm", "topology"}

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0].lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0].lower())
    return roots

def validate_static_separation(source_dir: Path) -> list[str]:
    errors: list[str] = []
    conditional = source_dir / "conditional_state.py"
    analytic = source_dir / "analytic_reference.py"
    if "analytic_reference" in imported_roots(conditional):
        errors.append("conditional path imports analytic reference")
    if "conditional_state" in imported_roots(analytic):
        errors.append("analytic path imports conditional path")
    for path in source_dir.glob("*.py"):
        leaked = imported_roots(path) & FORBIDDEN_IMPORT_ROOTS
        if leaked:
            errors.append(f"{path.name} imports forbidden roots: {sorted(leaked)}")
    return errors
