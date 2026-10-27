"""Non-adjudicative provenance and source-structure helpers."""
from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import subprocess

from . import SCIENTIFIC_EXECUTION_AUTHORIZED
from .conditional_state import interacting_constraint, near_kernel_projector_continuation
from .controls import CONTROL_IMPLEMENTATIONS, ControlId
from .protocol_binding import FROZEN_STRUCTURE, FROZEN_THRESHOLDS

FORBIDDEN_IMPORT_ROOTS = {"hem", "hubble", "pfm", "topology"}

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def committed_blob_bytes(repo_root: Path, commit: str, relative_path: str) -> bytes:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("canonical path must be repository-relative and traversal-free")
    return subprocess.check_output(["git", "show", f"{commit}:{relative_path}"], cwd=repo_root)

def canonical_committed_identity(repo_root: Path, commit: str, relative_path: str) -> dict[str, object]:
    data = committed_blob_bytes(repo_root, commit, relative_path)
    blob = subprocess.check_output(["git", "rev-parse", f"{commit}:{relative_path}"], cwd=repo_root, text=True).strip()
    return {"path": relative_path.replace("\\", "/"), "byte_length": len(data), "git_blob": blob, "sha256": hashlib.sha256(data).hexdigest()}

def verify_canonical_manifest(repo_root: Path, manifest: dict[str, object]) -> list[str]:
    return [item["path"] for item in manifest["files"] if canonical_committed_identity(repo_root, str(manifest["source_baseline_commit"]), item["path"]) != item]

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
    continuation_parameters = tuple(inspect.signature(near_kernel_projector_continuation).parameters)
    if continuation_parameters != ("operator", "state"):
        errors.append("scientific continuation API exposes non-frozen arguments")
    interaction_parameters = tuple(inspect.signature(interacting_constraint).parameters)
    if interaction_parameters != ("k0", "g"):
        errors.append("interaction API exposes arbitrary interaction input")
    if set(CONTROL_IMPLEMENTATIONS) != set(ControlId) or len(CONTROL_IMPLEMENTATIONS) != 6:
        errors.append("exact six control implementations are not structurally bound")
    forbidden_names = {"inverse_scramble", "unscramble_clock_labels", "set_thresholds", "set_spectral_window"}
    exported_names = set()
    for path in source_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        exported_names.update(node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = {arg.arg.lower() for arg in node.args.args}
                if path.name == "conditional_state.py" and ({"trajectory", "external_time", "exclusions", "thresholds"} & args):
                    errors.append(f"{node.name} exposes forbidden leakage/mutation input")
    leaked_names = forbidden_names & exported_names
    if leaked_names:
        errors.append(f"forbidden recovery/mutation helpers exist: {sorted(leaked_names)}")
    if len(FROZEN_THRESHOLDS) != 22:
        errors.append("full frozen threshold inventory is incomplete")
    required_structure = {"D_C", "D_S", "OMEGA", "HBAR", "RESOLUTION_D_C", "INTERACTION_G", "SPECTRAL_WINDOW", "SEEDS", "CLOCK_B_SHIFT", "CLOCK_SCRAMBLE"}
    if not required_structure.issubset(FROZEN_STRUCTURE):
        errors.append("frozen structural parameter inventory is incomplete")
    if SCIENTIFIC_EXECUTION_AUTHORIZED:
        errors.append("scientific execution lock is not active")
    runner = (source_dir / "runner.py").read_text(encoding="utf-8") if (source_dir / "runner.py").exists() else ""
    adjudication = (source_dir / "adjudication.py").read_text(encoding="utf-8") if (source_dir / "adjudication.py").exists() else ""
    verdict = (source_dir / "verdict.py").read_text(encoding="utf-8") if (source_dir / "verdict.py").exists() else ""
    for forbidden in ("--force", "threshold_override", "tau_override", "control_plugins", "exclude_outcome"):
        if forbidden in runner + adjudication + verdict:
            errors.append(f"forbidden runner/adjudication bypass token: {forbidden}")
    verdict_module = __import__("qf0a.verdict", fromlist=["derive_machine_verdict"])
    if "pipeline" in inspect.signature(verdict_module.derive_machine_verdict).parameters:
        errors.append("scientific verdict accepts caller pipeline verdict")
    return errors
