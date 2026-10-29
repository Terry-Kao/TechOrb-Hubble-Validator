"""Predetermined v1.1 evidence inventory and fail-closed writer primitives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from .protocol_binding import PROTOCOL_SHA256

FIXED_ROOT_FILES = (
    "execution_authorization.json", "run_manifest.json", "environment_manifest.json",
    "source_manifest.json", "source_manifest.sha256", "raw_output_index.json",
    "negative_controls.json", "leakage_audit.json", "diagnostics.json",
    "machine_verdict.json", "human_report.md", "SHA256SUMS.txt",
)
FIXED_RAW_ARRAYS = (
    "conditional_densities.npy", "conditional_probabilities.npy", "analytic_reference_densities.npy",
    "fidelities.npy", "trace_distances.npy", "neighbor_step_distances.npy",
    "interaction_diagnostics.npy", "clock_b_diagnostics.npy", "coarse_clock_diagnostics.npy",
)

RAW_ARRAY_CONTRACT = {
    "conditional_densities.npy": ("8", True),
    "conditional_probabilities.npy": ("8", True),
    "analytic_reference_densities.npy": ("8", True),
    "fidelities.npy": ("8", True),
    "trace_distances.npy": ("8", True),
    "neighbor_step_distances.npy": ("8", True),
    "interaction_diagnostics.npy": ("9", True),
    "clock_b_diagnostics.npy": ("10", True),
    "coarse_clock_diagnostics.npy": ("11", False),
}

if tuple(RAW_ARRAY_CONTRACT) != FIXED_RAW_ARRAYS:
    raise RuntimeError("raw-array contract must exactly match frozen inventory order")

def expected_relative_paths() -> tuple[str, ...]:
    return FIXED_ROOT_FILES + tuple(f"raw/{name}" for name in FIXED_RAW_ARRAYS) + ("logs/run.log",)

def expected_final_inventory() -> frozenset[str]:
    return frozenset(expected_relative_paths())

def expected_preverdict_inventory() -> frozenset[str]:
    return expected_final_inventory() - {"machine_verdict.json", "SHA256SUMS.txt"}

def actual_evidence_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}

def actual_evidence_directories(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()}

def _validate_exact_inventory(root: Path, expected: frozenset[str], stage: str) -> None:
    actual = actual_evidence_files(root)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(f"{stage} evidence inventory mismatch: missing={missing}, unexpected={unexpected}")
    directories = actual_evidence_directories(root)
    if directories != {"raw", "logs"}:
        raise RuntimeError(
            f"{stage} evidence directory inventory mismatch: "
            f"missing={sorted({'raw', 'logs'} - directories)}, "
            f"unexpected={sorted(directories - {'raw', 'logs'})}"
        )

def validate_exact_preverdict_inventory(root: Path) -> None:
    _validate_exact_inventory(root, expected_preverdict_inventory(), "pre-verdict")

def validate_exact_final_inventory(root: Path) -> None:
    _validate_exact_inventory(root, expected_final_inventory(), "final")

def canonical_evidence_root(repo_root: Path, run_id: str) -> Path:
    return (Path(repo_root).resolve() / "foundation/qf0/qf0a_impl/evidence" / run_id).resolve()

def prepare_new_evidence_root(root: Path, repo_root: Path | None = None, run_id: str | None = None) -> None:
    if repo_root is not None and run_id is not None and root.resolve() != canonical_evidence_root(repo_root, run_id):
        raise RuntimeError("evidence root must equal the canonical repository run path")
    if root.exists():
        raise RuntimeError("evidence root must not exist before official execution")
    root.mkdir(parents=True, exist_ok=False)
    (root / "raw").mkdir()
    (root / "logs").mkdir()

def _array_identity(path: Path, array: np.ndarray) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": f"raw/{path.name}", "dtype": str(array.dtype), "shape": list(array.shape),
            "byte_size": len(data), "sha256": hashlib.sha256(data).hexdigest()}

def write_array_once(raw_root: Path, name: str, value: np.ndarray) -> dict[str, object]:
    if name not in FIXED_RAW_ARRAYS:
        raise ValueError("array is outside frozen inventory")
    array = np.asarray(value)
    if array.dtype not in (np.dtype("float64"), np.dtype("complex128")):
        raise ValueError("raw arrays require float64 or complex128")
    path = raw_root / name
    if path.exists():
        raise FileExistsError(f"refusing to overwrite raw evidence: {path}")
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
    return _array_identity(path, array)

def register_existing_array(raw_root: Path, name: str, expected_sha256: str) -> dict[str, object]:
    path = raw_root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise RuntimeError(f"immutable raw evidence mismatch: {name}")
    array = np.load(path, allow_pickle=False)
    return _array_identity(path, array)

def write_remaining_registered_arrays(raw_root: Path, arrays: dict[str, np.ndarray], frozen: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    if set(arrays) | set(frozen) != set(FIXED_RAW_ARRAYS) or set(arrays) & set(frozen):
        raise ValueError("raw array inventory must equal the frozen inventory exactly once")
    index = {name: register_existing_array(raw_root, name, str(meta["sha256"])) for name, meta in frozen.items()}
    for name, array in arrays.items():
        index[name] = write_array_once(raw_root, name, array)
    return {name: index[name] for name in FIXED_RAW_ARRAYS}

def write_json_once(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

def write_text_once(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.write_text(text, encoding="utf-8", newline="\n")

def write_source_manifest_with_sha(root: Path, payload: object) -> str:
    manifest_path = root / "source_manifest.json"
    write_json_once(manifest_path, payload)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    write_text_once(root / "source_manifest.sha256", digest + "\n")
    return digest

def write_source_manifest_bytes_with_sha(root: Path, committed_bytes: bytes) -> str:
    """Copy the authorization-bound committed manifest bytes without reserialization."""
    manifest_path = root / "source_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {manifest_path}")
    manifest_path.write_bytes(committed_bytes)
    digest = hashlib.sha256(committed_bytes).hexdigest()
    write_text_once(root / "source_manifest.sha256", digest + "\n")
    return digest

def serialize_raw_output_index(run_id: str, index: dict[str, dict[str, object]]) -> dict[str, object]:
    if tuple(index) != FIXED_RAW_ARRAYS:
        raise ValueError("raw array index must equal frozen inventory in frozen order")
    arrays = []
    for name in FIXED_RAW_ARRAYS:
        section, adjudicative = RAW_ARRAY_CONTRACT[name]
        arrays.append({
            "identifier": name.removesuffix(".npy"),
            "path": str(index[name]["path"]),
            "format": "npy-little-endian-no-object",
            "dtype": str(index[name]["dtype"]),
            "shape": list(index[name]["shape"]),
            "byte_size": int(index[name]["byte_size"]),
            "sha256": str(index[name]["sha256"]),
            "protocol_section": section,
            "adjudicative": adjudicative,
        })
    return {
        "schema_version": "1.1", "run_id": run_id,
        "protocol_file_sha256": PROTOCOL_SHA256,
        "outcome_count_expected": len(FIXED_RAW_ARRAYS),
        "outcome_count_recorded": len(arrays),
        "outcomes_dropped": False, "exclusions": [], "arrays": arrays,
    }

def serialize_negative_controls(run_id: str, observations: tuple, definitions: tuple,
                                evidence_path: str, evidence_sha256: str,
                                timestamp_utc: str) -> dict[str, object]:
    definition_map = {item.control_id: item for item in definitions}
    if len(observations) != 6 or len(definition_map) != 6:
        raise ValueError("negative-control inventory must contain exactly six controls")
    controls = []
    seen = set()
    for control, observation, reason in observations:
        if control in seen or control not in definition_map:
            raise ValueError("negative controls must occur exactly once")
        seen.add(control)
        definition = definition_map[control]
        controls.append({
            "control_id": control.value,
            "protocol_section": definition.protocol_section,
            "expected_failure_mode": definition.expected_failure_mode,
            "observed_status": "FAIL" if observation.value == "EXPECTED_FAILURE" else "BLOCKED",
            "expected_failure_observed": observation.value == "EXPECTED_FAILURE",
            "post_hoc_mutation": False,
            "evidence_path": evidence_path,
            "evidence_sha256": evidence_sha256,
            "timestamp_utc": timestamp_utc,
            "reason_code": reason,
        })
    return {
        "schema_version": "1.1", "run_id": run_id,
        "protocol_file_sha256": PROTOCOL_SHA256,
        "all_frozen_controls_present": seen == set(definition_map),
        "controls": controls,
    }

def serialize_machine_verdict(verdict, adjudication) -> dict[str, object]:
    controls = {item[0] for item in adjudication.control_observations}
    expected_controls = len(controls) == 6 and len(adjudication.control_observations) == 6
    return {
        "schema_version": "1.1", "run_id": verdict.run_id,
        "protocol_file_sha256": PROTOCOL_SHA256,
        "pipeline_reproduction_verdict": verdict.pipeline_reproduction_verdict,
        "foundation_model_verdict": verdict.foundation_model_verdict.value,
        "reason_codes": list(verdict.reason_codes),
        "required_evidence_complete": adjudication.required_evidence_complete,
        "negative_controls_complete": expected_controls,
        "leakage_audit_verdict": "PASS" if adjudication.leakage_audit_pass else "BLOCKED",
    }

def validate_complete_inventory(root: Path) -> None:
    """Compatibility name for the exact final-inventory contract."""
    validate_exact_final_inventory(root)

def validate_hash_manifest(root: Path) -> None:
    validate_exact_final_inventory(root)
    expected = {}
    for line in (root / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        if line.count("  ") != 1:
            raise RuntimeError("malformed hash-manifest line")
        digest, rel = line.split("  ", 1)
        if len(digest) != 64 or digest != digest.lower() or any(char not in "0123456789abcdef" for char in digest):
            raise RuntimeError(f"invalid SHA-256 digest for hash-manifest path: {rel}")
        if rel in expected:
            raise RuntimeError(f"duplicate hash-manifest path: {rel}")
        expected[rel] = digest
    expected_checksum_paths = expected_final_inventory() - {"SHA256SUMS.txt"}
    if set(expected) != expected_checksum_paths:
        raise RuntimeError("hash manifest path set does not equal frozen inventory")
    for rel, digest in expected.items():
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"sealed evidence hash mismatch: {rel}")
