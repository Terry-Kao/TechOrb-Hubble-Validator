"""Predetermined v1.1 evidence inventory and fail-closed writer primitives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

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

def expected_relative_paths() -> tuple[str, ...]:
    return FIXED_ROOT_FILES + tuple(f"raw/{name}" for name in FIXED_RAW_ARRAYS) + ("logs/run.log",)

def prepare_new_evidence_root(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("evidence root must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
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

def validate_complete_inventory(root: Path) -> None:
    missing = [path for path in expected_relative_paths() if not (root / path).is_file()]
    if missing:
        raise RuntimeError(f"evidence inventory incomplete: {missing}")
