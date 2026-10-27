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
    "machine_verdict.json", "human_report_metadata.json", "SHA256SUMS.txt",
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

def write_all_registered_arrays(raw_root: Path, arrays: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    if set(arrays) != set(FIXED_RAW_ARRAYS):
        raise ValueError("raw array inventory must equal the frozen inventory")
    index = {}
    for name in FIXED_RAW_ARRAYS:
        array = np.asarray(arrays[name])
        if array.dtype not in (np.dtype("float64"), np.dtype("complex128")):
            raise ValueError("raw arrays require float64 or complex128")
        path = raw_root / name
        np.save(path, array, allow_pickle=False)
        data = path.read_bytes()
        index[name] = {"path": f"raw/{name}", "dtype": str(array.dtype), "shape": list(array.shape), "byte_size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return index

def write_json_once(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

def validate_complete_inventory(root: Path) -> None:
    missing = [path for path in expected_relative_paths() if not (root / path).is_file()]
    if missing:
        raise RuntimeError(f"evidence inventory incomplete: {missing}")
