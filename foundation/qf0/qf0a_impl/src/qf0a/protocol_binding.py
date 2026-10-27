"""Immutable, fail-closed binding to the reviewed QF-0A protocol."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
from types import MappingProxyType

PROTOCOL_PATH = "foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.0.md"
PROTOCOL_VERSION = "QF0A_FROZEN_PROTOCOL_v1.0"
PROTOCOL_GIT_BLOB = "c11599a058caa8dfb0912c4e00a0656bef4688bd"
PROTOCOL_SHA256 = "fb7969c11e9e2d4b1eebdd2495692dc422eca418bbbaa85bbabcda2815811d8b"
PREREGISTRATION_MAIN = "4c2920871898add6254902fb3963784e2761d375"
PREREGISTRATION_APPROVED_HEAD = "9b29b40156a68797dc36c6ebd7555e9a3515ae4f"
PREREGISTRATION_MERGE_MAIN = "2d125671c0d882df6d1de4eb2a0066d5bad989de"

D_C = 16
D_S = 2
OMEGA = 1.0
HBAR = 1.0
CONSTRAINT_SIGN = "H_C tensor I_S - I_C tensor H_S"
CLOCK_PHASE_SIGN = +1
HISTORY_STATE_TERMS = ((0, 0), (1, 1))
DTYPES = ("float64", "complex128")
RESOLUTION_D_C = (8, 16, 32)
INTERACTION_G = (0.0, 1e-4, 1e-3, 1e-2)
SPECTRAL_WINDOW = (-0.5, 0.5)
SEEDS = (104729, 130363, 155921)
CLOCK_B_SHIFT = 3
CLOCK_SCRAMBLE = (1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12)
FROZEN_THRESHOLDS = MappingProxyType({
    "TRACE_NORMALIZATION_TOL": 1e-12,
    "HERMITICITY_TOL": 1e-12,
    "MIN_EIGENVALUE_TOL": -1e-12,
    "CONSTRAINT_RESIDUAL_TOL": 1e-12,
    "POVM_COMPLETENESS_TOL": 1e-12,
    "PROBABILITY_SUM_TOL": 1e-12,
    "MIN_OUTCOME_PROBABILITY": 1e-12,
    "CONDITIONAL_TRACE_TOL": 1e-12,
    "CONDITIONAL_MIN_EIGENVALUE_TOL": -1e-12,
    "FIDELITY_DEFECT_TOL": 1e-10,
    "TRACE_DISTANCE_TOL": 1e-10,
    "UNITARY_STEP_TRACE_DISTANCE_TOL": 1e-10,
    "UNITARY_NORM_DEFECT_TOL": 1e-12,
    "RESOLUTION_DETERIORATION_TOL": 1e-8,
    "WINDOW_BOUNDARY_TOL": 1e-10,
    "PROJECTOR_HERMITICITY_TOL": 1e-12,
    "PROJECTOR_IDEMPOTENCE_TOL": 1e-12,
    "PROJECTION_NORM_MIN": 1e-12,
    "G0_RECOVERY_TRACE_DISTANCE_TOL": 1e-12,
    "DETERMINISM_TRACE_DISTANCE_TOL": 1e-12,
    "CLOCK_MAPPED_TRACE_DISTANCE_TOL": 1e-10,
    "CLOCK_PROBABILITY_DIFFERENCE_TOL": 1e-12,
})

FROZEN_STRUCTURE = MappingProxyType({
    "D_C": D_C, "D_S": D_S, "OMEGA": OMEGA, "HBAR": HBAR,
    "RESOLUTION_D_C": RESOLUTION_D_C, "INTERACTION_G": INTERACTION_G,
    "SPECTRAL_WINDOW": SPECTRAL_WINDOW, "SEEDS": SEEDS,
    "CLOCK_B_SHIFT": CLOCK_B_SHIFT, "CLOCK_SCRAMBLE": CLOCK_SCRAMBLE,
    "CONSTRAINT_SIGN": CONSTRAINT_SIGN, "CLOCK_PHASE_SIGN": CLOCK_PHASE_SIGN,
    "HISTORY_STATE_TERMS": HISTORY_STATE_TERMS,
})

def frozen_binding_snapshot() -> dict[str, object]:
    return {"thresholds": dict(FROZEN_THRESHOLDS), "structure": dict(FROZEN_STRUCTURE)}

@dataclass(frozen=True)
class ProtocolIdentity:
    path: str = PROTOCOL_PATH
    version: str = PROTOCOL_VERSION
    git_blob: str = PROTOCOL_GIT_BLOB
    sha256: str = PROTOCOL_SHA256

def verify_protocol_identity(repo_root: Path) -> ProtocolIdentity:
    path = repo_root / PROTOCOL_PATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise RuntimeError("frozen protocol SHA-256 mismatch")
    blob = subprocess.check_output(
        ["git", "hash-object", PROTOCOL_PATH], cwd=repo_root, text=True
    ).strip()
    if blob != PROTOCOL_GIT_BLOB:
        raise RuntimeError("frozen protocol Git blob mismatch")
    return ProtocolIdentity()
