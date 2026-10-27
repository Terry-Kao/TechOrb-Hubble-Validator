"""Immutable, fail-closed binding to the reviewed QF-0A protocol."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
from types import MappingProxyType

PROTOCOL_PATH = "foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.1.md"
PROTOCOL_VERSION = "QF0A_FROZEN_PROTOCOL_v1.1"
PROTOCOL_GIT_BLOB = "3af303a471d99397429d33f77ca512578f40f972"
PROTOCOL_SHA256 = "1cf9f7f3a8cfa6a5b0e12a3290d7e545f44586f9359aaa5367754415cdcc5f3a"
PREREGISTRATION_PATH = "foundation/qf0/QF0A_IMPLEMENTATION_PREREGISTRATION_v1.1.md"
PREREGISTRATION_MAIN = "a9e9a8ec0b637d4a724a6d3baaded83b54c0058e"
CANONICAL_HASH_POLICY = "git-committed-bytes-v1"

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

def verify_protocol_identity(repo_root: Path, commit: str = "HEAD") -> ProtocolIdentity:
    committed = subprocess.check_output(["git", "show", f"{commit}:{PROTOCOL_PATH}"], cwd=repo_root)
    if hashlib.sha256(committed).hexdigest() != PROTOCOL_SHA256:
        raise RuntimeError("frozen protocol SHA-256 mismatch")
    blob = subprocess.check_output(
        ["git", "rev-parse", f"{commit}:{PROTOCOL_PATH}"], cwd=repo_root, text=True
    ).strip()
    if blob != PROTOCOL_GIT_BLOB:
        raise RuntimeError("frozen protocol Git blob mismatch")
    return ProtocolIdentity()
