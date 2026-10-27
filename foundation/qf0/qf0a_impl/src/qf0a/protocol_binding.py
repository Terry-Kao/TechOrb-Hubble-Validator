"""Immutable, fail-closed binding to the reviewed QF-0A protocol."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess

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
TOLERANCES = {
    "state_identity": 1e-12,
    "fidelity_defect": 1e-10,
    "trace_distance": 1e-10,
    "window_boundary": 1e-10,
    "projector": 1e-12,
    "projection_norm_min": 1e-12,
}

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
