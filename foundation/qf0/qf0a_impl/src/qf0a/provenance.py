"""Stdlib-only runtime source identity closure; performs no scientific imports."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys

MANIFEST_PATH = "foundation/qf0/qf0a_impl/manifests/QF0A_SOURCE_MANIFEST_v1.1.json"
RUNNER_PATH = "foundation/qf0/qf0a_impl/src/qf0a/runner.py"
PROTOCOL_PATH = "foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.1.md"
PREREGISTRATION_PATH = "foundation/qf0/QF0A_IMPLEMENTATION_PREREGISTRATION_v1.1.md"
ENVIRONMENT_PATH = "foundation/qf0/qf0a_impl/environment/QF0A_ENVIRONMENT_FREEZE_v1.1.json"
ALLOWED_POST_BASELINE_PATHS = frozenset({
    "README.md",
    "control/DECISION_LOG.md",
    "control/EVIDENCE_INDEX.md",
    "control/PROJECT_STATE.md",
    MANIFEST_PATH,
    "foundation/qf0/qf0a_impl/manifests/QF0A_V1_1_ADJUDICATION_EVIDENCE_REFREEZE_REPORT.md",
})
ACTIVE_PREFIXES = (
    "foundation/qf0/qf0a_impl/src/qf0a/",
    "foundation/qf0/qf0a_impl/schemas/",
    "foundation/qf0/qf0a_impl/environment/",
)
ACTIVE_EXACT_PATHS = frozenset({PROTOCOL_PATH, PREREGISTRATION_PATH})

class ProvenanceViolation(RuntimeError):
    pass

@dataclass(frozen=True)
class ExecutionSourceSeal:
    current_execution_head: str
    source_baseline_commit: str
    source_manifest_sha256: str
    runner_blob: str
    verified_paths: tuple[str, ...]

def _git(repo_root: Path, *args: str, text: bool = False):
    try:
        return subprocess.check_output(["git", *args], cwd=repo_root, text=text, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise ProvenanceViolation(f"git provenance query failed: {' '.join(args)}") from exc

def _committed_bytes(repo_root: Path, commit: str, path: str) -> bytes:
    return _git(repo_root, "show", f"{commit}:{path}")

def _blob(repo_root: Path, commit: str, path: str) -> str:
    return str(_git(repo_root, "rev-parse", f"{commit}:{path}", text=True)).strip()

def _active_inventory(repo_root: Path, head: str) -> set[str]:
    tracked = set(str(_git(repo_root, "ls-tree", "-r", "--name-only", head, text=True)).splitlines())
    active = set()
    for path in tracked:
        if path in ACTIVE_EXACT_PATHS:
            active.add(path)
        elif path.startswith("foundation/qf0/qf0a_impl/src/qf0a/") and path.endswith(".py"):
            active.add(path)
        elif path.startswith("foundation/qf0/qf0a_impl/schemas/"):
            active.add(path)
        elif path.startswith("foundation/qf0/qf0a_impl/environment/") and "_v1.0" not in path:
            active.add(path)
    return active

def assert_fresh_official_process() -> None:
    allowed = {"qf0a", "qf0a.entrypoint", "qf0a.provenance"}
    loaded = sorted(name for name in sys.modules if name.startswith("qf0a.") and name not in allowed)
    if loaded:
        raise ProvenanceViolation(f"official execution requires a fresh process; preloaded modules: {loaded}")
    if hasattr(sys, "ps1") or "ipykernel" in sys.modules:
        raise ProvenanceViolation("interactive/notebook execution is prohibited")

def verify_execution_head_against_source_manifest(repo_root: Path, authorization: dict[str, object]) -> ExecutionSourceSeal:
    """Bind authorization -> committed HEAD manifest -> every active committed HEAD file."""
    root = Path(repo_root).resolve()
    head = str(_git(root, "rev-parse", "HEAD", text=True)).strip()
    manifest_bytes = _committed_bytes(root, head, MANIFEST_PATH)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != authorization["source_manifest_sha256"]:
        raise ProvenanceViolation("current HEAD source manifest is not bound by authorization")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProvenanceViolation("current HEAD source manifest is not valid canonical JSON") from exc
    baseline = str(manifest.get("source_baseline_commit", ""))
    if baseline != authorization["source_baseline_commit"]:
        raise ProvenanceViolation("source baseline differs from authorization")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ProvenanceViolation("source manifest files inventory is absent")
    by_path = {str(item.get("path")): item for item in entries if isinstance(item, dict)}
    if len(by_path) != len(entries):
        raise ProvenanceViolation("source manifest contains duplicate or invalid paths")
    for path, item in by_path.items():
        data = _committed_bytes(root, head, path)
        if _blob(root, head, path) != item.get("git_blob"):
            raise ProvenanceViolation(f"current HEAD blob mismatch: {path}")
        if len(data) != item.get("byte_length"):
            raise ProvenanceViolation(f"current HEAD byte-length mismatch: {path}")
        if hashlib.sha256(data).hexdigest() != item.get("sha256"):
            raise ProvenanceViolation(f"current HEAD canonical SHA-256 mismatch: {path}")
    required = {RUNNER_PATH, PROTOCOL_PATH, PREREGISTRATION_PATH, ENVIRONMENT_PATH}
    if not required <= set(by_path):
        raise ProvenanceViolation("required runtime identity is missing from source manifest")
    if by_path[RUNNER_PATH]["git_blob"] != authorization["runner_blob"]:
        raise ProvenanceViolation("current HEAD runner is not authorization-bound")
    if by_path[PROTOCOL_PATH]["git_blob"] != authorization["protocol_blob"] or by_path[PROTOCOL_PATH]["sha256"] != authorization["protocol_sha256"]:
        raise ProvenanceViolation("current HEAD protocol is not authorization-bound")
    if by_path[PREREGISTRATION_PATH]["git_blob"] != authorization["preregistration_blob"] or by_path[PREREGISTRATION_PATH]["sha256"] != authorization["preregistration_sha256"]:
        raise ProvenanceViolation("current HEAD preregistration is not authorization-bound")
    if by_path[ENVIRONMENT_PATH]["sha256"] != authorization["environment_sha256"]:
        raise ProvenanceViolation("current HEAD environment is not authorization-bound")
    expected_active = {path for path in by_path if path in ACTIVE_EXACT_PATHS or any(path.startswith(prefix) for prefix in ACTIVE_PREFIXES)}
    actual_active = _active_inventory(root, head)
    if actual_active != expected_active:
        raise ProvenanceViolation(f"unmanifested or missing active execution path: {sorted(actual_active ^ expected_active)}")
    changed = set(str(_git(root, "diff", "--name-only", baseline, head, text=True)).splitlines())
    if not changed <= ALLOWED_POST_BASELINE_PATHS:
        raise ProvenanceViolation(f"post-baseline change outside governance allowlist: {sorted(changed - ALLOWED_POST_BASELINE_PATHS)}")
    return ExecutionSourceSeal(head, baseline, manifest_sha, str(authorization["runner_blob"]), tuple(sorted(by_path)))
