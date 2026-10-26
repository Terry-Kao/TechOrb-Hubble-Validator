"""Structural/static validation for QF-0A preregistration; no science is run."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
QF0 = ROOT / "foundation" / "qf0"
IMPL = QF0 / "qf0a_impl"
PROTOCOL = QF0 / "QF0A_FROZEN_PROTOCOL_v1.0.md"
SPEC = QF0 / "QF0A_IMPLEMENTATION_PREREGISTRATION_v1.0.md"
EXPECTED_MAIN = "4c2920871898add6254902fb3963784e2761d375"
EXPECTED_BLOB = "c11599a058caa8dfb0912c4e00a0656bef4688bd"
EXPECTED_SHA256 = "fb7969c11e9e2d4b1eebdd2495692dc422eca418bbbaa85bbabcda2815811d8b"
SCHEMAS = {
    "QF0A_RUN_MANIFEST.schema.json",
    "QF0A_MACHINE_VERDICT.schema.json",
    "QF0A_RAW_OUTPUT.schema.json",
    "QF0A_NEGATIVE_CONTROL_RESULT.schema.json",
}
FORBIDDEN_SUFFIXES = {".npy", ".npz", ".h5", ".hdf5", ".pkl", ".pickle", ".parquet"}
FORBIDDEN_RESULT_NAMES = {"machine_verdict.json", "raw_output_index.json", "negative_controls.json", "environment_manifest.json", "source_manifest.json", "leakage_audit.json", "SHA256SUMS.txt"}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    errors: list[str] = []
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        errors.append(f"protocol SHA-256 mismatch: {digest}")
    blob = git("rev-parse", f"{EXPECTED_MAIN}:foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.0.md")
    if blob != EXPECTED_BLOB:
        errors.append(f"protocol Git blob mismatch: {blob}")

    spec = SPEC.read_text(encoding="utf-8")
    for marker in (EXPECTED_MAIN, EXPECTED_BLOB, EXPECTED_SHA256, "PROPOSED / DRAFT / NOT EXECUTED", "Result blindness", "No-external-time", "NOT_APPLICABLE", "NOT_ESTABLISHED"):
        if marker not in spec:
            errors.append(f"preregistration marker missing: {marker}")

    schema_dir = IMPL / "schemas"
    if {path.name for path in schema_dir.glob("*.schema.json")} != SCHEMAS:
        errors.append("schema inventory differs from frozen four-schema contract")
    for path in sorted(schema_dir.glob("*.schema.json")):
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: JSON parse failure: {exc}")
            continue
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            errors.append(f"{path.name}: wrong JSON Schema dialect")

    verdict = json.loads((schema_dir / "QF0A_MACHINE_VERDICT.schema.json").read_text(encoding="utf-8"))
    props = verdict["properties"]
    if props["pipeline_reproduction_verdict"]["enum"] != ["PASS", "FAIL", "BLOCKED", "INVALIDATED"]:
        errors.append("pipeline verdict vocabulary mismatch")
    if set(props["foundation_model_verdict"]["enum"]) != {"NOT_APPLICABLE", "NOT_ESTABLISHED"}:
        errors.append("Foundation/model verdict vocabulary mismatch")

    tracked_or_untracked = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    for name in tracked_or_untracked:
        path = Path(name)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"scientific/generated array forbidden in checkpoint: {name}")
        normalized = name.replace("\\", "/")
        if normalized.startswith("foundation/qf0/qf0a_impl/") and path.name in FORBIDDEN_RESULT_NAMES:
            errors.append(f"result/evidence instance forbidden in checkpoint: {name}")
        if re.search(r"^foundation/qf0/qf0a_impl/(?:evidence|raw)/[^/]+", normalized):
            errors.append(f"populated evidence/raw directory forbidden: {name}")

    changed = set(git("diff", "--name-only", EXPECTED_MAIN).splitlines())
    if "foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.0.md" in changed:
        errors.append("frozen protocol was modified")

    if errors:
        print("QF-0A preregistration validation: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("QF-0A preregistration validation: PASS")
    print(f"- protocol blob: {EXPECTED_BLOB}")
    print(f"- protocol SHA-256: {EXPECTED_SHA256}")
    print(f"- schemas parsed: {len(SCHEMAS)}")
    print("- scientific/numerical execution: NOT PERFORMED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
