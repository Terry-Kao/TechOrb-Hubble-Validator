"""Draft 2020-12 and semantic validation only; no QF-0A science is run."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import posixpath
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError

from fixtures.synthetic_fixtures import CONTROL_IDS, fixtures


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
FORBIDDEN_SOURCE = {"analytic_reference.py", "conditional_state.py", "protocol_binding.py", "controls.py", "audit.py", "verdict.py"}
FORBIDDEN_SUFFIXES = {".npy", ".npz", ".h5", ".hdf5", ".pkl", ".pickle", ".parquet"}
FORBIDDEN_RESULT_NAMES = {"machine_verdict.json", "raw_output_index.json", "negative_controls.json", "environment_manifest.json", "source_manifest.json", "leakage_audit.json", "SHA256SUMS.txt"}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def validate_canonical_relative_path(value: object, *, raw_only: bool = False, run_id: str = "synthetic-contract-test") -> list[str]:
    errors: list[str] = []
    if not isinstance(value, str) or not value:
        return ["path must be a non-empty string"]
    if value.startswith("/") or value.startswith("//"):
        errors.append("absolute/UNC POSIX path forbidden")
    if "\\" in value:
        errors.append("backslash, UNC, and Windows device notation forbidden")
    if re.match(r"^[A-Za-z]:", value):
        errors.append("Windows drive prefix forbidden")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        errors.append("empty/dot/parent path segment forbidden")
    if posixpath.normpath(value) != value:
        errors.append("path is not canonical POSIX notation")
    pure = PurePosixPath(value)
    root = PurePosixPath("foundation/qf0/qf0a_impl/evidence") / run_id
    joined = root / pure
    if tuple(joined.parts[: len(root.parts)]) != tuple(root.parts):
        errors.append("path escapes declared evidence run root")
    if raw_only and (not value.startswith("raw/") or not value.endswith(".npy")):
        errors.append("raw array path must be raw/<canonical-name>.npy")
    return errors


def semantic_errors(schema_name: str, instance: object) -> list[str]:
    if not isinstance(instance, dict):
        return ["instance must be an object"]
    errors: list[str] = []
    run_id = str(instance.get("run_id", "invalid-run-id"))
    if schema_name == "QF0A_RUN_MANIFEST.schema.json":
        paths: list[object] = []
        implementation = instance.get("implementation", {})
        evidence = instance.get("evidence", {})
        if isinstance(implementation, dict):
            paths.append(implementation.get("source_manifest_path"))
        if isinstance(evidence, dict):
            paths.extend(evidence.values())
        for value in paths:
            errors.extend(validate_canonical_relative_path(value, run_id=run_id))
        execution = instance.get("execution", {})
        if isinstance(execution, dict) and not str(execution.get("timestamp_utc", "")).endswith("Z"):
            errors.append("timestamp_utc must be RFC 3339 UTC using Z")
    elif schema_name == "QF0A_NEGATIVE_CONTROL_RESULT.schema.json":
        controls = instance.get("controls", [])
        if not isinstance(controls, list):
            return ["controls must be a list"]
        ids = [item.get("control_id") for item in controls if isinstance(item, dict)]
        if len(ids) != 6 or set(ids) != set(CONTROL_IDS) or len(ids) != len(set(ids)):
            errors.append("control inventory must equal the exact six unique frozen IDs")
        paths = [item.get("evidence_path") for item in controls if isinstance(item, dict)]
        if len(paths) != len(set(paths)):
            errors.append("negative-control evidence paths must be unique")
        for item in controls:
            if not isinstance(item, dict):
                errors.append("control record must be an object")
                continue
            errors.extend(validate_canonical_relative_path(item.get("evidence_path"), run_id=run_id))
            if not str(item.get("timestamp_utc", "")).endswith("Z"):
                errors.append(f"{item.get('control_id')}: timestamp_utc must use Z")
            if item.get("observed_status") != "FAIL" or item.get("expected_failure_observed") is not True:
                errors.append(f"{item.get('control_id')}: expected negative-control failure was not observed")
        if instance.get("all_frozen_controls_present") is not True:
            errors.append("all_frozen_controls_present must be true and derived from exact inventory")
    elif schema_name == "QF0A_RAW_OUTPUT.schema.json":
        if instance.get("outcome_count_expected") != instance.get("outcome_count_recorded"):
            errors.append("outcome_count_expected must equal outcome_count_recorded")
        if instance.get("outcomes_dropped") is not False:
            errors.append("outcomes_dropped must be false")
        if instance.get("exclusions") != []:
            errors.append("QF-0A primary run permits no exclusions")
        arrays = instance.get("arrays", [])
        if not isinstance(arrays, list):
            return errors + ["arrays must be a list"]
        identifiers = [item.get("identifier") for item in arrays if isinstance(item, dict)]
        paths = [item.get("path") for item in arrays if isinstance(item, dict)]
        if len(identifiers) != len(set(identifiers)):
            errors.append("raw array identifiers must be unique")
        if len(paths) != len(set(paths)):
            errors.append("raw array paths must be unique")
        for item in arrays:
            if not isinstance(item, dict):
                errors.append("array record must be an object")
                continue
            errors.extend(validate_canonical_relative_path(item.get("path"), raw_only=True, run_id=run_id))
            if item.get("dtype") not in {"float64", "complex128"}:
                errors.append("raw dtype is not frozen allowed dtype")
            if not isinstance(item.get("byte_size"), int) or item.get("byte_size", 0) <= 0:
                errors.append("raw byte_size must be positive")
            shape = item.get("shape")
            if not isinstance(shape, list) or not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
                errors.append("raw shape dimensions must be positive")
            if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
                errors.append("raw SHA-256 invalid")
    return errors


def main() -> int:
    errors: list[str] = []
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        errors.append(f"protocol SHA-256 mismatch: {digest}")
    blob = git("rev-parse", f"{EXPECTED_MAIN}:foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.0.md")
    if blob != EXPECTED_BLOB:
        errors.append(f"protocol Git blob mismatch: {blob}")

    if "PROTOCOL FROZEN; REPRODUCTION NOT EXECUTED" not in PROTOCOL.read_text(encoding="utf-8"):
        errors.append("protocol scientific status changed")
    if "foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.0.md" in set(git("diff", "--name-only", EXPECTED_MAIN).splitlines()):
        errors.append("frozen protocol was modified")

    schema_dir = IMPL / "schemas"
    schema_paths = sorted(schema_dir.glob("*.schema.json"))
    if {path.name for path in schema_paths} != SCHEMAS:
        errors.append("schema inventory differs from frozen four-schema contract")
    schemas: dict[str, dict] = {}
    for path in schema_paths:
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            schemas[path.name] = schema
        except (UnicodeDecodeError, json.JSONDecodeError, SchemaError) as exc:
            errors.append(f"{path.name}: schema validation failed: {exc}")

    positive, negative = fixtures()
    format_checker = FormatChecker()
    positive_passed = 0
    if len(schemas) == 4:
        for name, (schema_name, instance) in positive.items():
            schema_fail = list(Draft202012Validator(schemas[schema_name], format_checker=format_checker).iter_errors(instance))
            semantic_fail = semantic_errors(schema_name, instance)
            if schema_fail or semantic_fail:
                errors.append(f"positive fixture {name} failed: schema={len(schema_fail)} semantic={semantic_fail}")
            else:
                positive_passed += 1

    negative_rejected = 0
    if len(schemas) == 4:
        for case in negative:
            schema_fail = bool(list(Draft202012Validator(schemas[case["schema"]], format_checker=format_checker).iter_errors(case["fixture"])))
            semantic_fail = bool(semantic_errors(case["schema"], case["fixture"]))
            expected = case["expected_layer"]
            rejected = ((expected == "schema" and schema_fail) or (expected == "semantic" and semantic_fail) or (expected == "both" and schema_fail and semantic_fail))
            if not rejected:
                errors.append(f"negative fixture {case['name']} unexpectedly accepted at {expected}: {case['reason']} (schema={schema_fail}, semantic={semantic_fail})")
            else:
                negative_rejected += 1

    tracked_or_untracked = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    for name in tracked_or_untracked:
        path = Path(name)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"scientific/generated array forbidden in checkpoint: {name}")
        normalized = name.replace("\\", "/")
        if normalized.startswith("foundation/qf0/qf0a_impl/src/") or path.name in FORBIDDEN_SOURCE:
            errors.append(f"scientific source module forbidden in checkpoint: {name}")
        if normalized.startswith("foundation/qf0/qf0a_impl/") and path.name in FORBIDDEN_RESULT_NAMES:
            errors.append(f"result/evidence instance forbidden in checkpoint: {name}")
        if re.search(r"^foundation/qf0/qf0a_impl/(?:evidence|raw)/[^/]+", normalized):
            errors.append(f"populated evidence/raw directory forbidden: {name}")

    if errors:
        print("QF-0A preregistration validation: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("QF-0A preregistration validation: PASS")
    print(f"- jsonschema: {version('jsonschema')}")
    print(f"- Draft202012Validator.check_schema: {len(schemas)}/4 PASS")
    print("- FormatChecker: PASS")
    print(f"- positive fixtures: {positive_passed}/{len(positive)} PASS")
    print(f"- negative fixtures: {negative_rejected}/{len(negative)} expected rejection PASS")
    print("- canonical/path traversal suite: PASS")
    print("- Windows drive/backslash/UNC/device path rejection suite: PASS")
    print("- exact six-control inventory and duplicate rejection: PASS")
    print("- raw count equality and identifier/path uniqueness rejection: PASS")
    print(f"- protocol blob: {EXPECTED_BLOB}")
    print(f"- protocol SHA-256: {EXPECTED_SHA256}")
    print("- scientific source/results: ABSENT")
    print("- scientific/numerical execution: NOT PERFORMED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
