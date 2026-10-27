"""Draft 2020-12 and semantic validation only; no QF-0A science is run."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
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
PROTOCOL = QF0 / "QF0A_FROZEN_PROTOCOL_v1.1.md"
SPEC = QF0 / "QF0A_IMPLEMENTATION_PREREGISTRATION_v1.1.md"
EXPECTED_MAIN = "a9e9a8ec0b637d4a724a6d3baaded83b54c0058e"
PROTOCOL_COMMIT = "902bdacd0e556d2a19079ab46c6c44d17f665372"
EXPECTED_BLOB = "3af303a471d99397429d33f77ca512578f40f972"
EXPECTED_SHA256 = "1cf9f7f3a8cfa6a5b0e12a3290d7e545f44586f9359aaa5367754415cdcc5f3a"
SCHEMAS = {
    "QF0A_RUN_MANIFEST.schema.json",
    "QF0A_MACHINE_VERDICT.schema.json",
    "QF0A_RAW_OUTPUT.schema.json",
    "QF0A_NEGATIVE_CONTROL_RESULT.schema.json",
    "QF0A_EXECUTION_AUTHORIZATION.schema.json",
}
FIXTURE_SCHEMAS = SCHEMAS - {"QF0A_EXECUTION_AUTHORIZATION.schema.json"}
AUTHORIZED_FREEZE_SOURCE = {"__init__.py", "protocol_binding.py", "analytic_reference.py", "conditional_state.py", "controls.py", "metrics.py", "adjudication.py", "evidence.py", "runner.py", "audit.py", "verdict.py"}
FORBIDDEN_SUFFIXES = {".npy", ".npz", ".h5", ".hdf5", ".pkl", ".pickle", ".parquet"}
FORBIDDEN_RESULT_NAMES = {"machine_verdict.json", "raw_output_index.json", "negative_controls.json", "environment_manifest.json", "source_manifest.json", "leakage_audit.json", "SHA256SUMS.txt"}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$")
CANONICAL_EVIDENCE_PATHS = {
    "source_manifest_path": "source_manifest.json",
    "environment_manifest": "environment_manifest.json",
    "raw_output_index": "raw_output_index.json",
    "negative_controls": "negative_controls.json",
    "leakage_audit": "leakage_audit.json",
    "machine_verdict": "machine_verdict.json",
    "human_report": "human_report.md",
    "hash_manifest": "SHA256SUMS.txt",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def validate_canonical_run_id(value: object) -> list[str]:
    if not isinstance(value, str):
        return ["run_id must be a string"]
    errors: list[str] = []
    if not RUN_ID_PATTERN.fullmatch(value):
        errors.append("run_id violates canonical bounded single-component syntax")
    if value in {"", ".", ".."} or "/" in value or "\\" in value:
        errors.append("run_id cannot be empty, dot, dot-dot, or contain path separators")
    if re.match(r"^[A-Za-z]:", value) or value.startswith(("//", "\\\\", "\\\\?\\")):
        errors.append("run_id cannot use drive, UNC, or device semantics")
    if posixpath.normpath(value) != value or len(PurePosixPath(value).parts) != 1:
        errors.append("run_id must normalize unchanged to one POSIX path component")
    return errors


def validate_canonical_relative_path(value: object, *, raw_only: bool = False, run_id: object) -> list[str]:
    errors: list[str] = []
    run_id_errors = validate_canonical_run_id(run_id)
    if run_id_errors:
        return [f"cannot construct evidence root before valid run_id: {error}" for error in run_id_errors]
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
    root = PurePosixPath("foundation/qf0/qf0a_impl/evidence") / str(run_id)
    joined = root / pure
    if tuple(joined.parts[: len(root.parts)]) != tuple(root.parts):
        errors.append("path escapes declared evidence run root")
    if raw_only and (not value.startswith("raw/") or not value.endswith(".npy")):
        errors.append("raw array path must be raw/<canonical-name>.npy")
    return errors


def validate_timestamp_pair(execution: object) -> list[str]:
    if not isinstance(execution, dict):
        return ["execution must be an object"]
    utc_text = execution.get("timestamp_utc")
    taipei_text = execution.get("timestamp_asia_taipei")
    errors: list[str] = []
    if not isinstance(utc_text, str) or not utc_text.endswith("Z"):
        errors.append("timestamp_utc must be RFC 3339 UTC with Z suffix")
    if not isinstance(taipei_text, str) or not taipei_text.endswith("+08:00"):
        errors.append("timestamp_asia_taipei must use +08:00 offset")
    try:
        utc_value = datetime.fromisoformat(utc_text.replace("Z", "+00:00"))
        taipei_value = datetime.fromisoformat(taipei_text)
    except (AttributeError, TypeError, ValueError):
        return errors + ["timestamp pair must contain parseable timezone-aware RFC 3339 values"]
    if utc_value.utcoffset() != timezone.utc.utcoffset(utc_value):
        errors.append("timestamp_utc offset must be UTC")
    if taipei_value.utcoffset() is None or taipei_value.utcoffset().total_seconds() != 8 * 3600:
        errors.append("timestamp_asia_taipei offset must equal +08:00")
    if utc_value.astimezone(timezone.utc) != taipei_value.astimezone(timezone.utc):
        errors.append("UTC and Asia/Taipei timestamps must represent the same instant")
    return errors


def validate_cross_artifact_run_identity(package: object) -> list[str]:
    if not isinstance(package, dict):
        return ["evidence package must be an object"]
    names = ("run_manifest", "raw_output", "negative_controls", "machine_verdict")
    run_ids: dict[str, object] = {}
    errors: list[str] = []
    for name in names:
        artifact = package.get(name)
        if not isinstance(artifact, dict):
            errors.append(f"{name} missing from evidence package")
            continue
        run_ids[name] = artifact.get("run_id")
        errors.extend(f"{name}: {error}" for error in validate_canonical_run_id(artifact.get("run_id")))
    values = list(run_ids.values())
    if not values or any(value != values[0] for value in values[1:]):
        errors.append("all four artifact run_id values must be identical")
    directory_basename = package.get("evidence_directory_basename")
    errors.extend(f"evidence directory: {error}" for error in validate_canonical_run_id(directory_basename))
    if values and directory_basename != values[0]:
        errors.append("evidence directory basename must equal artifact run_id")
    return errors


def semantic_errors(schema_name: str, instance: object) -> list[str]:
    if not isinstance(instance, dict):
        return ["instance must be an object"]
    errors: list[str] = []
    run_id = instance.get("run_id")
    errors.extend(validate_canonical_run_id(run_id))
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
        errors.extend(validate_timestamp_pair(execution))
        if isinstance(implementation, dict) and implementation.get("source_manifest_path") != CANONICAL_EVIDENCE_PATHS["source_manifest_path"]:
            errors.append("source manifest path must use canonical root-level filename")
        if isinstance(evidence, dict):
            for key, expected in CANONICAL_EVIDENCE_PATHS.items():
                if key != "source_manifest_path" and evidence.get(key) != expected:
                    errors.append(f"{key} must use canonical root-level filename {expected}")
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
    protocol_bytes = subprocess.check_output(["git", "show", f"{PROTOCOL_COMMIT}:foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.1.md"], cwd=ROOT)
    digest = hashlib.sha256(protocol_bytes).hexdigest()
    if digest != EXPECTED_SHA256:
        errors.append(f"protocol SHA-256 mismatch: {digest}")
    blob = git("rev-parse", f"{PROTOCOL_COMMIT}:foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.1.md")
    if blob != EXPECTED_BLOB:
        errors.append(f"protocol Git blob mismatch: {blob}")

    if "PROTOCOL FROZEN; REPRODUCTION NOT EXECUTED" not in PROTOCOL.read_text(encoding="utf-8"):
        errors.append("protocol scientific status changed")
    if not SPEC.exists():
        errors.append("v1.1 preregistration missing")

    schema_dir = IMPL / "schemas"
    schema_paths = sorted(schema_dir.glob("*.schema.json"))
    if {path.name for path in schema_paths} != SCHEMAS:
        errors.append("schema inventory differs from frozen five-schema contract")
    schemas: dict[str, dict] = {}
    for path in schema_paths:
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            schemas[path.name] = schema
        except (UnicodeDecodeError, json.JSONDecodeError, SchemaError) as exc:
            errors.append(f"{path.name}: schema validation failed: {exc}")

    positive, negative, package_positive, package_negative = fixtures()
    format_checker = FormatChecker()
    positive_passed = 0
    if FIXTURE_SCHEMAS.issubset(schemas):
        for name, (schema_name, instance) in positive.items():
            schema_fail = list(Draft202012Validator(schemas[schema_name], format_checker=format_checker).iter_errors(instance))
            semantic_fail = semantic_errors(schema_name, instance)
            if schema_fail or semantic_fail:
                errors.append(f"positive fixture {name} failed: schema={len(schema_fail)} semantic={semantic_fail}")
            else:
                positive_passed += 1

    negative_rejected = 0
    if FIXTURE_SCHEMAS.issubset(schemas):
        for case in negative:
            schema_fail = bool(list(Draft202012Validator(schemas[case["schema"]], format_checker=format_checker).iter_errors(case["fixture"])))
            semantic_fail = bool(semantic_errors(case["schema"], case["fixture"]))
            expected = case["expected_layer"]
            rejected = ((expected == "schema" and schema_fail) or (expected == "semantic" and semantic_fail) or (expected == "both" and schema_fail and semantic_fail))
            if not rejected:
                errors.append(f"negative fixture {case['name']} unexpectedly accepted at {expected}: {case['reason']} (schema={schema_fail}, semantic={semantic_fail})")
            else:
                negative_rejected += 1

    package_positive_passed = 0
    for name, package in package_positive.items():
        package_fail = validate_cross_artifact_run_identity(package)
        if package_fail:
            errors.append(f"positive package fixture {name} failed: {package_fail}")
        else:
            package_positive_passed += 1
    package_negative_rejected = 0
    for case in package_negative:
        if validate_cross_artifact_run_identity(case["fixture"]):
            package_negative_rejected += 1
        else:
            errors.append(f"negative package fixture {case['name']} unexpectedly accepted: {case['reason']}")

    tracked_or_untracked = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    for name in tracked_or_untracked:
        path = Path(name)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"scientific/generated array forbidden in checkpoint: {name}")
        normalized = name.replace("\\", "/")
        if normalized.startswith("foundation/qf0/qf0a_impl/src/qf0a/") and path.name not in AUTHORIZED_FREEZE_SOURCE:
            errors.append(f"source outside authorized freeze inventory: {name}")
        if normalized.startswith("foundation/qf0/qf0a_impl/src/") and not normalized.startswith("foundation/qf0/qf0a_impl/src/qf0a/"):
            errors.append(f"source outside authorized package: {name}")
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
    print(f"- Draft202012Validator.check_schema: {len(schemas)}/5 PASS")
    print("- FormatChecker: PASS")
    total_positive = len(positive) + len(package_positive)
    total_negative = len(negative) + len(package_negative)
    print(f"- positive fixtures: {positive_passed + package_positive_passed}/{total_positive} PASS")
    print(f"- negative fixtures: {negative_rejected + package_negative_rejected}/{total_negative} expected rejection PASS")
    print("- canonical run_id suite: PASS")
    print("- cross-artifact run-id and evidence-directory basename suite: PASS")
    print("- UTC/Asia-Taipei timestamp equivalence suite: PASS")
    print("- canonical evidence layout suite: PASS")
    print("- canonical/path traversal suite: PASS")
    print("- Windows drive/backslash/UNC/device path rejection suite: PASS")
    print("- exact six-control inventory and duplicate rejection: PASS")
    print("- raw count equality and identifier/path uniqueness rejection: PASS")
    print(f"- protocol blob: {EXPECTED_BLOB}")
    print(f"- protocol SHA-256: {EXPECTED_SHA256}")
    print("- authorized source-freeze modules: PRESENT; generated scientific results: ABSENT")
    print("- scientific/numerical execution: NOT PERFORMED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
