"""Artificial contract fixtures only; contains no QF-0A scientific values."""

from __future__ import annotations

from copy import deepcopy

SHA = "a" * 64
PROTOCOL_SHA = "fb7969c11e9e2d4b1eebdd2495692dc422eca418bbbaa85bbabcda2815811d8b"
PROTOCOL_BLOB = "c11599a058caa8dfb0912c4e00a0656bef4688bd"
MAIN = "4c2920871898add6254902fb3963784e2761d375"
CONTROL_IDS = ("BROKEN_CONSTRAINT", "PRODUCT_NO_HISTORY", "WRONG_CONDITIONING", "HIDDEN_EXTERNAL_TIME", "CLOCK_SCRAMBLE", "POST_HOC_THRESHOLD_MUTATION")
RUN_ID = "qf0a-20261027-001"

RUN_MANIFEST = {
    "schema_version": "1.0", "run_id": RUN_ID,
    "protocol": {"version": "QF0A_FROZEN_PROTOCOL_v1.0", "path": "foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.0.md", "git_blob_sha": PROTOCOL_BLOB, "file_sha256": PROTOCOL_SHA, "authoritative_main_sha": MAIN},
    "implementation": {"commit_sha": "b" * 40, "branch": "synthetic", "dirty": False, "source_manifest_path": "source_manifest.json", "source_manifest_sha256": SHA},
    "environment": {"os": "synthetic", "kernel": "synthetic", "cpu_architecture": "x86_64", "cpu_model": "synthetic", "python": "CPython 3.12.7", "packages_lock_sha256": SHA, "numpy_version": "2.1.3", "scipy_version": "1.14.1", "blas_lapack": {"vendor": "synthetic", "version_or_build": "synthetic"}, "locale": "C", "timezone": "Asia/Taipei", "numeric_dtypes": ["float64", "complex128"]},
    "execution": {"timestamp_utc": "2026-01-01T00:00:00Z", "timestamp_asia_taipei": "2026-01-01T08:00:00+08:00", "command_line": ["python", "synthetic.py"], "cwd": "workspace/qf0a"},
    "determinism": {"seeds": {"synthetic": 1}, "thread_environment": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "UNSET", "VECLIB_MAXIMUM_THREADS": "UNSET", "NUMEXPR_NUM_THREADS": "1"}, "eigensolver_policy": "frozen-full-spectral-projector", "tolerance_policy": "frozen-protocol-only"},
    "evidence": {"environment_manifest": "environment_manifest.json", "raw_output_index": "raw_output_index.json", "negative_controls": "negative_controls.json", "leakage_audit": "leakage_audit.json", "machine_verdict": "machine_verdict.json", "human_report": "human_report.md", "hash_manifest": "SHA256SUMS.txt"},
}
MACHINE_VERDICT = {"schema_version": "1.0", "run_id": RUN_ID, "protocol_file_sha256": PROTOCOL_SHA, "pipeline_reproduction_verdict": "BLOCKED", "foundation_model_verdict": "NOT_APPLICABLE", "reason_codes": ["SYNTHETIC_BLOCKED"], "required_evidence_complete": False, "negative_controls_complete": False, "leakage_audit_verdict": "BLOCKED"}
NEGATIVE_CONTROLS = {"schema_version": "1.0", "run_id": RUN_ID, "protocol_file_sha256": PROTOCOL_SHA, "all_frozen_controls_present": True, "controls": [{"control_id": cid, "protocol_section": "12", "expected_failure_mode": "synthetic expected failure", "observed_status": "FAIL", "expected_failure_observed": True, "post_hoc_mutation": False, "evidence_path": f"controls/{cid.lower()}.json", "evidence_sha256": SHA, "timestamp_utc": "2026-01-01T00:00:00Z", "reason_code": "EXPECTED_FAILURE_OBSERVED"} for cid in CONTROL_IDS]}
RAW_OUTPUT = {"schema_version": "1.0", "run_id": RUN_ID, "protocol_file_sha256": PROTOCOL_SHA, "outcome_count_expected": 2, "outcome_count_recorded": 2, "outcomes_dropped": False, "exclusions": [], "arrays": [{"identifier": "synthetic_a", "path": "raw/synthetic_a.npy", "format": "npy-little-endian-no-object", "dtype": "float64", "shape": [2], "byte_size": 16, "sha256": SHA, "protocol_section": "3.4", "adjudicative": False}, {"identifier": "synthetic_b", "path": "raw/synthetic_b.npy", "format": "npy-little-endian-no-object", "dtype": "complex128", "shape": [2], "byte_size": 32, "sha256": "b" * 64, "protocol_section": "3.4", "adjudicative": False}]}
EVIDENCE_PACKAGE = {"run_manifest": RUN_MANIFEST, "raw_output": RAW_OUTPUT, "negative_controls": NEGATIVE_CONTROLS, "machine_verdict": MACHINE_VERDICT, "evidence_directory_basename": RUN_ID}

def _case(name: str, schema: str, fixture: dict, layer: str, reason: str) -> dict:
    return {"name": name, "schema": schema, "fixture": fixture, "expected_layer": layer, "reason": reason}

def fixtures() -> tuple[dict[str, tuple[str, dict]], list[dict], dict[str, dict], list[dict]]:
    pass_verdict = deepcopy(MACHINE_VERDICT)
    pass_verdict.update(pipeline_reproduction_verdict="PASS", required_evidence_complete=True, negative_controls_complete=True, leakage_audit_verdict="PASS", reason_codes=["ALL_FROZEN_GATES_PASS"])
    run_upper = deepcopy(RUN_MANIFEST); run_upper["run_id"] = "QF0A_RUN_001"
    run_dotted = deepcopy(RUN_MANIFEST); run_dotted["run_id"] = "run.001"
    positives = {"valid_run_manifest": ("QF0A_RUN_MANIFEST.schema.json", deepcopy(RUN_MANIFEST)), "valid_run_id_upper_underscore": ("QF0A_RUN_MANIFEST.schema.json", run_upper), "valid_run_id_dotted": ("QF0A_RUN_MANIFEST.schema.json", run_dotted), "valid_nonpass_machine_verdict": ("QF0A_MACHINE_VERDICT.schema.json", deepcopy(MACHINE_VERDICT)), "valid_pass_machine_verdict": ("QF0A_MACHINE_VERDICT.schema.json", pass_verdict), "valid_negative_controls": ("QF0A_NEGATIVE_CONTROL_RESULT.schema.json", deepcopy(NEGATIVE_CONTROLS)), "valid_raw_output": ("QF0A_RAW_OUTPUT.schema.json", deepcopy(RAW_OUTPUT))}
    bad: list[dict] = []
    def run(name, mutate, layer, reason):
        value = deepcopy(RUN_MANIFEST); mutate(value); bad.append(_case(name, "QF0A_RUN_MANIFEST.schema.json", value, layer, reason))
    run("run_missing_required", lambda x: x.pop("environment"), "schema", "missing required field")
    run("run_wrong_protocol_hash", lambda x: x["protocol"].update(file_sha256=SHA), "schema", "wrong protocol hash")
    run("run_wrong_protocol_blob", lambda x: x["protocol"].update(git_blob_sha="d" * 40), "schema", "wrong protocol blob")
    run("run_wrong_main", lambda x: x["protocol"].update(authoritative_main_sha="d" * 40), "schema", "wrong authoritative main")
    run("run_dirty", lambda x: x["implementation"].update(dirty=True), "schema", "dirty tree")
    run("run_bad_python", lambda x: x["environment"].update(python="CPython 3.11.9"), "schema", "invalid CPython")
    run("run_bad_numpy", lambda x: x["environment"].update(numpy_version="2.2.0"), "schema", "invalid NumPy")
    run("run_bad_scipy", lambda x: x["environment"].update(scipy_version="1.15.0"), "schema", "invalid SciPy")
    run("run_bad_timestamp", lambda x: x["execution"].update(timestamp_utc="not-a-time"), "schema", "malformed timestamp")
    for name, attack in (("parent_posix", "../outside"), ("parent_windows", "..\\outside"), ("drive_backslash", "C:\\outside"), ("drive_slash", "C:/outside"), ("absolute_posix", "/outside"), ("unc", "\\\\server\\share"), ("device", "\\\\?\\C:\\outside"), ("nested_parent", "raw/../outside.npy"), ("double_separator", "raw//file.npy")):
        run(f"run_path_{name}", lambda x, attack=attack: x["evidence"].update(human_report=attack), "both", f"reject {attack!r}")
    run("run_timestamp_utc_without_z", lambda x: x["execution"].update(timestamp_utc="2026-01-01T00:00:00+00:00"), "semantic", "UTC timestamp must use Z")
    run("run_timestamp_taipei_no_offset", lambda x: x["execution"].update(timestamp_asia_taipei="2026-01-01T08:00:00"), "both", "Taipei timestamp must be timezone-aware")
    run("run_timestamp_taipei_plus09", lambda x: x["execution"].update(timestamp_asia_taipei="2026-01-01T09:00:00+09:00"), "semantic", "Taipei offset must be +08:00")
    run("run_timestamp_different_instant", lambda x: x["execution"].update(timestamp_asia_taipei="2026-01-01T09:00:00+08:00"), "semantic", "timestamp pair must represent same instant")
    run("run_timestamp_malformed_taipei", lambda x: x["execution"].update(timestamp_asia_taipei="not-a-time"), "schema", "malformed Taipei timestamp")
    run("run_layout_manifests_environment", lambda x: x["evidence"].update(environment_manifest="manifests/environment.json"), "both", "noncanonical manifests layout")
    run("run_layout_other_raw", lambda x: x["evidence"].update(raw_output_index="other/raw_output.json"), "both", "noncanonical raw index filename")
    run("run_layout_parent_verdict", lambda x: x["evidence"].update(machine_verdict="../machine_verdict.json"), "both", "layout traversal")
    def verdict(name, mutate, reason):
        value = deepcopy(MACHINE_VERDICT); mutate(value); bad.append(_case(name, "QF0A_MACHINE_VERDICT.schema.json", value, "schema", reason))
    verdict("verdict_unknown", lambda x: x.update(pipeline_reproduction_verdict="UNKNOWN"), "unknown verdict")
    for field, value in (("required_evidence_complete", False), ("negative_controls_complete", False), ("leakage_audit_verdict", "FAIL")):
        def mutate(x, field=field, value=value):
            x.update(pipeline_reproduction_verdict="PASS", required_evidence_complete=True, negative_controls_complete=True, leakage_audit_verdict="PASS")
            x[field] = value
        verdict(f"verdict_pass_bad_{field}", mutate, f"PASS prerequisite {field}")
    verdict("verdict_invalid_foundation", lambda x: x.update(foundation_model_verdict="SUPPORTED"), "Foundation support forbidden")
    verdict("verdict_established_foundation", lambda x: x.update(foundation_model_verdict="ESTABLISHED"), "Foundation establishment forbidden")
    def controls(name, mutate, layer, reason):
        value = deepcopy(NEGATIVE_CONTROLS); mutate(value); bad.append(_case(name, "QF0A_NEGATIVE_CONTROL_RESULT.schema.json", value, layer, reason))
    controls("controls_one_claims_complete", lambda x: x.update(controls=x["controls"][:1]), "both", "one control")
    controls("controls_five", lambda x: x.update(controls=x["controls"][:5]), "both", "five controls")
    controls("controls_duplicate_id", lambda x: x["controls"][5].update(control_id=x["controls"][0]["control_id"]), "semantic", "duplicate ID")
    controls("controls_unknown_replacement", lambda x: x["controls"][5].update(control_id="UNKNOWN_CONTROL"), "both", "unknown ID")
    controls("controls_duplicate_evidence_path", lambda x: x["controls"][5].update(evidence_path=x["controls"][0]["evidence_path"]), "semantic", "duplicate evidence path")
    controls("controls_missing_timestamp", lambda x: x["controls"][0].pop("timestamp_utc"), "schema", "missing timestamp")
    controls("controls_invalid_timestamp", lambda x: x["controls"][0].update(timestamp_utc="yesterday"), "schema", "invalid timestamp")
    controls("controls_nonutc_timestamp", lambda x: x["controls"][0].update(timestamp_utc="2026-01-01T08:00:00+08:00"), "semantic", "timestamp_utc must use Z")
    controls("controls_path_traversal", lambda x: x["controls"][0].update(evidence_path="controls/../outside.json"), "both", "path traversal")
    controls("controls_post_hoc", lambda x: x["controls"][0].update(post_hoc_mutation=True), "schema", "post-hoc mutation")
    controls("controls_unexpected_pass", lambda x: x["controls"][0].update(observed_status="PASS"), "semantic", "unexpected PASS")
    controls("controls_expected_failure_false", lambda x: x["controls"][0].update(expected_failure_observed=False), "semantic", "failure not observed")
    def raw(name, mutate, layer, reason):
        value = deepcopy(RAW_OUTPUT); mutate(value); bad.append(_case(name, "QF0A_RAW_OUTPUT.schema.json", value, layer, reason))
    raw("raw_parent_traversal", lambda x: x["arrays"][0].update(path="raw/../outside.npy"), "both", "raw traversal")
    raw("raw_backslash", lambda x: x["arrays"][0].update(path="raw\\file.npy"), "both", "raw backslash")
    raw("raw_absolute", lambda x: x["arrays"][0].update(path="/raw/file.npy"), "both", "raw absolute")
    raw("raw_duplicate_identifier", lambda x: x["arrays"][1].update(identifier=x["arrays"][0]["identifier"]), "semantic", "duplicate identifier")
    raw("raw_duplicate_path", lambda x: x["arrays"][1].update(path=x["arrays"][0]["path"]), "semantic", "duplicate path")
    raw("raw_count_mismatch", lambda x: x.update(outcome_count_recorded=1), "semantic", "count mismatch")
    raw("raw_dropped", lambda x: x.update(outcomes_dropped=True), "both", "outcome dropped")
    raw("raw_invalid_dtype", lambda x: x["arrays"][0].update(dtype="float32"), "schema", "invalid dtype")
    raw("raw_zero_dimension", lambda x: x["arrays"][0].update(shape=[0]), "schema", "zero dimension")
    raw("raw_invalid_sha", lambda x: x["arrays"][0].update(sha256="bad"), "schema", "invalid SHA")
    raw("raw_nonempty_exclusions", lambda x: x.update(exclusions=[{"identifier": "x"}]), "schema", "exclusions forbidden")

    schema_bases = {
        "QF0A_RUN_MANIFEST.schema.json": RUN_MANIFEST,
        "QF0A_MACHINE_VERDICT.schema.json": MACHINE_VERDICT,
        "QF0A_RAW_OUTPUT.schema.json": RAW_OUTPUT,
        "QF0A_NEGATIVE_CONTROL_RESULT.schema.json": NEGATIVE_CONTROLS,
    }
    run_id_attacks = (".", "..", "../escape", "..\\escape", "run/child", "run\\child", "C:", "C:/run", "\\\\server", "\\\\?\\C:\\run", "-valid?")
    for schema_name, base in schema_bases.items():
        for index, attack in enumerate(run_id_attacks):
            value = deepcopy(base); value["run_id"] = attack
            bad.append(_case(f"run_id_attack_{PathName(schema_name)}_{index}", schema_name, value, "both", f"reject canonical run_id attack {attack!r}"))

    package_positives = {"valid_cross_artifact_run_identity": deepcopy(EVIDENCE_PACKAGE)}
    package_bad: list[dict] = []
    for name, artifact in (("package_raw_run_id_mismatch", "raw_output"), ("package_controls_run_id_mismatch", "negative_controls"), ("package_verdict_run_id_mismatch", "machine_verdict")):
        value = deepcopy(EVIDENCE_PACKAGE); value[artifact]["run_id"] = "run-B"
        package_bad.append({"name": name, "fixture": value, "reason": f"{artifact} run_id mismatch"})
    value = deepcopy(EVIDENCE_PACKAGE); value["evidence_directory_basename"] = "run-B"
    package_bad.append({"name": "package_directory_basename_mismatch", "fixture": value, "reason": "evidence directory basename mismatch"})
    return positives, bad, package_positives, package_bad


def PathName(schema_name: str) -> str:
    return schema_name.split(".", 1)[0].lower()
