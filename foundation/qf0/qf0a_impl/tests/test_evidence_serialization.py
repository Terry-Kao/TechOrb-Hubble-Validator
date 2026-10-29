"""Synthetic integration coverage for runner evidence serialization and sealing."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "foundation/qf0/qf0a_impl/src"))

from qf0a.adjudication import AdjudicationRecord, ControlObservation, EXPECTED_CONTROL_REASON, ThresholdResult
from qf0a.controls import CONTROL_DEFINITIONS, ControlId
from qf0a.evidence import (FIXED_RAW_ARRAYS, expected_final_inventory,
    expected_preverdict_inventory, expected_relative_paths, prepare_new_evidence_root,
    serialize_machine_verdict, serialize_negative_controls, serialize_raw_output_index,
    validate_exact_final_inventory, validate_hash_manifest, write_array_once,
    write_json_once, write_source_manifest_bytes_with_sha, write_text_once)
from qf0a.protocol_binding import FROZEN_THRESHOLDS, PROTOCOL_SHA256
from qf0a.provenance import ExecutionSourceSeal
from qf0a.runner import FrozenQF0ARunner, build_run_manifest
from qf0a.verdict import derive_machine_verdict


def _schema(name: str) -> dict:
    return json.loads((ROOT / "foundation/qf0/qf0a_impl/schemas" / name).read_text(encoding="utf-8"))


def _validate(name: str, document: dict) -> None:
    Draft202012Validator(_schema(name), format_checker=FormatChecker()).validate(document)


def _record(**changes) -> AdjudicationRecord:
    thresholds = tuple(ThresholdResult(name, 0.0, "synthetic", limit, True, "SYNTHETIC", f"{name}_FAILED")
                       for name, limit in FROZEN_THRESHOLDS.items())
    controls = tuple((control, ControlObservation.EXPECTED_FAILURE, EXPECTED_CONTROL_REASON[control]) for control in ControlId)
    data = dict(provenance_contaminated=False, post_hoc_mutation=False, identity_mismatch=False,
                required_capability_complete=True, required_evidence_complete=True,
                required_measurements_complete=True, threshold_results=thresholds,
                control_observations=controls, leakage_audit_pass=True)
    data.update(changes)
    return AdjudicationRecord(**data)


class EvidenceSerializationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "synthetic-run"
        prepare_new_evidence_root(self.root)
        self.run_id = "synthetic-run"
        self.observations = _record().control_observations

    def tearDown(self):
        self.temp.cleanup()

    def _raw_index(self) -> tuple[dict, dict]:
        index = {}
        for offset, name in enumerate(FIXED_RAW_ARRAYS, start=1):
            index[name] = write_array_once(self.root / "raw", name, np.asarray([float(offset)], dtype=np.float64))
        document = serialize_raw_output_index(self.run_id, index)
        return index, document

    def _controls(self) -> dict:
        write_json_once(self.root / "diagnostics.json", {"synthetic": True})
        digest = hashlib.sha256((self.root / "diagnostics.json").read_bytes()).hexdigest()
        return serialize_negative_controls(self.run_id, self.observations, CONTROL_DEFINITIONS,
            "diagnostics.json", digest, datetime.now(timezone.utc).isoformat())

    def test_runner_serializers_validate_all_three_schemas(self):
        _, raw = self._raw_index()
        controls = self._controls()
        verdict = serialize_machine_verdict(derive_machine_verdict(self.run_id, _record()), _record())
        _validate("QF0A_RAW_OUTPUT.schema.json", raw)
        _validate("QF0A_NEGATIVE_CONTROL_RESULT.schema.json", controls)
        _validate("QF0A_MACHINE_VERDICT.schema.json", verdict)
        self.assertEqual([item["identifier"] + ".npy" for item in raw["arrays"]], list(FIXED_RAW_ARRAYS))
        self.assertEqual({item["control_id"] for item in controls["controls"]}, {item.value for item in ControlId})

    def _materialize_preverdict_package(self) -> tuple[FrozenQF0ARunner, dict, AdjudicationRecord]:
        index, raw = self._raw_index()
        write_json_once(self.root / "raw_output_index.json", raw)
        controls = self._controls()
        write_json_once(self.root / "negative_controls.json", controls)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        manifest_path = "foundation/qf0/qf0a_impl/manifests/QF0A_SOURCE_MANIFEST_v1.1.json"
        manifest_bytes = subprocess.check_output(["git", "show", f"{head}:{manifest_path}"], cwd=ROOT)
        source_sha = write_source_manifest_bytes_with_sha(self.root, manifest_bytes)
        environment = json.loads((ROOT / "foundation/qf0/qf0a_impl/environment/QF0A_ENVIRONMENT_FREEZE_v1.1.json").read_text())
        write_json_once(self.root / "environment_manifest.json", environment)
        seal = ExecutionSourceSeal(head, "a" * 40, source_sha, "b" * 40, tuple())
        run_manifest = build_run_manifest(ROOT, self.run_id, json.loads(manifest_bytes), environment, source_sha, seal)
        write_json_once(self.root / "run_manifest.json", run_manifest)
        authorization = {
            "authorization_version":"1.1", "run_id":self.run_id, "authorized":True,
            "protocol_blob":"3af303a471d99397429d33f77ca512578f40f972", "protocol_sha256":PROTOCOL_SHA256,
            "preregistration_blob":"a"*40, "preregistration_sha256":"b"*64,
            "environment_sha256":"c"*64, "source_manifest_sha256":source_sha,
            "source_baseline_commit":"d"*40, "runner_blob":"e"*40,
            "issuer":"synthetic-test", "governance_record":"synthetic-only",
            "timestamp_utc":"2026-10-29T00:00:00Z"
        }
        write_json_once(self.root / "execution_authorization.json", authorization)
        write_json_once(self.root / "leakage_audit.json", {"run_id":self.run_id,"verdict":"PASS","scope":"SYNTHETIC"})
        write_text_once(self.root / "human_report.md", "synthetic\n")
        write_text_once(self.root / "logs/run.log", "synthetic\n")
        runner = FrozenQF0ARunner.__new__(FrozenQF0ARunner)
        runner.repo_root, runner.evidence_root = ROOT, self.root
        runner.authorization, runner.run_id = authorization, self.run_id
        conditional = {"stage_b_identities": {name:index[name] for name in FIXED_RAW_ARRAYS[:2]}}
        return runner, conditional, _record(required_evidence_complete=False)

    def test_full_synthetic_package_passes_g2_and_final_seal(self):
        runner, conditional, provisional = self._materialize_preverdict_package()
        complete = runner._stage_g2_validate_evidence(conditional, provisional)
        self.assertTrue(complete.required_evidence_complete)
        runner._stage_g3_final_sealing(derive_machine_verdict(self.run_id, complete), complete)
        _validate("QF0A_MACHINE_VERDICT.schema.json", json.loads((self.root / "machine_verdict.json").read_text()))
        validate_exact_final_inventory(self.root)
        validate_hash_manifest(self.root)

    def _assert_g2_extra_blocked(self, relative_path: str, payload: bytes = b"unexpected\n") -> None:
        runner, conditional, provisional = self._materialize_preverdict_package()
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        result = runner._stage_g2_validate_evidence(conditional, provisional)
        self.assertFalse(result.required_evidence_complete)
        self.assertNotEqual(derive_machine_verdict(self.run_id, result).pipeline_reproduction_verdict, "PASS")

    def test_unexpected_root_file_rejected_at_g2(self):
        self._assert_g2_extra_blocked("unexpected.json", b"{}\n")

    def test_extra_raw_array_rejected_at_g2(self):
        extra = self.root / "raw" / "extra.npy"
        with extra.open("wb") as handle:
            np.save(handle, np.asarray([1.0], dtype=np.float64), allow_pickle=False)
        runner, conditional, provisional = self._materialize_preverdict_package_after_existing_raw()
        result = runner._stage_g2_validate_evidence(conditional, provisional)
        self.assertFalse(result.required_evidence_complete)

    def _materialize_preverdict_package_after_existing_raw(self):
        """Add an extra raw file after ordinary package materialization."""
        extra_bytes = (self.root / "raw" / "extra.npy").read_bytes()
        (self.root / "raw" / "extra.npy").unlink()
        package = self._materialize_preverdict_package()
        (self.root / "raw" / "extra.npy").write_bytes(extra_bytes)
        return package

    def test_extra_log_rejected_at_g2(self):
        self._assert_g2_extra_blocked("logs/extra.log")

    def test_unexpected_directory_rejected_at_g2(self):
        runner, conditional, provisional = self._materialize_preverdict_package()
        (self.root / "tmp").mkdir()
        result = runner._stage_g2_validate_evidence(conditional, provisional)
        self.assertFalse(result.required_evidence_complete)

    def _sealed_package(self) -> None:
        runner, conditional, provisional = self._materialize_preverdict_package()
        complete = runner._stage_g2_validate_evidence(conditional, provisional)
        self.assertTrue(complete.required_evidence_complete)
        runner._stage_g3_final_sealing(derive_machine_verdict(self.run_id, complete), complete)

    def test_final_package_with_23rd_file_rejected(self):
        self._sealed_package()
        (self.root / "unexpected.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            validate_exact_final_inventory(self.root)

    def test_correctly_hashed_extra_checksum_entry_still_rejected(self):
        self._sealed_package()
        extra = self.root / "unexpected.json"
        extra.write_text("{}\n", encoding="utf-8")
        digest = hashlib.sha256(extra.read_bytes()).hexdigest()
        with (self.root / "SHA256SUMS.txt").open("a", encoding="ascii", newline="\n") as handle:
            handle.write(f"{digest}  unexpected.json\n")
        with self.assertRaises(RuntimeError):
            validate_hash_manifest(self.root)

    def test_checksum_missing_expected_path_rejected(self):
        self._sealed_package()
        path = self.root / "SHA256SUMS.txt"
        lines = path.read_text(encoding="ascii").splitlines()
        path.write_text("\n".join(lines[1:]) + "\n", encoding="ascii", newline="\n")
        with self.assertRaises(RuntimeError):
            validate_hash_manifest(self.root)

    def test_checksum_duplicate_path_rejected(self):
        self._sealed_package()
        path = self.root / "SHA256SUMS.txt"
        first = path.read_text(encoding="ascii").splitlines()[0]
        with path.open("a", encoding="ascii", newline="\n") as handle:
            handle.write(first + "\n")
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            validate_hash_manifest(self.root)

    def test_checksum_hash_mismatch_rejected(self):
        self._sealed_package()
        path = self.root / "SHA256SUMS.txt"
        lines = path.read_text(encoding="ascii").splitlines()
        _, rel = lines[0].split("  ", 1)
        lines[0] = f"{'0' * 64}  {rel}"
        path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            validate_hash_manifest(self.root)

    def test_frozen_inventory_counts(self):
        self.assertEqual(len(expected_relative_paths()), 22)
        self.assertEqual(len(expected_final_inventory()), 22)
        self.assertEqual(len(expected_preverdict_inventory()), 20)

    def test_malformed_raw_or_controls_and_missing_evidence_block_pass(self):
        for target in ("raw_output_index.json", "negative_controls.json"):
            with self.subTest(target=target):
                self.tearDown(); self.setUp()
                runner, conditional, provisional = self._materialize_preverdict_package()
                document = json.loads((self.root / target).read_text())
                document.pop("schema_version")
                (self.root / target).write_text(json.dumps(document), encoding="utf-8")
                result = runner._stage_g2_validate_evidence(conditional, provisional)
                self.assertFalse(result.required_evidence_complete)
                self.assertNotEqual(derive_machine_verdict(self.run_id, result).pipeline_reproduction_verdict, "PASS")
        self.tearDown(); self.setUp()
        runner, conditional, provisional = self._materialize_preverdict_package()
        (self.root / "human_report.md").unlink()
        self.assertFalse(runner._stage_g2_validate_evidence(conditional, provisional).required_evidence_complete)

    def test_stage_b_mutation_invalidates(self):
        runner, conditional, provisional = self._materialize_preverdict_package()
        (self.root / "raw" / FIXED_RAW_ARRAYS[0]).write_bytes(b"mutated")
        result = runner._stage_g2_validate_evidence(conditional, provisional)
        self.assertTrue(result.post_hoc_mutation)
        self.assertEqual(derive_machine_verdict(self.run_id, result).pipeline_reproduction_verdict, "INVALIDATED")

    def test_malformed_machine_verdict_is_caught_by_final_seal(self):
        runner, conditional, provisional = self._materialize_preverdict_package()
        complete = runner._stage_g2_validate_evidence(conditional, provisional)
        with patch("qf0a.runner.serialize_machine_verdict", return_value={"run_id": self.run_id}):
            with self.assertRaises(Exception):
                runner._stage_g3_final_sealing(derive_machine_verdict(self.run_id, complete), complete)

    def test_source_manifest_exact_committed_bytes_policy(self):
        runner, conditional, provisional = self._materialize_preverdict_package()
        self.assertEqual(hashlib.sha256((self.root / "source_manifest.json").read_bytes()).hexdigest(),
                         runner.authorization["source_manifest_sha256"])
        (self.root / "source_manifest.json").write_bytes(b"{}\n")
        self.assertFalse(runner._stage_g2_validate_evidence(conditional, provisional).required_evidence_complete)

    def test_pass_impossible_before_contract_completion_and_no_caller_pass(self):
        incomplete = _record(required_evidence_complete=False)
        self.assertEqual(derive_machine_verdict(self.run_id, incomplete).pipeline_reproduction_verdict, "BLOCKED")
        self.assertNotIn("verdict", inspect.signature(derive_machine_verdict).parameters)
        source = inspect.getsource(FrozenQF0ARunner)
        self.assertNotIn("QF0A-v1.1-QUAL-001", source)


if __name__ == "__main__":
    unittest.main()
