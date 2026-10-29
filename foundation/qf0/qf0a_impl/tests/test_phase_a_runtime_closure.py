"""Phase-A-only readiness and observation tests. No runner execution or science."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "foundation/qf0/qf0a_impl/src"))

from qf0a.provenance import MANIFEST_PATH, ProvenanceViolation, verify_execution_head_against_source_manifest
from qf0a.runner import observe_runtime_environment
from qf0a.audit import validate_static_separation


class PhaseARuntimeClosureTests(unittest.TestCase):
    def test_real_repository_head_is_explicitly_pre_refreeze(self):
        data = subprocess.check_output(["git", "show", f"HEAD:{MANIFEST_PATH}"], cwd=ROOT)
        manifest = json.loads(data)
        by_path = {item["path"]: item for item in manifest["files"]}
        authorization = {
            "source_manifest_sha256": hashlib.sha256(data).hexdigest(),
            "source_baseline_commit": manifest["source_baseline_commit"],
            "runner_blob": by_path["foundation/qf0/qf0a_impl/src/qf0a/runner.py"]["git_blob"],
            "protocol_blob": by_path["foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.1.md"]["git_blob"],
            "protocol_sha256": by_path["foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.1.md"]["sha256"],
            "preregistration_blob": by_path["foundation/qf0/QF0A_IMPLEMENTATION_PREREGISTRATION_v1.1.md"]["git_blob"],
            "preregistration_sha256": by_path["foundation/qf0/QF0A_IMPLEMENTATION_PREREGISTRATION_v1.1.md"]["sha256"],
            "environment_sha256": by_path["foundation/qf0/qf0a_impl/environment/QF0A_ENVIRONMENT_FREEZE_v1.1.json"]["sha256"],
        }
        with self.assertRaisesRegex(ProvenanceViolation, "not an ancestor"):
            verify_execution_head_against_source_manifest(ROOT, authorization)

    def test_runtime_mismatch_fails_closed(self):
        expected = json.loads((ROOT / "foundation/qf0/qf0a_impl/environment/QF0A_ENVIRONMENT_FREEZE_v1.1.json").read_text())
        with mock.patch("platform.python_version", return_value="3.12.9"):
            with self.assertRaisesRegex(RuntimeError, "python version"):
                observe_runtime_environment(expected)

    def test_thread_mismatch_fails_closed(self):
        expected = json.loads((ROOT / "foundation/qf0/qf0a_impl/environment/QF0A_ENVIRONMENT_FREEZE_v1.1.json").read_text())
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS":"2"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "OMP_NUM_THREADS"):
                observe_runtime_environment(expected)

    def test_numpy_scipy_and_base_executable_mismatches_fail_closed(self):
        base = json.loads((ROOT / "foundation/qf0/qf0a_impl/environment/QF0A_ENVIRONMENT_FREEZE_v1.1.json").read_text())
        cases = (("NumPy version", lambda item: item["packages"][0].update(version="0.0")),
                 ("SciPy version", lambda item: item["packages"][1].update(version="0.0")),
                 ("base executable SHA-256", lambda item: item["python"].update(base_executable_sha256="0" * 64)))
        for label, mutate in cases:
            with self.subTest(label=label):
                expected = json.loads(json.dumps(base)); mutate(expected)
                with self.assertRaisesRegex(RuntimeError, label):
                    observe_runtime_environment(expected)

    def test_leakage_audit_executes_static_validation(self):
        errors = validate_static_separation(ROOT / "foundation/qf0/qf0a_impl/src/qf0a")
        self.assertEqual(errors, [])
        runner = (ROOT / "foundation/qf0/qf0a_impl/src/qf0a/runner.py").read_text(encoding="utf-8")
        self.assertNotIn('"verdict": "PASS", "scope": "RUNTIME_PROVENANCE"', runner)

    def test_no_qual001_numeric_fixture(self):
        source = Path(__file__).read_text(encoding="utf-8")
        self.assertNotIn("QF0A-v1.1-" + "QUAL-001", source)


if __name__ == "__main__":
    unittest.main()
