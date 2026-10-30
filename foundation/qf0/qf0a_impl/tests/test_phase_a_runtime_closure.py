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

from qf0a.provenance import ExecutionSourceSeal, MANIFEST_PATH, ProvenanceViolation, verify_execution_head_against_source_manifest
from qf0a.runner import (build_run_manifest, normalize_blas_runtime_identity,
                         observe_runtime_environment, validate_blas_runtime_identity)
from qf0a.audit import validate_static_separation


class PhaseARuntimeClosureTests(unittest.TestCase):
    @staticmethod
    def blas_config(*, version="0.3.27", ilp64=True, dynamic=True, target="Haswell", backend="scipy-openblas"):
        flags = ["OpenBLAS", version]
        if ilp64:
            flags.append("USE64BITINT")
        if dynamic:
            flags.append("DYNAMIC_ARCH")
        flags.extend(["NO_AFFINITY", target, "MAX_THREADS=64"])
        item = {"found": True, "name": backend, "version": version,
                "has ilp64": ilp64, "openblas configuration": " ".join(flags)}
        return {"Build Dependencies": {"blas": dict(item), "lapack": dict(item)}}

    def test_valid_numpy_and_scipy_blas_identities(self):
        numpy_identity = normalize_blas_runtime_identity("numpy", self.blas_config())
        scipy_identity = normalize_blas_runtime_identity(
            "scipy", self.blas_config(version="0.3.27.dev", ilp64=False, target="Zen"))
        validate_blas_runtime_identity(numpy_identity, "scipy-openblas 0.3.27 ILP64 DYNAMIC_ARCH Haswell")
        validate_blas_runtime_identity(scipy_identity, "scipy-openblas 0.3.27.dev DYNAMIC_ARCH Zen")
        self.assertEqual(numpy_identity.integer_interface, "ILP64")
        self.assertEqual(scipy_identity.architecture_target, "zen")

    def test_same_version_wrong_numpy_build_components_reject(self):
        cases = (
            ("integer_interface", self.blas_config(ilp64=False)),
            ("dynamic_arch", self.blas_config(dynamic=False)),
            ("architecture_target", self.blas_config(target="Zen")),
        )
        for field, config in cases:
            with self.subTest(field=field):
                identity = normalize_blas_runtime_identity("numpy", config)
                with self.assertRaisesRegex(RuntimeError, field):
                    validate_blas_runtime_identity(identity, "scipy-openblas 0.3.27 ILP64 DYNAMIC_ARCH Haswell")

    def test_same_version_wrong_scipy_build_components_reject(self):
        for field, config in (
            ("dynamic_arch", self.blas_config(version="0.3.27.dev", ilp64=False, dynamic=False, target="Zen")),
            ("architecture_target", self.blas_config(version="0.3.27.dev", ilp64=False, target="Haswell")),
        ):
            with self.subTest(field=field):
                identity = normalize_blas_runtime_identity("scipy", config)
                with self.assertRaisesRegex(RuntimeError, field):
                    validate_blas_runtime_identity(identity, "scipy-openblas 0.3.27.dev DYNAMIC_ARCH Zen")

    def test_wrong_backend_and_unparseable_identity_reject(self):
        wrong = normalize_blas_runtime_identity("numpy", self.blas_config(backend="mkl"))
        with self.assertRaisesRegex(RuntimeError, "backend"):
            validate_blas_runtime_identity(wrong, "scipy-openblas 0.3.27 ILP64 DYNAMIC_ARCH Haswell")
        for config in ({}, {"Build Dependencies": {"blas": {"found": False}}}):
            with self.assertRaisesRegex(RuntimeError, "not observable|missing or unparseable"):
                normalize_blas_runtime_identity("numpy", config)

    def test_run_manifest_blas_identity_is_observed_input(self):
        environment = json.loads((ROOT / "foundation/qf0/qf0a_impl/environment/QF0A_ENVIRONMENT_FREEZE_v1.1.json").read_text())
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        seal = ExecutionSourceSeal(head, "a" * 40, "b" * 64, "c" * 40, tuple())
        observed = {
            "os":"observed-os", "kernel":"observed-kernel", "cpu_architecture":"AMD64", "cpu_model":"observed-cpu",
            "implementation":"CPython", "version":"3.12.10", "python_executable":"observed-python.exe",
            "numpy_version":"2.1.3", "scipy_version":"1.14.1", "numeric_dtypes":["float64","complex128"],
            "thread_environment": environment["threads"],
            "blas_lapack_observation": {
                "numpy":{"backend":"scipy-openblas","canonical":"OBSERVED-NUMPY-CANONICAL"},
                "scipy":{"backend":"scipy-openblas","canonical":"OBSERVED-SCIPY-CANONICAL"},
            },
        }
        document = build_run_manifest(ROOT, "synthetic-observation", {}, environment, "d" * 64, seal, observed)
        recorded = document["environment"]["blas_lapack"]
        self.assertEqual(recorded["vendor"], "scipy-openblas")
        self.assertEqual(recorded["version_or_build"],
                         "numpy[OBSERVED-NUMPY-CANONICAL]|scipy[OBSERVED-SCIPY-CANONICAL]")
        self.assertNotEqual(recorded["version_or_build"], environment["blas_lapack"]["scipy"])
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
        self.assertIn('"verified_modules": list(module_origins)', runner)

    def test_no_qual001_numeric_fixture(self):
        source = Path(__file__).read_text(encoding="utf-8")
        self.assertNotIn("QF0A-v1.1-" + "QUAL-001", source)


if __name__ == "__main__":
    unittest.main()
