"""Synthetic-only evidence and canonical-byte tests."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "foundation/qf0/qf0a_impl/src"))

from qf0a.audit import canonical_committed_identity
from qf0a.evidence import FIXED_RAW_ARRAYS, expected_relative_paths, prepare_new_evidence_root, validate_complete_inventory, write_array_once, register_existing_array, write_json_once
from qf0a.runner import build_run_manifest

class EvidenceContractTests(unittest.TestCase):
    def test_canonical_git_bytes_ignore_checkout_newlines(self):
        path = "foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.1.md"
        identity = canonical_committed_identity(ROOT, "902bdacd0e556d2a19079ab46c6c44d17f665372", path)
        committed = subprocess.check_output(["git","show",f"902bdacd0e556d2a19079ab46c6c44d17f665372:{path}"], cwd=ROOT)
        crlf_diagnostic = hashlib.sha256(committed.replace(b"\n", b"\r\n")).hexdigest()
        self.assertEqual(identity["sha256"], "1cf9f7f3a8cfa6a5b0e12a3290d7e545f44586f9359aaa5367754415cdcc5f3a")
        self.assertNotEqual(crlf_diagnostic, identity["sha256"])

    def test_fixed_inventory(self):
        self.assertEqual(len(FIXED_RAW_ARRAYS), 9)
        self.assertEqual(len(expected_relative_paths()), 22)
        self.assertIn("human_report.md", expected_relative_paths())
        self.assertNotIn("human_report_metadata.json", expected_relative_paths())

    def test_raw_array_write_once_and_hash_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            meta = write_array_once(root, "conditional_probabilities.npy", np.asarray([0.5,0.5], dtype=np.float64))
            self.assertEqual(register_existing_array(root, "conditional_probabilities.npy", meta["sha256"])["sha256"], meta["sha256"])
            with self.assertRaises(FileExistsError):
                write_array_once(root, "conditional_probabilities.npy", np.asarray([1.0], dtype=np.float64))
            (root / "conditional_probabilities.npy").write_bytes(b"mutated")
            with self.assertRaises(RuntimeError): register_existing_array(root, "conditional_probabilities.npy", meta["sha256"])

    def test_source_manifest_sha_is_exact_written_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source_manifest.json"
            write_json_once(path, {"z":1,"a":2})
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_runner_generated_manifest_validates_frozen_schema(self):
        environment = json.loads((ROOT / "foundation/qf0/qf0a_impl/environment/QF0A_ENVIRONMENT_FREEZE_v1.1.json").read_text())
        document = build_run_manifest(ROOT, "synthetic-manifest-only", {"source_baseline_commit":"a"*40}, environment, "b"*64)
        schema = json.loads((ROOT / "foundation/qf0/qf0a_impl/schemas/QF0A_RUN_MANIFEST.schema.json").read_text())
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(document)

    def test_empty_root_and_completeness(self):
        root = ROOT / ".qf0a-synthetic-evidence-test"
        if root.exists():
            import shutil; shutil.rmtree(root)
        try:
            prepare_new_evidence_root(root)
            with self.assertRaises(RuntimeError): validate_complete_inventory(root)
            with self.assertRaises(RuntimeError): prepare_new_evidence_root(root)
        finally:
            if root.exists():
                import shutil; shutil.rmtree(root)

if __name__ == "__main__": unittest.main()
