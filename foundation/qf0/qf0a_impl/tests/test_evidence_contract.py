"""Synthetic-only evidence and canonical-byte tests."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "foundation/qf0/qf0a_impl/src"))

from qf0a.audit import canonical_committed_identity
from qf0a.evidence import FIXED_RAW_ARRAYS, expected_relative_paths, prepare_new_evidence_root, validate_complete_inventory

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
