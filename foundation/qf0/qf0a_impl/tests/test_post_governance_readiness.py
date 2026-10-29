"""Synthetic execution-context metadata tests; no QF-0A runner execution occurs."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "foundation/qf0/qf0a_impl/src"))

from qf0a.provenance import ExecutionSourceSeal
from qf0a.runner import build_run_manifest

SCHEMA_PATH = ROOT / "foundation/qf0/qf0a_impl/schemas/QF0A_RUN_MANIFEST.schema.json"
ENVIRONMENT_PATH = ROOT / "foundation/qf0/qf0a_impl/environment/QF0A_ENVIRONMENT_FREEZE_v1.1.json"
STALE_MAIN = "a9e9a8ec0b637d4a724a6d3baaded83b54c0058e"
STALE_BRANCH = "qf0a/v1.1-pre-execution-repair-refreeze"


class PostGovernanceRunManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".qf0a-context-", dir=ROOT)
        self.repo = Path(self.temp.name)
        subprocess.check_call(["git", "init", "-q"], cwd=self.repo)
        subprocess.check_call(["git", "config", "user.email", "synthetic@example.invalid"], cwd=self.repo)
        subprocess.check_call(["git", "config", "user.name", "Synthetic Context Test"], cwd=self.repo)
        (self.repo / "README.md").write_text("synthetic\n", encoding="utf-8")
        subprocess.check_call(["git", "add", "README.md"], cwd=self.repo)
        subprocess.check_call(["git", "commit", "-qm", "synthetic execution head"], cwd=self.repo)
        self.head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()
        self.branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=self.repo, text=True).strip()
        self.seal = ExecutionSourceSeal(self.head, self.head, "1" * 64, "2" * 40, tuple())
        self.source_manifest = {"source_baseline_commit": self.head}
        self.environment = json.loads(ENVIRONMENT_PATH.read_text(encoding="utf-8"))
        self.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        return build_run_manifest(
            self.repo, "synthetic-context", self.source_manifest,
            self.environment, "3" * 64, self.seal,
        )

    def assert_schema_valid(self, document):
        Draft202012Validator(self.schema, format_checker=FormatChecker()).validate(document)

    def test_validated_execution_head_is_authoritative_metadata(self):
        document = self.build()
        self.assertEqual(document["protocol"]["authoritative_main_sha"], self.head)
        self.assertEqual(document["implementation"]["commit_sha"], self.head)
        self.assertEqual(document["implementation"]["current_execution_head"], self.head)
        self.assertNotIn(STALE_MAIN, json.dumps(document, sort_keys=True))
        self.assert_schema_valid(document)

    def test_attached_branch_is_truthful(self):
        document = self.build()
        self.assertEqual(document["implementation"]["branch"], self.branch)
        self.assertNotEqual(document["implementation"]["branch"], STALE_BRANCH)
        self.assert_schema_valid(document)

    def test_detached_head_is_truthful(self):
        subprocess.check_call(["git", "checkout", "--detach", "-q", self.head], cwd=self.repo)
        document = self.build()
        self.assertEqual(document["implementation"]["branch"], "DETACHED")
        self.assert_schema_valid(document)

    def test_stale_repair_branch_is_not_generated(self):
        self.assertNotIn(STALE_BRANCH, json.dumps(self.build(), sort_keys=True))

    def test_mismatched_unvalidated_head_is_rejected(self):
        wrong = ExecutionSourceSeal("f" * 40, self.head, "1" * 64, "2" * 40, tuple())
        with self.assertRaisesRegex(RuntimeError, "validated source seal"):
            build_run_manifest(self.repo, "synthetic-context", self.source_manifest, self.environment, "3" * 64, wrong)


if __name__ == "__main__":
    unittest.main()
