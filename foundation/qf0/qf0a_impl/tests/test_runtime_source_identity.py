"""Synthetic Git-only provenance attacks; never imports or instantiates QF-0A science."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "foundation/qf0/qf0a_impl/src"))
from qf0a.provenance import MANIFEST_PATH, ProvenanceViolation, assert_fresh_official_process, verify_execution_head_against_source_manifest

RUNNER = "foundation/qf0/qf0a_impl/src/qf0a/runner.py"
PROTOCOL = "foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.1.md"
PREREG = "foundation/qf0/QF0A_IMPLEMENTATION_PREREGISTRATION_v1.1.md"
ENV = "foundation/qf0/qf0a_impl/environment/QF0A_ENVIRONMENT_FREEZE_v1.1.json"
SCHEMA = "foundation/qf0/qf0a_impl/schemas/QF0A_EXECUTION_AUTHORIZATION.schema.json"
ACCEPTED_GOVERNANCE = (
    "README.md",
    "control/DECISION_LOG.md",
    "control/EVIDENCE_INDEX.md",
    "control/PROJECT_STATE.md",
)

def git(root: Path, *args: str, text=False):
    return subprocess.check_output(["git", *args], cwd=root, text=text)

def write(root: Path, path: str, data: bytes) -> None:
    target = root / path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)

def commit(root: Path, message: str) -> str:
    subprocess.check_call(["git", "add", "."], cwd=root, stdout=subprocess.DEVNULL)
    subprocess.check_call(["git", "commit", "-m", message], cwd=root, stdout=subprocess.DEVNULL)
    return git(root, "rev-parse", "HEAD", text=True).strip()

def entries(root: Path, revision: str, paths: list[str]) -> list[dict[str, object]]:
    result=[]
    for path in paths:
        data=git(root,"show",f"{revision}:{path}")
        result.append({"path":path,"byte_length":len(data),"git_blob":git(root,"rev-parse",f"{revision}:{path}",text=True).strip(),"sha256":hashlib.sha256(data).hexdigest()})
    return result

class RuntimeSourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix=".qf0a-runtime-", dir=ROOT); self.repo=Path(self.temp.name)
        subprocess.check_call(["git","init","-q"],cwd=self.repo)
        subprocess.check_call(["git","config","core.autocrlf","false"],cwd=self.repo)
        subprocess.check_call(["git","config","user.email","synthetic@example.invalid"],cwd=self.repo)
        subprocess.check_call(["git","config","user.name","Synthetic Provenance Test"],cwd=self.repo)
        fixtures={RUNNER:b"RUNNER_FROZEN=True\n",PROTOCOL:b"protocol\n",PREREG:b"prereg\n",ENV:b"{}\n",SCHEMA:b"{}\n"}
        self.paths=list(fixtures)
        fixtures.update({path: b"governance baseline\n" for path in ACCEPTED_GOVERNANCE})
        for path,data in fixtures.items(): write(self.repo,path,data)
        self.baseline=commit(self.repo,"synthetic source baseline")
        self._write_manifest(self.baseline)
        self.freeze_head=commit(self.repo,"synthetic manifest freeze")
        manifest_bytes=git(self.repo,"show",f"HEAD:{MANIFEST_PATH}")
        by_path={item["path"]:item for item in json.loads(manifest_bytes)["files"]}
        self.authorization={"source_manifest_sha256":hashlib.sha256(manifest_bytes).hexdigest(),"source_baseline_commit":self.baseline,
          "runner_blob":by_path[RUNNER]["git_blob"],"protocol_blob":by_path[PROTOCOL]["git_blob"],"protocol_sha256":by_path[PROTOCOL]["sha256"],
          "preregistration_blob":by_path[PREREG]["git_blob"],"preregistration_sha256":by_path[PREREG]["sha256"],"environment_sha256":by_path[ENV]["sha256"]}
        verify_execution_head_against_source_manifest(self.repo,self.authorization)

    def tearDown(self): self.temp.cleanup()

    def _write_manifest(self, revision: str) -> None:
        payload={"schema_version":"1.1","source_baseline_commit":revision,"files":entries(self.repo,revision,self.paths)}
        write(self.repo,MANIFEST_PATH,(json.dumps(payload,sort_keys=True)+"\n").encode())

    def test_later_clean_source_commit_is_rejected(self):
        write(self.repo,RUNNER,b"RUNNER_FROZEN=False\n"); commit(self.repo,"clean attacker source mutation")
        self.assertEqual(git(self.repo,"status","--porcelain",text=True),"")
        with self.assertRaises(ProvenanceViolation): verify_execution_head_against_source_manifest(self.repo,self.authorization)

    def test_exact_accepted_post_baseline_governance_changes_pass(self):
        for path in ACCEPTED_GOVERNANCE:
            write(self.repo,path,b"accepted final governance\n")
        commit(self.repo,"accepted governance synchronization")
        seal=verify_execution_head_against_source_manifest(self.repo,self.authorization)
        self.assertEqual(seal.current_execution_head,git(self.repo,"rev-parse","HEAD",text=True).strip())

    def test_unlisted_governance_file_is_rejected_by_policy(self):
        write(self.repo,"control/UNAUTHORIZED.md",b"not allowlisted\n"); commit(self.repo,"unlisted governance change")
        with self.assertRaisesRegex(ProvenanceViolation,"outside governance allowlist"):
            verify_execution_head_against_source_manifest(self.repo,self.authorization)

    def test_active_source_schema_and_environment_mutations_remain_rejected(self):
        mutations=(
            (RUNNER,b"RUNNER_FROZEN=False\n"),
            (SCHEMA,b'{"changed":true}\n'),
            (ENV,b'{"changed":true}\n'),
        )
        for path,data in mutations:
            with self.subTest(path=path):
                subprocess.check_call(["git","reset","--hard",self.freeze_head],cwd=self.repo,stdout=subprocess.DEVNULL)
                write(self.repo,path,data); commit(self.repo,f"mutate {path}")
                with self.assertRaises(ProvenanceViolation):
                    verify_execution_head_against_source_manifest(self.repo,self.authorization)

    def test_readiness_check_does_not_import_scientific_modules(self):
        script = """
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
from qf0a.provenance import verify_execution_head_against_source_manifest
verify_execution_head_against_source_manifest(pathlib.Path(sys.argv[2]), json.loads(sys.argv[3]))
scientific={'qf0a.conditional_state','qf0a.controls','qf0a.metrics','qf0a.runner'}
assert scientific.isdisjoint(sys.modules), sorted(scientific & set(sys.modules))
"""
        subprocess.check_call([
            sys.executable, "-c", script,
            str(ROOT / "foundation/qf0/qf0a_impl/src"),
            str(self.repo), json.dumps(self.authorization),
        ])

    def test_added_alternate_active_module_is_rejected(self):
        write(self.repo,"foundation/qf0/qf0a_impl/src/qf0a/runner_alt.py",b"ALT=True\n"); commit(self.repo,"clean alternate module")
        self.assertEqual(git(self.repo,"status","--porcelain",text=True),"")
        with self.assertRaises(ProvenanceViolation): verify_execution_head_against_source_manifest(self.repo,self.authorization)

    def test_changed_source_and_self_consistent_manifest_rejected_by_old_authorization(self):
        write(self.repo,RUNNER,b"RUNNER_ATTACKER=True\n"); attacked=commit(self.repo,"attacker source")
        self._write_manifest(attacked); commit(self.repo,"attacker self-consistent manifest")
        with self.assertRaisesRegex(ProvenanceViolation,"not bound by authorization"):
            verify_execution_head_against_source_manifest(self.repo,self.authorization)

    def test_runner_blob_mutation_is_rejected_with_clean_tree(self):
        write(self.repo,RUNNER,b"RUNNER_FROZEN='different'\n"); commit(self.repo,"clean runner mutation")
        self.assertEqual(git(self.repo,"status","--porcelain",text=True),"")
        with self.assertRaises(ProvenanceViolation): verify_execution_head_against_source_manifest(self.repo,self.authorization)

    def test_fresh_process_gate_rejects_preloaded_scientific_module(self):
        sys.modules["qf0a.metrics"] = object()
        try:
            with self.assertRaisesRegex(ProvenanceViolation, "fresh process"):
                assert_fresh_official_process()
        finally:
            del sys.modules["qf0a.metrics"]

if __name__ == "__main__": unittest.main()
