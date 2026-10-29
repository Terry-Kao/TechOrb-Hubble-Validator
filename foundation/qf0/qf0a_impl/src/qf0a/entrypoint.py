"""Only supported QF-0A execution entry point: validate committed source before imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

def _bootstrap_bind_entrypoint(repo_root: Path) -> None:
    """Stdlib-only self-binding before importing project/runtime dependencies."""
    expected = (repo_root.resolve() / "foundation/qf0/qf0a_impl/src/qf0a")
    actual = Path(__file__).resolve()
    try:
        relative = actual.relative_to(expected).as_posix()
    except ValueError as exc:
        raise RuntimeError("entrypoint was imported outside the verified repository source") from exc
    repository_relative = f"foundation/qf0/qf0a_impl/src/qf0a/{relative}"
    committed = subprocess.check_output(["git", "show", f"HEAD:{repository_relative}"], cwd=repo_root)
    if hashlib.sha256(actual.read_bytes()).digest() != hashlib.sha256(committed).digest():
        raise RuntimeError("entrypoint bytes differ from committed HEAD")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    args = parser.parse_args()
    _bootstrap_bind_entrypoint(args.repo_root)
    from jsonschema import Draft202012Validator, FormatChecker
    from .provenance import (assert_fresh_official_process, verify_execution_head_against_source_manifest,
                             verify_loaded_qf0a_modules)
    assert_fresh_official_process()
    schema_path = args.repo_root / "foundation/qf0/qf0a_impl/schemas/QF0A_EXECUTION_AUTHORIZATION.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(authorization)
    seal = verify_execution_head_against_source_manifest(args.repo_root, authorization)
    verify_loaded_qf0a_modules(args.repo_root, seal.current_execution_head)
    from .runner import FrozenQF0ARunner
    verify_loaded_qf0a_modules(args.repo_root, seal.current_execution_head)
    FrozenQF0ARunner(args.repo_root, args.authorization, args.evidence_root, seal).execute()

if __name__ == "__main__":
    main()
