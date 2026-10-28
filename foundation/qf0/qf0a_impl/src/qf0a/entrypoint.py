"""Only supported QF-0A execution entry point: validate committed source before imports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from .provenance import assert_fresh_official_process, verify_execution_head_against_source_manifest

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    args = parser.parse_args()
    assert_fresh_official_process()
    schema_path = args.repo_root / "foundation/qf0/qf0a_impl/schemas/QF0A_EXECUTION_AUTHORIZATION.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(authorization)
    seal = verify_execution_head_against_source_manifest(args.repo_root, authorization)
    from .runner import FrozenQF0ARunner
    FrozenQF0ARunner(args.repo_root, args.authorization, args.evidence_root, seal).execute()

if __name__ == "__main__":
    main()
