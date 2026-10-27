"""Verdict contract construction only; no scientific adjudication is performed here."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

class PipelineVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    INVALIDATED = "INVALIDATED"

class FoundationVerdict(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"

@dataclass(frozen=True)
class VerdictRecord:
    run_id: str
    pipeline_reproduction_verdict: PipelineVerdict
    foundation_model_verdict: FoundationVerdict
    reason_codes: tuple[str, ...]

def construct_verdict(run_id: str, pipeline: PipelineVerdict, foundation: FoundationVerdict, reason_codes: tuple[str, ...]) -> VerdictRecord:
    if not run_id or not reason_codes:
        raise ValueError("run_id and reason_codes are required")
    return VerdictRecord(run_id, pipeline, foundation, reason_codes)
