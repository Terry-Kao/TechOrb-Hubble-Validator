"""Machine-verdict projection from frozen adjudication only."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .adjudication import AdjudicationRecord, derive_pipeline_verdict

class FoundationVerdict(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"

@dataclass(frozen=True)
class VerdictRecord:
    run_id: str
    pipeline_reproduction_verdict: str
    foundation_model_verdict: FoundationVerdict
    reason_codes: tuple[str, ...]

def derive_machine_verdict(run_id: str, adjudication_record: AdjudicationRecord, foundation: FoundationVerdict = FoundationVerdict.NOT_APPLICABLE) -> VerdictRecord:
    if not run_id:
        raise ValueError("run_id is required")
    decision = derive_pipeline_verdict(adjudication_record)
    return VerdictRecord(run_id, decision.verdict.value, foundation, decision.reason_codes)
