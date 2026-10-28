"""Result-independent v1.1 adjudication. Pipeline verdict is derived internally."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .controls import ControlId
from .protocol_binding import FROZEN_THRESHOLDS

@dataclass(frozen=True)
class ThresholdResult:
    name: str
    observed: float | None
    relation: str
    limit: float
    passed: bool
    scope: str
    reason_code: str

    def __post_init__(self) -> None:
        if self.name not in FROZEN_THRESHOLDS:
            raise ValueError(f"unknown frozen threshold: {self.name}")
        if self.limit != FROZEN_THRESHOLDS[self.name]:
            raise ValueError(f"limit mismatch for {self.name}")

class DerivedPipelineVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    INVALIDATED = "INVALIDATED"

class ControlObservation(str, Enum):
    EXPECTED_FAILURE = "EXPECTED_FAILURE"
    UNEXPECTED_PASS = "UNEXPECTED_PASS"
    EXECUTION_EXCEPTION = "EXECUTION_EXCEPTION"
    MISSING_OUTPUT = "MISSING_OUTPUT"
    IMPLEMENTATION_CRASH = "IMPLEMENTATION_CRASH"
    PROVENANCE_VIOLATION = "PROVENANCE_VIOLATION"

EXPECTED_CONTROL_REASON = {
    ControlId.BROKEN_CONSTRAINT: "CONSTRAINT_RESIDUAL_EXPECTED_FAILURE",
    ControlId.PRODUCT_NO_HISTORY: "ZERO_PROBABILITY_OR_ORBIT_ABSENCE_EXPECTED_FAILURE",
    ControlId.WRONG_CONDITIONING: "ORBIT_OR_STEP_MISMATCH_EXPECTED_FAILURE",
    ControlId.HIDDEN_EXTERNAL_TIME: "CONSTRUCTION_ORIGIN_ATTESTATION_REJECTION",
    ControlId.CLOCK_SCRAMBLE: "SCRAMBLED_REFERENCE_MISMATCH_EXPECTED_FAILURE",
    ControlId.POST_HOC_THRESHOLD_MUTATION: "IDENTITY_MUTATION_INVALIDATION",
}

@dataclass(frozen=True)
class AdjudicationRecord:
    provenance_contaminated: bool
    post_hoc_mutation: bool
    identity_mismatch: bool
    required_capability_complete: bool
    required_evidence_complete: bool
    required_measurements_complete: bool
    threshold_results: tuple[ThresholdResult, ...]
    control_observations: tuple[tuple[ControlId, ControlObservation, str], ...]
    leakage_audit_pass: bool

@dataclass(frozen=True)
class AdjudicationDecision:
    verdict: DerivedPipelineVerdict
    reason_codes: tuple[str, ...]

def derive_pipeline_verdict(record: AdjudicationRecord) -> AdjudicationDecision:
    reasons: list[str] = []
    if record.provenance_contaminated: reasons.append("PROVENANCE_CONTAMINATION")
    if record.post_hoc_mutation: reasons.append("POST_HOC_MUTATION")
    if record.identity_mismatch: reasons.append("FROZEN_IDENTITY_MISMATCH")
    controls = {item[0]: item for item in record.control_observations}
    for control, observation, reason in record.control_observations:
        if control is ControlId.POST_HOC_THRESHOLD_MUTATION and observation is ControlObservation.PROVENANCE_VIOLATION:
            reasons.append(reason)
    if reasons:
        return AdjudicationDecision(DerivedPipelineVerdict.INVALIDATED, tuple(sorted(set(reasons))))
    if not record.required_capability_complete: reasons.append("REQUIRED_CAPABILITY_MISSING")
    if not record.required_evidence_complete: reasons.append("REQUIRED_EVIDENCE_MISSING")
    if not record.required_measurements_complete: reasons.append("REQUIRED_MEASUREMENT_MISSING")
    names = tuple(item.name for item in record.threshold_results)
    if len(names) != len(set(names)): reasons.append("THRESHOLD_RESULT_DUPLICATE")
    if set(names) != set(FROZEN_THRESHOLDS): reasons.append("THRESHOLD_INVENTORY_INCOMPLETE")
    if not record.leakage_audit_pass: reasons.append("LEAKAGE_AUDIT_NOT_PASS")
    if set(controls) != set(ControlId): reasons.append("CONTROL_INVENTORY_INCOMPLETE")
    for control in ControlId:
        if control not in controls: continue
        observation = controls[control][1]
        if observation in {ControlObservation.EXECUTION_EXCEPTION, ControlObservation.MISSING_OUTPUT, ControlObservation.IMPLEMENTATION_CRASH}:
            reasons.append(f"{control.value}_{observation.value}")
    if reasons:
        return AdjudicationDecision(DerivedPipelineVerdict.BLOCKED, tuple(sorted(set(reasons))))
    failures = [item.reason_code for item in record.threshold_results if not item.passed]
    for control, observation, reason in record.control_observations:
        if observation is ControlObservation.UNEXPECTED_PASS:
            failures.append(f"{control.value}_UNEXPECTED_PASS")
        elif observation is not ControlObservation.EXPECTED_FAILURE:
            failures.append(reason)
        elif reason != EXPECTED_CONTROL_REASON[control]:
            failures.append(f"{control.value}_WRONG_REASON")
    if failures:
        return AdjudicationDecision(DerivedPipelineVerdict.FAIL, tuple(sorted(set(failures))))
    return AdjudicationDecision(DerivedPipelineVerdict.PASS, ("ALL_FROZEN_QUALIFICATION_GATES_PASS",))
