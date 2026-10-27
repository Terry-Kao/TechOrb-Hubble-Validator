"""Result-independent v1.1 adjudication. Pipeline verdict is derived internally."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .controls import ControlId

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
    threshold_checks: tuple[bool, ...]
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
    if not record.leakage_audit_pass: reasons.append("LEAKAGE_AUDIT_NOT_PASS")
    if set(controls) != set(ControlId): reasons.append("CONTROL_INVENTORY_INCOMPLETE")
    for control in ControlId:
        if control not in controls: continue
        observation = controls[control][1]
        if observation in {ControlObservation.EXECUTION_EXCEPTION, ControlObservation.MISSING_OUTPUT, ControlObservation.IMPLEMENTATION_CRASH}:
            reasons.append(f"{control.value}_{observation.value}")
    if reasons:
        return AdjudicationDecision(DerivedPipelineVerdict.BLOCKED, tuple(sorted(set(reasons))))
    failures = [f"THRESHOLD_{i}_FAILED" for i, passed in enumerate(record.threshold_checks) if not passed]
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
